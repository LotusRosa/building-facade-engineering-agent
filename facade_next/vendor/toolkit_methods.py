"""FP32 losses and transactional update steps for the BF16 candidate kernel.

This module never starts a run. The caller owns CUDA BF16 autocast and performs
all microbatch backward calls inside ``backward_pass``. No loss scaling is used.
SAM replays that callback from its original RNG state, retains only first-pass
buffer updates, and restores exact saved parameters before AdamW sees them.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


_STAT_KEYS = ("loss_sum", "draws", "supervised_loss_sum",
              "distillation_loss_sum", "teacher_draws")
_LOSS_KEYS = ("loss_sum", "supervised_loss_sum", "distillation_loss_sum")


def _finite(tensor, label):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Nonfinite {label}")


def _logit_shape(logits, label):
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError(f"{label} must be a floating-point tensor")
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"{label} must have shape [N,3]")


def supervised_loss(logits, targets, method, pos_weight):
    """Mean FP32 loss across images and three independent binary labels.

    ``method`` is the complete frozen method dictionary. ASL deliberately ignores
    the inherited ``pos_weight``. Its explicit ``detach_focal_weight`` setting
    distinguishes the official detached-focal-weight formulation from the
    differentiable weighting variant; neither policy is silently substituted.
    Setting both gammas and clip to zero gives unweighted BCE.
    """
    _logit_shape(logits, "logits")
    if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape:
        raise ValueError("targets must match logits shape [N,3]")
    if targets.device != logits.device or logits.shape[0] == 0:
        raise ValueError("loss requires a nonempty batch on one device")
    if not isinstance(method, Mapping) or not isinstance(method.get("loss"), Mapping):
        raise TypeError("method must contain its frozen loss dictionary")
    config = dict(method["loss"])
    name = config.get("name")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values, labels = logits.float(), targets.float()
        _finite(values, "student logits")
        _finite(labels, "targets")
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("targets must be the original binary labels")
        if name == "bce":
            if set(config) != {"name"}:
                raise ValueError("Unknown BCE loss configuration keys")
            if pos_weight is None:
                raise ValueError("BCE requires the frozen parent pos_weight")
            weights = torch.as_tensor(pos_weight, dtype=torch.float32, device=values.device)
            if weights.shape != (3,):
                raise ValueError("parent pos_weight must have shape [3]")
            _finite(weights, "parent pos_weight")
            if not bool((weights > 0).all()):
                raise ValueError("parent pos_weight must be positive")
            loss = F.binary_cross_entropy_with_logits(values, labels, pos_weight=weights)
        elif name == "asl":
            expected = {"name", "gamma_pos", "gamma_neg", "clip", "detach_focal_weight"}
            if set(config) != expected:
                raise ValueError(f"ASL requires exactly {sorted(expected)}")
            gamma_pos = _number(config["gamma_pos"], "gamma_pos", minimum=0)
            gamma_neg = _number(config["gamma_neg"], "gamma_neg", minimum=0)
            clip = _number(config["clip"], "clip", minimum=0)
            if clip >= 1 or not isinstance(config["detach_focal_weight"], bool):
                raise ValueError("ASL clip must be <1 and detach_focal_weight must be bool")
            log_positive = F.logsigmoid(values)
            log_negative = F.logsigmoid(-values)
            if clip:
                # log(min(sigmoid(-x) + clip, 1)), stable even for extreme x.
                log_negative = torch.logaddexp(
                    log_negative, torch.full_like(values, math.log(clip))).clamp(max=0)
            probability = labels * log_positive.exp() + (1 - labels) * log_negative.exp()
            gamma = labels * gamma_pos + (1 - labels) * gamma_neg
            # Replacing the base by 1 for exponent 0 also avoids an undefined
            # autograd expression 0 * 0**(-1) at saturated BCE-limit entries.
            focal_base = (1 - probability).clamp(min=0, max=1)
            focal_base = torch.where(gamma == 0, torch.ones_like(focal_base), focal_base)
            focal_weight = focal_base.pow(gamma)
            if config["detach_focal_weight"]:
                focal_weight = focal_weight.detach()
            loss = -(focal_weight * (labels * log_positive + (1 - labels) * log_negative)).mean()
        else:
            raise ValueError(f"Unknown supervised loss: {name!r}")
        _finite(loss, "supervised loss")
        return loss


def historical_kl(student_logits, teacher_logits, history_mask):
    """Mean KL(teacher Bernoulli || student Bernoulli) on historical rows.

    Three sigmoid labels remain independent. The caller explicitly normalizes
    this selected mean over the frozen historical-draw budget before backward. Teacher
    tensors are detached here as a second guard in addition to caller no_grad.
    Empty history returns a differentiable scalar zero without reading teachers.
    """
    _logit_shape(student_logits, "student_logits")
    _logit_shape(teacher_logits, "teacher_logits")
    if student_logits.shape != teacher_logits.shape or student_logits.device != teacher_logits.device:
        raise ValueError("teacher/student logits must have equal [N,3] shape and device")
    if not isinstance(history_mask, torch.Tensor) or history_mask.dtype != torch.bool:
        raise TypeError("history_mask must be a bool tensor")
    if history_mask.shape != (student_logits.shape[0],) or history_mask.device != student_logits.device:
        raise ValueError("history_mask must have shape [N] on the logits device")
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        student = student_logits.float()[history_mask]
        if student.numel() == 0:
            # Empty indexing keeps the graph connection and avoids 0*NaN.
            return student.sum()
        teacher = teacher_logits.detach().float()[history_mask]
        _finite(student, "historical student logits")
        _finite(teacher, "historical teacher logits")
        log_p, log_q = F.logsigmoid(teacher), F.logsigmoid(-teacher)
        per_label = (log_p.exp() * (log_p - F.logsigmoid(student))
                     + log_q.exp() * (log_q - F.logsigmoid(-student)))
        # Only tiny FP32 cancellation can make a mathematically nonnegative KL <0.
        loss = per_label.mean().clamp(min=0)
        _finite(loss, "historical Bernoulli KL")
        return loss


def _scope(scope):
    if scope not in ("all", "head"):
        raise ValueError("scope must be 'all' or 'head'")
    return scope


def configure_trainable(model, scope):
    """Set the complete parameter policy and return auditable parameter counts."""
    scope = _scope(scope)
    if scope == "head" and (not isinstance(getattr(model, "encoder", None), nn.Module)
                            or not isinstance(getattr(model, "head", None), nn.Module)):
        raise ValueError("head-only scope requires model.encoder and model.head modules")
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")
        parameter.grad = None
    if scope == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    enforce_train_mode(model, scope)
    total = sum(parameter.numel() for parameter in model.parameters())
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise ValueError("training scope selected no parameters")
    return {"scope": scope, "total_parameters": total, "trainable_parameters": trainable,
            "frozen_parameters": total - trainable, "trainable_parameter_names": names,
            "encoder_eval": scope == "head"}


def enforce_train_mode(model, scope):
    """Reapply mode after validation; frozen encoder state never advances."""
    scope = _scope(scope)
    model.train()
    if scope == "head":
        if not isinstance(getattr(model, "encoder", None), nn.Module) or not isinstance(
                getattr(model, "head", None), nn.Module):
            raise ValueError("head-only scope requires model.encoder and model.head modules")
        for child in model.children():
            child.eval()
        model.head.train()


def _number(value, label, minimum=None):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be scalar")
        value = value.detach().item()
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise FloatingPointError(f"Invalid {label}: {number}")
    return number


def _stats(value):
    if not isinstance(value, Mapping) or any(key not in value for key in _STAT_KEYS):
        raise ValueError(f"backward_pass must return {_STAT_KEYS}")
    result = {key: _number(value[key], key) for key in _STAT_KEYS}
    for key in ("draws", "teacher_draws"):
        if result[key] != int(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        result[key] = int(result[key])
    if result["draws"] == 0 or result["teacher_draws"] > result["draws"]:
        raise ValueError("backward callback needs positive draws and teacher_draws <= draws")
    return result


def _snapshot(tensors):
    return [(value, value.detach().clone(memory_format=torch.preserve_format)) for value in tensors]


def _restore(saved):
    with torch.no_grad():
        for value, original in saved:
            value.copy_(original)


def _strict_clip(parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise ValueError("backward_pass produced no trainable gradients")
    for gradient in gradients:
        if gradient.is_sparse:
            raise ValueError("sparse gradients are unsupported by this AdamW protocol")
        _finite(gradient, "gradient elements")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
    _finite(norm, "gradient norm")
    return float(norm.detach())


def _finite_optimizer_state(value):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        _finite(value, "optimizer state")
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite_optimizer_state(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_optimizer_state(child)


def run_update_step(model, optimizer, backward_pass, capture_rng, restore_rng, sam_rho=None):
    """One successful strict FP32-parameter update, or complete rollback and raise.

    Callback loss sums describe per-image means; callback itself accumulates
    gradients over the effective batch (16) using BF16 forward autocast and FP32
    losses. ``draws`` is logical first-pass exposure; ``physical_draws`` includes
    both SAM passes. Loss sums report the first pass, ``applied_*`` the gradient
    pass, and ``work_*`` sums both. ``backward_passes`` counts complete effective
    batch callback invocations, rather than individual microbatch backward calls.

    Snapshots deliberately include parameters, buffers and optimizer state so
    even an optimizer exception after a partial mutation cannot leave an update.
    Failure exceptions carry an ``update_record``; no failed step is successful.
    """
    rho = None if sam_rho is None else _number(sam_rho, "sam_rho", minimum=0)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    all_parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("model has no trainable parameters")
    if any(parameter.dtype != torch.float32 for parameter in all_parameters):
        raise ValueError("candidate model parameters must remain FP32")
    model_ids, trainable_ids = {id(p) for p in all_parameters}, {id(p) for p in parameters}
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    optimizer_ids = {id(p) for p in optimizer_parameters}
    if (len(optimizer_parameters) != len(optimizer_ids) or not optimizer_ids <= model_ids
            or not trainable_ids <= optimizer_ids):
        raise ValueError("optimizer must contain each trainable model parameter exactly once")
    original_parameters = _snapshot(all_parameters)
    original_buffers = _snapshot(tuple(model.buffers()))
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_rng = capture_rng()
    completed = []
    attempts = 0
    first_norm = second_norm = None
    record = {"optimizer_steps": 0, "optimizer_step_attempts": 0,
              "backward_passes": 0, "physical_draws": 0,
              "student_draws": 0, "teacher_draws": 0, "sam_rho": rho,
              "finite": False, "gradient_clip_norm": 5.0}
    try:
        optimizer.zero_grad(set_to_none=True)
        attempts += 1
        first = _stats(backward_pass())
        completed.append(first)
        first_norm = _strict_clip(parameters)
        post_first_rng = capture_rng()
        post_first_buffers = _snapshot(tuple(model.buffers()))
        if rho is not None:
            # Normalize the (strictly clipped) first gradient, preserving its
            # direction. Copy originals back; subtraction would round twice.
            gradients = [p.grad for p in parameters if p.grad is not None]
            clipped_norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients]))
            _finite(clipped_norm, "SAM perturbation norm")
            scale = rho / (clipped_norm + 1e-12)
            with torch.no_grad():
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad * scale)
                        _finite(parameter, "SAM perturbed parameters")
            optimizer.zero_grad(set_to_none=True)
            _restore(original_buffers)
            restore_rng(original_rng)
            attempts += 1
            second = _stats(backward_pass())
            completed.append(second)
            if (second["draws"], second["teacher_draws"]) != (first["draws"], first["teacher_draws"]):
                raise ValueError("SAM replay changed logical draws or teacher draws")
            second_norm = _strict_clip(parameters)
            _restore(original_parameters)
            _restore(post_first_buffers)
            restore_rng(post_first_rng)
        # The base optimizer always receives unperturbed exact parameters.
        record["optimizer_step_attempts"] = 1
        optimizer.step()
        for parameter in all_parameters:
            _finite(parameter, "updated parameters")
        _finite_optimizer_state(optimizer.state)
        for buffer in model.buffers():
            if buffer.is_floating_point():
                _finite(buffer, "updated buffers")
        physical = sum(item["draws"] for item in completed)
        teachers = sum(item["teacher_draws"] for item in completed)
        record.update(first)
        record.update({"optimizer_steps": 1, "optimizer_step_attempts": 1,
                       "backward_passes": attempts, "physical_draws": physical,
                       "student_draws": physical, "student_forward_draws": physical,
                       "student_backward_draws": physical, "teacher_draws": teachers,
                       "teacher_forward_draws": teachers, "finite": True,
                       "gradient_norm": second_norm if rho is not None else first_norm,
                       "unscaled_gradient_norm_before_clipping": second_norm if rho is not None else first_norm,
                       "first_gradient_norm": first_norm, "second_gradient_norm": second_norm,
                       "bn_buffer_updates_retained": 1, "rng_advancements_retained": 1})
        for key in _LOSS_KEYS:
            record[f"first_{key}"] = first[key]
            record[f"applied_{key}"] = completed[-1][key]
            record[f"work_{key}"] = sum(item[key] for item in completed)
        return record
    except BaseException as error:
        physical = sum(item["draws"] for item in completed)
        teachers = sum(item["teacher_draws"] for item in completed)
        record.update({"backward_passes": attempts,
                       "physical_draws": physical, "student_draws": physical,
                       "student_forward_draws": physical, "student_backward_draws": physical,
                       "teacher_draws": teachers, "teacher_forward_draws": teachers,
                       "work_counts_complete": len(completed) == attempts,
                       "first_gradient_norm": first_norm, "second_gradient_norm": second_norm,
                       "error_type": type(error).__name__, "error": str(error)})
        try:
            _restore(original_parameters)
            _restore(original_buffers)
            optimizer.load_state_dict(original_optimizer)
            optimizer.zero_grad(set_to_none=True)
            restore_rng(original_rng)
            record["rollback_complete"] = True
        except BaseException as rollback_error:
            raise RuntimeError(f"Update failed and rollback also failed: {rollback_error}") from error
        try:
            error.update_record = record
        except (AttributeError, TypeError):
            pass
        raise
