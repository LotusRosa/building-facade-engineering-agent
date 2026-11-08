"""Source-extracted server-verified core; GPU validation still required for this adapter."""
from __future__ import annotations
import numpy as np
import torch

def capture_rng():
    import random
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import random
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def model_fisher(model, inputs, count, device, bf16=True):
    """inputs yields one [1,...] image; no labels. Restore state even on error."""
    names, parameters = zip(*[(n, v) for n, v in model.named_parameters() if v.requires_grad])
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(v, v.detach().clone()) for v in model.buffers()]
    rng = capture_rng()
    if any(v.grad is not None for v in parameters):
        raise ValueError('Fisher must begin without accumulated parameter gradients')
    diagonal = {n: torch.zeros_like(v, dtype=torch.float32) for n, v in zip(names, parameters)}
    seen = 0
    try:
        model.eval()
        for image in inputs:
            if image.shape[0] != 1:
                raise ValueError('Fisher requires per-image gradients, batch1')
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                output = model(image)
                logits = output[0] if isinstance(output, tuple) else output
            if logits.shape != (1, 3) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Fisher logits must be finite [1,3]')
            probabilities = logits.detach().float().sigmoid()[0]
            for column in range(3):
                gradients = torch.autograd.grad(logits[0, column].float(), parameters,
                                                retain_graph=column < 2, create_graph=False,
                                                allow_unused=True, materialize_grads=False)
                factor = probabilities[column] * (1 - probabilities[column])
                for name, gradient in zip(names, gradients):
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError('Nonfinite Fisher logit gradient: ' + name)
                        diagonal[name].addcmul_(gradient.detach().float(), gradient.detach().float(), value=float(factor))
            seen += 1
            del gradients, logits, output
        if count <= 0 or seen != count:
            raise ValueError('Fisher sample count differs from full frozen history pool')
        for name, value in diagonal.items():
            value.div_(3 * count)
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise FloatingPointError('Invalid Fisher diagonal: ' + name)
        for buffer, before in buffers:
            if not torch.equal(buffer, before):
                raise RuntimeError('Fisher modified a running buffer')
        if any(v.grad is not None for v in parameters):
            raise RuntimeError('autograd.grad must not accumulate parameter.grad')
        return diagonal
    finally:
        with torch.no_grad():
            for buffer, before in buffers:
                buffer.copy_(before)
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)


class EWCConstraint:
    def __init__(self, model, diagonal, weight, report):
        self.anchor = {n: v.detach().clone() for n, v in model.named_parameters() if v.requires_grad}
        self.diagonal, self.weight, self.report = diagonal, float(weight), report
        if set(self.anchor) != set(diagonal):
            raise ValueError('Fisher and parent parameter sets differ')
        for n, v in self.anchor.items():
            if diagonal[n].shape != v.shape or diagonal[n].dtype != torch.float32:
                raise ValueError('Fisher shape/dtype mismatch')

    def penalty(self, model):
        parameters = dict(model.named_parameters())
        if self.weight == 0:
            # Exact zero path; no unintended FP32 arithmetic changes to BCE.
            return next(iter(parameters.values())).reshape(-1)[:0].sum()
        with torch.autocast(device_type=next(iter(parameters.values())).device.type, enabled=False):
            value = sum((self.diagonal[n] * (parameters[n].float() - anchor).square()).sum()
                        for n, anchor in self.anchor.items()) * (self.weight / 2)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite EWC penalty')
        return value
