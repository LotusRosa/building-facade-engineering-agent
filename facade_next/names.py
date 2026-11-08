"""User-facing vocabulary; persisted method IDs and training presets stay stable."""
import re
from .contracts import METHODS

METHOD_LABELS = {
    'core': {'zh': 'Core · 初始建模', 'en': 'Core · Initial modeling'},
    'standard': {'zh': 'Standard · 常规维护', 'en': 'Standard · Adaptation'},
    'asl': {'zh': '非对称损失（ASL）', 'en': 'Asymmetric loss (ASL)'},
    'sam': {'zh': '锐度感知优化（SAM）', 'en': 'Sharpness-aware optimization (SAM)'},
    'distill': {'zh': '历史知识蒸馏', 'en': 'Historical knowledge distillation'},
    'hnm': {'zh': '困难负样本挖掘', 'en': 'Hard negative mining'},
    'hem': {'zh': '困难样本重加权', 'en': 'Hard example reweighting'},
    'head_only': {'zh': '仅更新分类头', 'en': 'Classifier head update only'},
    'ewc_replay': {'zh': '抗遗忘约束与历史回放（EWC）', 'en': 'EWC constraint with replay'},
}
METHOD_ALIASES = {
    'core': ['core', '首轮基础训练'],
    'standard': ['standard', 'baseline', '基础方法', '标准训练', '标准更新', '普通训练', '标准训练／更新', 'Standard training and update', 'Standard \u00b7 Maintenance update'],
    'asl': ['asl', '非对称损失'], 'sam': ['sam', '锐度感知优化'],
    'distill': ['distill', 'distillation', '蒸馏'],
    'hnm': ['hnm', '难负样本', '困难负样本'],
    'hem': ['hem', '难例', '困难样本'],
    'head_only': ['head_only', 'head only', '只训练分类头', '仅训练分类头', '只更新分类头', '仅更新分类头'],
    'ewc_replay': ['ewc_replay', 'ewc replay', 'ewc', '抗遗忘约束', '历史回放约束'],
}

for _method,_labels in METHOD_LABELS.items():
    METHOD_ALIASES[_method].extend(_labels.values())


WORKFLOW_METHODS = ['core', *METHODS]


def entry_tools(tools, kind):
    """Normalize public entries; the persisted/training method IDs stay stable."""
    if tools is None:
        tools=['core'] if kind=='initial' else ['standard']
    if not isinstance(tools,list) or any(not isinstance(t,str) for t in tools):
        raise ValueError('Select supported training entries as a list')
    if len(set(tools))!=len(tools) or set(tools)-set(WORKFLOW_METHODS):
        raise ValueError('Select distinct supported training entries')
    if kind=='initial':
        if tools not in ([],['core'],['standard']):
            raise ValueError('Initial modeling uses Core only; choose maintenance methods after adopting an initial model')
        return ['core']
    if 'core' in tools:
        raise ValueError('Core is for initial modeling only; choose Standard or other maintenance methods')
    if not tools:
        raise ValueError('Select at least one maintenance method, such as Standard')
    return list(tools)


def engine_tools(tools, kind):
    return ['standard' if t=='core' else t for t in entry_tools(tools,kind)]


def method_name(method, language='zh', kind=None):
    entry='core' if kind=='initial' and method=='standard' else method
    return METHOD_LABELS.get(entry, {}).get(language, entry)


def mentioned_methods(text):
    # Historical Core Safety is a data-role name, not a Core training request.
    text=re.sub(r'\bcore\s+safety\b','',text,flags=re.I)
    return [method for method, aliases in METHOD_ALIASES.items()
            if any(re.search(r'(?<![a-z0-9_])'+re.escape(name)+r'(?![a-z0-9_])', text, re.I)
                   if name.isascii() else name in text for name in aliases)]
