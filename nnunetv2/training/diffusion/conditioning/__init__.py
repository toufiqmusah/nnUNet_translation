from .base_conditioning import ConditioningMethod
from .concat_conditioning import ConcatConditioning

# Lazy import for cross-attention to allow future implementation
def __getattr__(name):
    if name == 'CrossAttentionConditioning':
        from .cross_attention_conditioning import CrossAttentionConditioning
        return CrossAttentionConditioning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'ConditioningMethod',
    'ConcatConditioning',
]
