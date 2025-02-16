from .base import BaseAdapter
from .registry import ADAPTATION_REGISTRY
from .tent import TENT
from .t3a import T3A
from .style import StyleTransfer
from .augtta import AugTTA
from .shot import SHOT
from .shot2 import SHOT2
from .memo import MEMO

print(f"DEBUG: Available adapters after loading: {list(ADAPTATION_REGISTRY.keys())}")

__all__ = ['BaseAdapter', 'ADAPTATION_REGISTRY', 'TENT', 'T3A', 'StyleTransfer', 'AugTTA', 'SHOT', 'SHOT2', 'MEMO']