from typing import Dict, Type
from .base import BaseAdapter
import logging

# Setup logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

ADAPTATION_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

def register_adapter(name):
    # logger.debug(f"Registering adapter: {name}")
    def register_adapter_cls(cls):
        # logger.debug(f"Processing registration for class: {cls.__name__}")
        if name in ADAPTATION_REGISTRY:
            raise ValueError(f"Adapter {name} already registered")
        if not issubclass(cls, BaseAdapter):
            raise ValueError(f"Adapter {name} must extend BaseAdapter")
        ADAPTATION_REGISTRY[name] = cls
        # logger.debug(f"Successfully registered {name}: {cls.__name__}")
        # logger.debug(f"Current registry: {list(ADAPTATION_REGISTRY.keys())}")
        return cls
    return register_adapter_cls