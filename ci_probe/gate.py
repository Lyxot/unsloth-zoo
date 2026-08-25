import logging, os
logging.basicConfig(level=logging.INFO)
from unsloth_zoo.mlx.int8_prefill import capability
print("is_supported:", capability.is_supported())
print("reason:", capability.reason())
