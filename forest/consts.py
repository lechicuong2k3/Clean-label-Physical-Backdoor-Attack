"""Setup constants, ymmv."""

NORMALIZE = False  # Normalize all datasets
PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 32
NUM_CLASSES = 10
FINETUNING_LR_DROP = 0.01

SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor
DISTRIBUTED_BACKEND = 'nccl'  # nccl would be faster, but require gpu-transfers for indexing and stuff
