import os
import math
# ────────────────────────────────────────────────────────────────
#  Constants (identical defaults + aug params)
# ────────────────────────────────────────────────────────────────
SEQ_LEN_IN		= 64
TIME_POOL = [1,]
SEQ_LEN_OUT = int(SEQ_LEN_IN // math.prod(TIME_POOL))
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
BATCH_SIZE		= 128
NUM_WORKERS		= 4

# Augmentation hyper-params
TIME_MASK_W		= 8			# frames
FREQ_MASK_W		= 8			# mel bins
MASKS_PER_EX	= 2

# Augmentation levels (0 disables augmentation)
NOISE_LEVEL = 0.0  # Standard deviation of Gaussian noise to add to input features (0 = no noise)
TIME_MASK_LEVEL = 1.0  # 0 disables time masking, 1.0 is default, >1 increases masking
FREQ_MASK_LEVEL = 1.0  # 0 disables freq masking, 1.0 is default, >1 increases masking
SPEC_AUGMENT_LEVEL = 1.0  # 0 disables SpecAugment, 1.0 is default, >1 increases masking
JITTER_LEVEL = 1.0  # 0 disables time jitter, 1.0 is default, >1 increases jitter
LABEL_PAD_LEVEL = 1.0  # 0 disables label padding, 1.0 is default, >1 increases padding

SAMPLE_RATE				= 44_100
HOP_LENGTH				= 2048 // 2
FPS_ORIG				= int(SAMPLE_RATE / HOP_LENGTH)		# ≈43 fps
FPS_OUT					= FPS_ORIG // math.prod(TIME_POOL)

# Architecture config
N_MELS				= 40		# used in dummy input + feature extraction
CONV_DEPTH			= 32		# was 64; reduce for fewer params
GRU1_UNITS			= 16		# was 32
GRU2_UNITS			= 8 		# was 16
DENSE1_UNITS		= 8			# was 16

# Loss function selection
LOSS_TYPE = "cross_entropy"  # Options: 'cross_entropy', 'focal'

INFER_STRIDE = max(1, SEQ_LEN_OUT // 4)

# Data augmentation parameters
LABEL_PAD_VALUE = 0.5  # Value for padded label regions
JITTER_RANGE_FRAMES = 2  # Max time jitter in frames (± this value)
LABEL_PAD_RANGE_MS = 100  # Max label padding in ms (± this value)
