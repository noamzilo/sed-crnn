import os
import math
# ────────────────────────────────────────────────────────────────
#  Constants (identical defaults + aug params)
# ────────────────────────────────────────────────────────────────
SEQ_LEN_IN		= 64
TIME_POOL = [2, 2, 2]
SEQ_LEN_OUT = int(SEQ_LEN_IN // math.prod(TIME_POOL))
CACHE_DIR		= os.path.expanduser("~/src/plai_cv/cache/decorte_metadata/features")
BATCH_SIZE		= 128
NUM_WORKERS		= 4

# Augmentation hyper-params
TIME_MASK_W		= 8			# frames
FREQ_MASK_W		= 8			# mel bins
MASKS_PER_EX	= 2

SAMPLE_RATE				= 44_100
HOP_LENGTH				= 2048 // 2
FPS_ORIG				= int(SAMPLE_RATE / HOP_LENGTH)		# ≈43 fps
FPS_OUT					= FPS_ORIG // math.prod(TIME_POOL)

# Architecture config
N_MELS				= 40		# used in dummy input + feature extraction
CONV_DEPTH			= 16		# was 64; reduce for fewer params
GRU1_UNITS			= 16		# was 32
GRU2_UNITS			= 8 		# was 16
DENSE1_UNITS		= 8			# was 16
