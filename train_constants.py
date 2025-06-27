import os
import math
# ────────────────────────────────────────────────────────────────
#  Constants (identical defaults + aug params)
# ────────────────────────────────────────────────────────────────
SEQ_LEN_IN		= 320
TIME_POOL = [5, 2, 2]
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