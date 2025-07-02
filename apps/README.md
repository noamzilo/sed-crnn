# SED-CRNN Video Processing

This directory contains the unified SED-CRNN video processing application.

## Structure

- `CRNNInferenceVisualizer.py` - Unified class for video processing
- `run_visualizer.py` - Single runner script with switch case for video selection

## Architecture

### CRNNInferenceVisualizer Responsibilities
- **Model Inference**: Loading model, sliding window inference, prediction generation
- **Video Processing**: Audio extraction, feature computation, video overlay creation
- **Visualization**: Plot generation, CSV export, color-coded hit detection
- **Batch Orchestration**: Train/val split organization, batch processing coordination

### Single Method Interface
- `visualize_videos(video_paths, val_fold)` - **Only method** that takes a list of video paths
- Single video = list with one video path
- Batch processing = list with multiple video paths
- Same code path, same configuration, same output structure

### Switch Case in Runner
The runner uses a simple switch case to choose which video paths to feed in:
```python
if MODE == "single":
    video_paths = [SINGLE_VIDEO_PATH]
elif MODE == "batch":
    video_paths = glob.glob(os.path.join(VIDEOS_DIR, "*.mp4"))

results = visualizer.visualize_videos(video_paths, val_fold=VAL_FOLD)
```

## Quick Start

### Configuration

Edit the configuration at the top of `run_visualizer.py`:

```python
# Model checkpoint path
CHECKPOINT_PATH = "models/sed_crnn_fold0.ckpt"

# Directory containing video files (for batch processing)
VIDEOS_DIR = "data/decorte/rallies"

# Single video path (for single video processing)
SINGLE_VIDEO_PATH = "data/decorte/rallies/sample_video.mp4"

# Base output directory
OUTPUT_DIR = "output/visualization"

# Validation fold (0-3) - only used for batch processing
VAL_FOLD = 0

# Alpha blending factor for video overlay (0.0-1.0)
ALPHA = 0.5

# Prediction threshold for binary classification
PREDICTION_THRESHOLD = 0.5

# Device to use (empty string for auto-detect, or "cuda", "cpu")
DEVICE = ""

# Processing mode: "single" or "batch"
MODE = "batch"
```

### Usage

1. **Single Video Processing:**
   ```bash
   cd sed_crnn/apps
   # Edit MODE = "single" in run_visualizer.py
   python run_visualizer.py
   ```

2. **Batch Processing:**
   ```bash
   cd sed_crnn/apps
   # Edit MODE = "batch" in run_visualizer.py
   python run_visualizer.py
   ```

## Output Structure

### Single Video
```
output/visualization/
├── train/
│   └── video_name/
│       ├── video_name_predictions.png
│       ├── video_name_overlay.mp4
│       ├── video_name_ground_truth.csv
│       ├── video_name_predictions.csv
│       └── video_name_intervals.csv
```

### Batch Processing
```
output/visualization/
├── train/
│   ├── video1/
│   │   ├── video1_predictions.png
│   │   ├── video1_overlay.mp4
│   │   └── ...
│   └── video2/
│       └── ...
└── val/
    ├── video3/
    │   ├── video3_predictions.png
    │   ├── video3_overlay.mp4
    │   └── ...
    └── video4/
        └── ...
```

## Features

- **Unified Processing**: Single `visualize_videos()` method handles all processing
- **Switch Case Selection**: Simple mode switch to choose video paths
- **Consistent Configuration**: Same paths, same parameters for all modes
- **Video Overlay**: Alpha-blended hit detection visualization
  - Green: True Positives (TP)
  - Yellow: False Positives (FP)  
  - Red: False Negatives (FN)
- **Prediction Plots**: Time-series plots with color-coded regions
- **CSV Exports**: Detailed interval analysis and classifications
- **Fold-based Organization**: Automatic train/val split based on fold assignments
- **Error Handling**: Graceful handling of missing files and processing errors

## Dependencies

- `sed_crnn.CRNNInferenceVisualizer` - Unified inference and visualization class
- `sed_crnn.decorte_data_loader` - Dataset metadata loading
- PyTorch, OpenCV, Matplotlib, Pandas

## Troubleshooting

1. **Import Errors**: Make sure you're running from the project root or the `sed_crnn/apps` directory
2. **Missing Files**: Check that checkpoint and video paths are correct
3. **CUDA Issues**: Set `DEVICE = "cpu"` if GPU is not available
4. **Memory Issues**: Reduce batch size in the visualizer if needed 