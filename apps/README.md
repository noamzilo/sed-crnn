# SED-CRNN Batch Visualization Apps

This directory contains applications for batch processing videos using the `SedRcnnInference` class.

## Files

- `SedRcnnInference.py` - Main inference and visualization class
- `batch_visualizer.py` - Simple batch processor (fold 0 = val, others = train)
- `fold_based_visualizer.py` - Flexible batch processor with configurable validation fold
- `single_video_demo.py` - Demo script for single video processing

## Usage

### Demo Single Video

First, test that everything works with a single video:

```bash
cd sed-crnn/apps
python single_video_demo.py
```

### Simple Batch Processing

Process all videos with fold 0 as validation:

```bash
cd sed-crnn/apps
python batch_visualizer.py
```

### Flexible Batch Processing

Process all videos with configurable validation fold:

```bash
cd sed-crnn/apps

# Edit the configuration at the top of fold_based_visualizer.py first:
# - VAL_FOLD: Which fold to use as validation (0-3)
# - ALPHA: Alpha blending factor (0.0-1.0)
# - THRESHOLD: Prediction threshold (0.0-1.0)
# - CKPT_PATH: Path to model checkpoint
# - OUTPUT_DIR: Output directory

# Then run:
python fold_based_visualizer.py
```

## Configuration (fold_based_visualizer.py)

Edit these values at the top of the file:

- `CKPT_PATH`: Path to model checkpoint
- `VIDEOS_DIR`: Directory containing video files
- `OUTPUT_DIR`: Base output directory
- `VAL_FOLD`: Fold ID to use as validation set (0-3)
- `ALPHA`: Alpha blending factor for video overlay (0.0-1.0)
- `THRESHOLD`: Prediction threshold for binary classification (0.0-1.0)
- `DEVICE`: Device to use (empty string for auto-detect)

## Output Structure

```
output/
├── train/
│   ├── 20230528_VIGO_00/
│   │   ├── 20230528_VIGO_00_overlay.mp4
│   │   ├── 20230528_VIGO_00_predictions.png
│   │   ├── 20230528_VIGO_00_ground_truth.csv
│   │   ├── 20230528_VIGO_00_predictions.csv
│   │   └── 20230528_VIGO_00_intervals.csv
│   └── ...
└── val/
    ├── 20230528_VIGO_04/
    │   ├── 20230528_VIGO_04_overlay.mp4
    │   ├── 20230528_VIGO_04_predictions.png
    │   ├── 20230528_VIGO_04_ground_truth.csv
    │   ├── 20230528_VIGO_04_predictions.csv
    │   └── 20230528_VIGO_04_intervals.csv
    └── ...
```

## Output Files

For each video, the following files are generated:

- `*_overlay.mp4`: Video with color-coded hit detection overlay
  - Green: True Positive (TP)
  - Yellow: False Positive (FP)
  - Red: False Negative (FN)
- `*_predictions.png`: Plot showing predictions vs ground truth
- `*_ground_truth.csv`: Ground truth intervals
- `*_predictions.csv`: Prediction intervals
- `*_intervals.csv`: Combined intervals with classifications

## Fold Assignments

The Decorte dataset uses 4-fold cross-validation. Each video is assigned to one of 4 folds (0-3). You can specify which fold to use as the validation set, and all other folds will be treated as training data.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Pandas
- PyTorch Lightning
- FFmpeg (for video processing)

## Notes

- The apps automatically skip videos that are not in the Decorte metadata
- Processing time depends on video length and hardware
- GPU acceleration is automatically used if available
- All outputs maintain the original video's audio synchronization 