# Quick Start Guide for FPS-Optimized YOLO System

## What Changed?

This update dramatically improves FPS by running camera capture and YOLO inference in separate threads, allowing them to work in parallel instead of blocking each other.

## New Features

### 1. Performance Configuration (in UI Config Tab)

**Inference FPS** (Default: 30)
- Controls how often YOLO runs
- Lower = faster capture, less frequent detection
- Higher = more detections, may slow capture
- Recommended: 10-15 for balanced performance

**Detection Input Size** (Default: 640)  
- Resolution sent to YOLO model
- Smaller = faster inference, less accurate
- Larger = slower inference, more accurate
- Recommended: 416-512 for speed, 640 for accuracy

**Enable NMS** (Default: Yes)
- Non-Maximum Suppression removes duplicate boxes
- Disable if your model already does this (end-to-end models)
- Keep enabled for standard YOLO models

**Top-K Limit** (Default: 300)
- Max detections to keep before NMS
- Reduces computational overhead
- Increase if missing detections

### 2. Performance Timing Display

New panel in "System Status" tab shows timing for:
- Capture: Camera frame capture time
- Inference: Model inference time
- Postprocess: NMS and filtering time
- Tracking: Object tracking time
- Render: Drawing boxes and display
- Total: Complete loop time

Use this to identify bottlenecks!

## Quick Tuning Guide

### For Maximum Speed (30+ FPS capture)
```
Inference FPS: 5-10
Detection Input Size: 320
Enable NMS: No (if model supports)
Top-K Limit: 100
Downsample Ratio: 0.5
```

### For Balanced Performance (15-20 FPS capture)
```
Inference FPS: 15
Detection Input Size: 512
Enable NMS: Yes
Top-K Limit: 300
Downsample Ratio: 0.75
```

### For Maximum Accuracy (10-15 FPS capture)
```
Inference FPS: 30
Detection Input Size: 640
Enable NMS: Yes
Top-K Limit: 1000
Downsample Ratio: 1.0
```

## How It Works

**Old System:**
```
Capture Frame → Wait for Inference → Wait for Tracking → Display
(Everything blocks everything else)
```

**New System:**
```
Main Thread:     Capture → Tracking → Display (30-60+ FPS)
                    ↓                ↑
Inference Thread:  Detect (runs at configured FPS, no blocking)
```

## Troubleshooting

**Q: FPS didn't improve**
- Lower "Inference FPS" to 5-10
- Reduce "Detection Input Size" to 320-416
- Check if GPU is available (should use CUDA)

**Q: Missing detections**
- Increase "Inference FPS" to 15-30
- Increase "Detection Input Size"
- Increase "Top-K Limit"
- Enable NMS if disabled

**Q: System crashes or errors**
- Check that model file exists
- Verify PyQt5 is installed: `pip install PyQt5`
- Verify onnxruntime-gpu is installed for GPU support

**Q: High latency/lag**
- Reduce "Inference FPS" (paradoxically helps by reducing load)
- Reduce "Detection Input Size"
- Check "Performance Timing" panel to see bottleneck

## Testing the Changes

1. Start the system with default settings
2. Check FPS in "System Status" tab - should be 20-30+ FPS
3. Watch "Performance Timing" panel
4. Try different configurations and observe FPS changes
5. Find the sweet spot for your use case

## Important Notes

- **First time setup**: System will use default settings (30 FPS inference)
- **Parameter changes**: Some require restart (Detection Input Size)
- **GPU usage**: System automatically uses GPU if available
- **Memory**: Monitor memory usage if running for long periods

## Expected Performance

| Setting | Capture FPS | Inference Rate | Accuracy |
|---------|-------------|----------------|----------|
| Max Speed | 40-60 | 5-10/sec | Medium |
| Balanced | 20-30 | 15/sec | Good |
| Max Accuracy | 10-15 | 30/sec | High |

*Actual FPS depends on camera hardware, model complexity, and Jetson load*

## For Advanced Users

### Dynamic Parameter Updates
While running, you can call:
```python
camera_thread.set_inference_fps(10)
camera_thread.set_enable_nms(False)
camera_thread.set_top_k_detections(200)
```

### Architecture Details
See `FPS_OPTIMIZATION_SUMMARY.md` for complete technical documentation.

## Need Help?

Check the timing panel first - it shows exactly where time is spent. Most issues can be solved by adjusting the configuration parameters.
