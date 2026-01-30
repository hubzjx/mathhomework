# FPS Optimization Summary for YOLO Inference on Jetson Orin NX

## Overview
This document summarizes the FPS optimization improvements made to `seedyolo_v13_完善engine.py` for real-time YOLO inference on Jetson Orin NX 8GB using ONNXRuntime.

## Problem Analysis

### Original Bottlenecks
1. **Synchronous capture+inference+render loop**: All operations happened sequentially in one thread, causing inference to block camera capture
2. **Hard sleep throttling**: Fixed 30 FPS sleep causing unnecessary delays
3. **Coupled UI refresh**: UI updates were tied to inference rate
4. **CPU-bound preprocessing**: cvtColor, resize, and transpose operations on CPU
5. **Unbounded detections**: No limit on number of detections before NMS
6. **No frame dropping**: Old frames queued up causing latency

### Target Resolution
- 3840×2750 camera resolution
- Current FPS: ~3 FPS
- Goal: Maximum FPS improvement

## Implemented Optimizations

### 1. Threaded Pipeline Architecture

#### LatestFrameHolder Class
```python
class LatestFrameHolder:
    """Thread-safe holder for the latest frame"""
```
- Thread-safe frame storage using locks
- Stores only the latest frame (no queue backlog)
- Frame ID tracking to prevent duplicate processing

#### InferenceWorker Thread
```python
class InferenceWorker(threading.Thread):
    """Independent inference thread"""
```
- Runs inference in a separate daemon thread
- Processes latest frame only (automatic frame dropping)
- Configurable inference interval
- Non-blocking callback for results

#### Benefits
- Camera capture runs at full speed without blocking
- Inference happens in parallel
- Automatic frame dropping prevents backlog
- UI refresh decoupled from inference rate

### 2. Performance Configuration Parameters

#### New Configuration Constants
```python
INFERENCE_FPS_DEFAULT = 30
INFERENCE_FPS_OPTIONS = [1, 3, 5, 10, 15, 20, 30, 60]
DETECTION_INPUT_SIZE_DEFAULT = 640
DETECTION_INPUT_SIZE_OPTIONS = [320, 416, 512, 640, 800, 1024]
ENABLE_NMS_DEFAULT = True
TOP_K_DETECTIONS = 300
```

#### UI Controls Added
- **Inference FPS**: Control how often inference runs (1-60 FPS)
- **Detection Input Size**: Resize detection input (320-1024 pixels)
- **Enable NMS**: Toggle NMS on/off for end-to-end models
- **Top-K Limit**: Cap detections before NMS (reduces NMS overhead)

### 3. Detector Improvements

#### Updated Detector Classes
All detector classes now support:
- `input_size` parameter for configurable detection resolution
- `enable_nms` parameter to disable NMS for end-to-end models
- Consistent interface across TRTDetector, OnnxDetector, PtDetector

#### Top-K Detection Limiting
```python
if len(detections) > self.top_k_detections:
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:self.top_k_detections]
```
Applied in InferenceWorker before NMS to reduce computational overhead.

### 4. Performance Instrumentation

#### Timing Measurements
New timing stats tracked for each frame:
- `capture`: Camera frame capture time
- `inference`: Model inference time
- `postprocess`: NMS and post-processing time
- `tracking`: Object tracking time
- `render`: Frame rendering and annotation time
- `total`: Total loop time

#### UI Display
Added "Performance Timing (ms)" panel in System Status page showing:
- Real-time timing breakdown
- Identifies bottlenecks
- Helps tune parameters

### 5. Optimized Main Loop

#### Key Changes
1. **Non-blocking inference submission**: `frame_holder.update(frame)` just stores the latest frame
2. **Non-blocking result retrieval**: `with self.detection_lock: detections = self.latest_detections.copy()`
3. **Precise timing**: Using `time.perf_counter()` for accurate measurements
4. **Configurable inference interval**: Only process frames at configured rate
5. **Reduced sleep time**: Changed from `1.0/30` to maintain UI responsiveness

## Usage Guide

### Basic Usage
1. **Select Model**: Choose or input model path
2. **Configure Performance**:
   - Set Inference FPS (lower = higher capture FPS, less accurate)
   - Set Detection Input Size (smaller = faster, less accurate)
   - Enable/disable NMS as needed
   - Set Top-K limit to reduce NMS overhead
3. **Start System**: Click "启动系统"
4. **Monitor Performance**: Check timing stats in System Status tab

### Performance Tuning Recommendations

#### For Maximum FPS
```
Inference FPS: 5-10
Detection Input Size: 320-416
Enable NMS: False (if model is end-to-end)
Top-K Limit: 100-200
Downsample Ratio: 0.5
```

#### For Maximum Accuracy
```
Inference FPS: 30
Detection Input Size: 640-800
Enable NMS: True
Top-K Limit: 1000+
Downsample Ratio: 1.0
```

#### Balanced Configuration
```
Inference FPS: 15
Detection Input Size: 512
Enable NMS: True
Top-K Limit: 300
Downsample Ratio: 0.75
```

### Dynamic Parameter Updates
Methods available for runtime updates:
- `camera_thread.set_inference_fps(fps)`
- `camera_thread.set_enable_nms(enable)`
- `camera_thread.set_top_k_detections(top_k)`
- `camera_thread.set_downsample_ratio(ratio)`

## Expected Performance Improvements

### Capture FPS
- **Before**: ~3 FPS (bottlenecked by synchronous inference)
- **After**: 20-60+ FPS (limited by camera hardware and configuration)

### Inference Latency
- **Frame dropping**: Old frames automatically dropped, no backlog
- **Configurable rate**: Inference runs at configured FPS independently
- **Parallel execution**: Capture continues during inference

### Resource Utilization
- **Better GPU utilization**: Continuous inference stream
- **Better CPU utilization**: Parallel capture and render
- **Memory efficient**: Only latest frame stored

## Architecture Diagram

```
┌─────────────┐
│   Camera    │
│   Thread    │ (Main Loop)
└──────┬──────┘
       │ Capture at full speed
       ↓
┌──────────────┐
│ LatestFrame  │ (Thread-safe storage)
│   Holder     │
└──────┬───────┘
       │ Latest frame only
       ↓
┌──────────────┐
│  Inference   │ (Separate thread)
│   Worker     │ Runs at configured FPS
└──────┬───────┘
       │ Results callback
       ↓
┌──────────────┐
│   Tracking   │ (Main loop)
│   Counting   │ Process results
│   Rendering  │ Draw and display
└──────────────┘
```

## Validation

### Testing Checklist
- [ ] System starts without errors
- [ ] Camera capture runs smoothly
- [ ] Detections appear correctly
- [ ] UI remains responsive
- [ ] Performance stats update correctly
- [ ] Parameter changes take effect
- [ ] System stops cleanly

### Monitoring Points
1. Check "Performance Timing" panel for bottlenecks
2. Monitor capture FPS vs inference rate
3. Verify detection quality at different settings
4. Check memory usage stability over time

## Future Enhancements

### Potential Further Optimizations
1. **GPU preprocessing**: Move cvtColor/resize to CUDA
2. **Batch inference**: Process multiple frames together
3. **Model optimization**: TensorRT INT8 quantization
4. **Zero-copy operations**: Use GPU memory directly
5. **Async CUDA streams**: Overlap preprocessing and inference

### Additional Features
1. **Adaptive FPS**: Auto-adjust based on load
2. **Multi-model support**: Different models for different conditions
3. **Performance presets**: One-click optimization profiles
4. **Benchmarking mode**: Automated performance testing

## Troubleshooting

### Low FPS despite optimizations
- Check if camera hardware supports higher FPS
- Verify CUDA/TensorRT are available
- Reduce detection input size
- Lower inference FPS
- Increase Top-K limit if too many detections

### High latency
- Increase inference FPS
- Reduce detection input size
- Disable NMS if not needed
- Check GPU memory usage

### Missing detections
- Increase inference FPS
- Increase detection input size
- Enable NMS
- Increase Top-K limit
- Increase downsample ratio

## Conclusion

These optimizations provide a significant FPS improvement by:
1. Decoupling capture, inference, and rendering operations
2. Adding configurable performance parameters
3. Implementing automatic frame dropping
4. Providing detailed performance instrumentation

The system now prioritizes maximum FPS while maintaining user control over accuracy/speed tradeoff.
