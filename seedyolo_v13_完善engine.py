#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVS 相机 + YOLO(Engine/ONNX/PT) + 过线计数 + 密度识别 + 里程对齐(时间对齐) + 门控识别 + UI界面
并支持：把"识别到的种子数量"按"千粒重"换算为克重，作为 EVENT 下发字段之一。

新增：
- 追踪器增强：IOU + 平滑中心 + 丢失容忍，可选计数区域/方向约束。
- UART 断连重连：RX/TX 都支持自动重连与状态告警。
- 现代化UI界面：三页设计（配置页+视频显示页+长图识别页），自动适配屏幕
- 兼容性改进：更好地支持 YOLO26 等新模型输出格式
- 新增长图拼接与识别：
    * UART 收发开启时：速度 >0 自动拼接；速度 ≤0 自动停止并整体 YOLO 识别。
    * UART 收发关闭时：通过快捷键或按钮手动开始/结束拼接，结束后整体 YOLO 识别。
    * 原始长图与识别结果保存到同一子目录（按年月日时分秒命名），保存根目录可配置。
- 新增截屏功能：截取程序全屏界面，保存路径可配置

修改：
- ROI识别逻辑：启用过线计数时只识别ROI区域，不启用时全屏识别
- 计数线参数化：计数线位置从上到下用百分比参数设置（如5%、17%）
- ROI自动计算：根据计数线位置和上下扩展像素数自动计算ROI区域
- UI显示调整：视频左上角不再显示统计信息，只在系统状态页显示
- TensorRT 10.x API兼容性：根据TensorRT版本自适应选择执行方法
"""

import os
import sys
import cv2
import time
import math
import json
import struct
import threading
import queue
import platform
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from ctypes import *

# ===== UI 库 =====
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                 QTabWidget, QGroupBox, QLabel, QPushButton, QComboBox, 
                                 QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QTextEdit,
                                 QGridLayout, QSplitter, QScrollArea, QFrame, QSizePolicy,
                                 QMessageBox, QProgressBar, QSlider, QFileDialog)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QRect
    from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor, QPainter, QPen
    PYQT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PyQt5不可用: {e}")
    print("请安装: pip install PyQt5")
    PYQT_AVAILABLE = False

# ===== MVS SDK =====
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *  # noqa

# ===== 可选：串口 =====
try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

# ===== 可选：YOLOv5 (仅用于.pt 走 yolov5 代码路径时) =====
YOLOV5_CODE_AVAILABLE = False
try:
    sys.path.append('/home/nvidia/Desktop/yolov5')
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    YOLOV5_CODE_AVAILABLE = True
except Exception:
    YOLOV5_CODE_AVAILABLE = False

# ===== PyTorch (用于.pt) =====
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ===== TensorRT (用于.engine) =====
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except Exception:
    TRT_AVAILABLE = False

# ==========================
# 配置区
# ==========================
SEED_TKW_GRAMS = 40.0  # 千粒重（g/1000粒）——可更改
DEFAULT_MODEL_PATH = "/home/nvidia/Desktop/yolov5/runs/train/1-22-26sbest.onnx"
MODEL_CANDIDATES = [
    "yolov12sbest.engine",
    "yolov12sbest.onnx",
    "yolov12sbest.pt",
    "yolov5s.pt",
    "yolov5s.onnx",
    "yolov5s.engine",
    "yolo26s.onnx",        # 添加YOLO26模型支持
    "yolo26s.engine",
    "yolo26s.pt",
]

# 串口配置（根据实际修改）
UART_PORT = "/dev/ttyUSB0"
UART_BAUD = 921600
UART_RETRY_SEC = 2.0

# 核心板 CTRL 帧类型
CTRL_MSG_TYPE = 0x01

# Jetson EVENT 帧类型
EVENT_MSG_TYPE = 0x10
EVENT_TYPE_LOW_DENSITY_SEG = 0x01

# 门控（正向即开；非正向或过期即关）
V_ON = 0      # 保留字段，现逻辑不依赖阈值
V_OFF = 0     # 保留字段，现逻辑不依赖阈值
GATE_MAX_AGE = 0.5  # s，CTRL 数据超过该年龄则 gate=false

CTRL_BUFFER_SECONDS = 10.0

# 密度识别参数
DENSITY_ENABLED_DEFAULT = True
DENSITY_MAP_SHOW_DEFAULT = False
DENSITY_GRID_COLS = 16
DENSITY_GRID_ROWS = 6
LOW_DENSITY_RATIO = 0.35
LOW_DENSITY_MIN_FRAMES = 3
RECOVER_MIN_FRAMES = 3
MIN_DETS_FOR_DENSITY = 3  # 少于该检测数不判定低密度

# 缓存文件（等待下一步下发，或也可边缓存边下发）
EVENT_CACHE_FILE = "low_density_segments.jsonl"
EVENT_MERGE_GAP_MM = 80
CACHE_QUEUE_MAX = 2000
CACHE_FLUSH_INTERVAL = 1.0  # s
CACHE_ROTATE_LINES = 200000  # 超过则轮转
CACHE_ROTATE_KEEP = 5  # 最多保留 5 个历史文件

# UI相关配置
UI_UPDATE_INTERVAL = 30  # ms
VIDEO_DISPLAY_FPS = 30   # 视频显示帧率

# 检测器配置
USE_NMS_FOR_YOLO26 = True  # 对于YOLO26端到端模型，可设为False
YOLO26_CONF_THRES = 0.15   # YOLO26专用置信度阈值
MAX_DETECTIONS = 20000     # 最大检测数量限制

# EVENT 发送队列
EVENT_QUEUE_MAX = 2000

# 长图拼接配置
DEFAULT_SAVE_DIR = "/home/nvidia/Desktop/yolov5/captures"
DEFAULT_HOTKEY_START = "S"
DEFAULT_HOTKEY_STOP = "E"
MAX_LONG_FRAMES = 400           # 防止无限增长
MIN_HEIGHT_FOR_SAVE = 50        # 少于该像素高度不保存

# 计数线位置参数（从上到下百分比）
COUNT_LINE_PERCENT_DEFAULT = 5.0  # 默认5%
COUNT_LINE_TOP_EXTEND_DEFAULT = 100  # 向上扩展像素数
COUNT_LINE_BOTTOM_EXTEND_DEFAULT = 100  # 向下扩展像素数

# 画面下采样参数
DOWNSAMPLE_RATIO_DEFAULT = 1.0   # 默认不下采样
DOWNSAMPLE_OPTIONS = [1.0, 0.75, 0.5, 0.25]  # 下采样选项

# ===== CRC16 (CCITT-FALSE) =====
def crc16_ccitt_false(data: bytes, poly=0x1021, init=0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# ===== 种子重量换算 =====
def seed_count_to_grams(seed_count: int, tkw_grams: float) -> float:
    """根据千粒重将检测到的seed数量换算为克重（g）。"""
    if seed_count <= 0:
        return 0.0
    return float(seed_count) * (float(tkw_grams) / 1000.0)


def grams_to_mg_u32(grams: float) -> int:
    """把克重转换为毫克并用 uint32 表示（避免浮点传输）。"""
    mg = int(round(max(0.0, grams) * 1000.0))
    return max(0, min(0xFFFFFFFF, mg))


# ===== 控制数据结构与对齐 =====
@dataclass
class CtrlSample:
    t_rx_host: float
    t_ctrl_us: int
    s_mm: int
    v_mmps: int


class CtrlBuffer:
    def __init__(self, keep_seconds=10.0):
        self.keep_seconds = keep_seconds
        self.buf = deque()
        self.lock = threading.Lock()
        self._time_map_inited = False
        self.a = None
        self.b = None

    def add(self, sample: CtrlSample):
        with self.lock:
            self.buf.append(sample)
            self._trim_locked()

            if len(self.buf) >= 2:
                s2 = self.buf[-1]
                s1 = self.buf[-2]
                dt_ctrl = (s2.t_ctrl_us - s1.t_ctrl_us) / 1e6
                dt_host = (s2.t_rx_host - s1.t_rx_host)
                # 异常过滤：dt_ctrl 太小或负值直接丢弃拟合
                if dt_ctrl > 1e-4 and dt_host > 0:
                    a = dt_host / dt_ctrl
                    b = s2.t_rx_host - a * (s2.t_ctrl_us / 1e6)
                    # 过滤离群的 a：若变化过大则忽略
                    if self.a is None or abs(a - self.a) / max(1e-9, abs(self.a)) < 0.2:
                        if self.a is None:
                            self.a, self.b = a, b
                        else:
                            self.a = 0.9 * self.a + 0.1 * a
                            self.b = 0.9 * self.b + 0.1 * b
                        self._time_map_inited = True

    def _trim_locked(self):
        now = time.monotonic()
        while self.buf and (now - self.buf[0].t_rx_host) > self.keep_seconds:
            self.buf.popleft()

    def get_latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def gate_enabled(self, v_on=V_ON, v_off=V_OFF, prev_state=False, max_age=GATE_MAX_AGE):
        latest = self.get_latest()
        if latest is None:
            return False
        age = time.monotonic() - latest.t_rx_host
        if age > max_age:
            return False
        v = latest.v_mmps
        # 速度为正就开，零/负就关
        return v > 0

    def query_s_at_host_time(self, t_cap_host: float):
        with self.lock:
            if len(self.buf) < 2:
                return None, None, None

            if self._time_map_inited and self.a is not None and self.b is not None:
                if abs(self.a) < 1e-9:
                    return None, None, None
                t_ctrl_target_us = int(((t_cap_host - self.b) / self.a) * 1e6)

                prev = None
                for cur in self.buf:
                    if cur.t_ctrl_us >= t_ctrl_target_us:
                        if prev is None:
                            age = t_cap_host - cur.t_rx_host
                            if age > self.keep_seconds:
                                return None, None, None
                            return cur.s_mm, cur.v_mmps, age
                        t1, t2 = prev.t_ctrl_us, cur.t_ctrl_us
                        if t2 == t1:
                            age = t_cap_host - cur.t_rx_host
                            if age > self.keep_seconds:
                                return None, None, None
                            return cur.s_mm, cur.v_mmps, age
                        alpha = (t_ctrl_target_us - t1) / (t2 - t1)
                        s = int(round(prev.s_mm + alpha * (cur.s_mm - prev.s_mm)))
                        v = int(round(prev.v_mmps + alpha * (cur.v_mmps - prev.v_mmps)))
                        age = t_cap_host - (prev.t_rx_host + alpha * (cur.t_rx_host - prev.t_rx_host))
                        if age > self.keep_seconds:
                            return None, None, None
                        return s, v, age
                    prev = cur

                last = self.buf[-1]
                age = t_cap_host - last.t_rx_host
                if age > self.keep_seconds:
                    return None, None, None
                return last.s_mm, last.v_mmps, age

            # fallback：用 t_rx_host
            prev = None
            for cur in self.buf:
                if cur.t_rx_host >= t_cap_host:
                    if prev is None:
                        age = t_cap_host - cur.t_rx_host
                        if age > self.keep_seconds:
                            return None, None, None
                        return cur.s_mm, cur.v_mmps, age
                    t1, t2 = prev.t_rx_host, cur.t_rx_host
                    if t2 == t1:
                        age = t_cap_host - cur.t_rx_host
                        if age > self.keep_seconds:
                            return None, None, None
                        return cur.s_mm, cur.v_mmps, age
                    alpha = (t_cap_host - t1) / (t2 - t1)
                    s = int(round(prev.s_mm + alpha * (cur.s_mm - prev.s_mm)))
                    v = int(round(prev.v_mmps + alpha * (cur.v_mmps - prev.v_mmps)))
                    age = t_cap_host - (t1 + alpha * (t2 - t1))
                    if age > self.keep_seconds:
                        return None, None, None
                    return s, v, age
                prev = cur

            last = self.buf[-1]
            age = t_cap_host - last.t_rx_host
            if age > self.keep_seconds:
                return None, None, None
            return last.s_mm, last.v_mmps, age


# ===== UART 接收线程（CTRL） 支持断连重连 =====
class UartCtrlReceiver(threading.Thread):
    def __init__(self, port, baud, ctrl_buf: CtrlBuffer, retry_sec=UART_RETRY_SEC):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.ctrl_buf = ctrl_buf
        self.retry_sec = retry_sec
        self.stop_flag = threading.Event()
        self.ser = None

    def stop(self):
        self.stop_flag.set()
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

    def _open(self):
        if not SERIAL_AVAILABLE:
            print("⚠️  pyserial 不可用：无法接收核心板 CTRL 数据。")
            return False
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            self.ser.reset_input_buffer()
            print(f"✅ UART 已打开(CTRL RX): {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"⚠️  UART 打开失败: {e}, 将在 {self.retry_sec}s 后重试")
            return False

    def run(self):
        buf = bytearray()
        while not self.stop_flag.is_set():
            if self.ser is None or (not self.ser.is_open):
                if not self._open():
                    time.sleep(self.retry_sec)
                    continue

            try:
                chunk = self.ser.read(512)
                if chunk:
                    buf.extend(chunk)
                else:
                    continue

                while True:
                    sof_idx = buf.find(b'\xAA\x55')
                    if sof_idx < 0:
                        if len(buf) > 4096:
                            buf = buf[-16:]
                        break
                    if sof_idx > 0:
                        buf = buf[sof_idx:]

                    # 最小长度：2+1+1+18+2 = 24
                    if len(buf) < 24:
                        break

                    msg_type = buf[2]
                    payload_len = buf[3]
                    frame_len = 2 + 1 + 1 + payload_len + 2
                    if len(buf) < frame_len:
                        break

                    frame = bytes(buf[:frame_len])
                    buf = buf[frame_len:]

                    if msg_type != CTRL_MSG_TYPE:
                        continue

                    recv_crc = struct.unpack_from('<H', frame, frame_len - 2)[0]
                    calc_crc = crc16_ccitt_false(frame[:-2])
                    if recv_crc != calc_crc:
                        continue

                    if payload_len < 18:
                        continue

                    seq = struct.unpack_from('<H', frame, 4)[0]
                    t_ctrl_us = struct.unpack_from('<Q', frame, 6)[0]
                    s_mm = struct.unpack_from('<I', frame, 14)[0]
                    v_mmps = struct.unpack_from('<i', frame, 18)[0]

                    sample = CtrlSample(
                        t_rx_host=time.monotonic(),
                        t_ctrl_us=int(t_ctrl_us),
                        s_mm=int(s_mm),
                        v_mmps=int(v_mmps)
                    )
                    self.ctrl_buf.add(sample)

            except Exception as e:
                print(f"⚠️  UART RX 异常: {e}, 将重连")
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(self.retry_sec)


# ===== UART 下发线程（EVENT，异步队列 + 重连）=====
class UartEventSender(threading.Thread):
    """
    异步发送 EVENT：主线程调用 send_low_density_segment() 仅入队；后台线程串口发送。
    """
    def __init__(self, port, baud, queue_max=EVENT_QUEUE_MAX, retry_sec=UART_RETRY_SEC):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.q = queue.Queue(maxsize=queue_max)
        self.stop_flag = threading.Event()
        self.ser = None
        self.ready = False
        self.retry_sec = retry_sec

    def open_serial(self):
        if not SERIAL_AVAILABLE:
            print("⚠️  pyserial 不可用：无法下发 EVENT。")
            return False
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            print(f"✅ UART 已打开(EVENT TX): {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"⚠️  EVENT UART 打开失败: {e}")
            return False

    def run(self):
        while not self.stop_flag.is_set():
            if not self.ready:
                self.ready = self.open_serial()
                if not self.ready:
                    time.sleep(self.retry_sec)
                    continue

            try:
                payload = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._send_frame(payload)
            except Exception as e:
                print(f"⚠️ EVENT 发送失败: {e}，将重连")
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self.ready = False
                time.sleep(self.retry_sec)
            finally:
                self.q.task_done()

    def stop(self):
        self.stop_flag.set()
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

    def _send_frame(self, payload: bytes):
        header = b'\xAA\x55' + bytes([EVENT_MSG_TYPE]) + bytes([len(payload)])
        frame_wo_crc = header + payload
        crc = crc16_ccitt_false(frame_wo_crc)
        frame = frame_wo_crc + struct.pack('<H', crc)
        if self.ser:
            self.ser.write(frame)

    def send_low_density_segment(self, event_id: int, severity: int,
                                 s_start_mm: int, s_end_mm: int,
                                 seed_count: int, weight_mg: int):
        payload = struct.pack(
            '<HBBIIHI',
            int(event_id) & 0xFFFF,
            EVENT_TYPE_LOW_DENSITY_SEG,
            int(severity) & 0xFF,
            int(s_start_mm) & 0xFFFFFFFF,
            int(s_end_mm) & 0xFFFFFFFF,
            int(seed_count) & 0xFFFF,
            int(weight_mg) & 0xFFFFFFFF
        )
        try:
            self.q.put(payload, timeout=0.01)
        except queue.Full:
            print("⚠️ EVENT 队列已满，丢弃一条低密度事件。")


# ===== 相机 =====
class MVS_Camera:
    def __init__(self):
        self.camera = None
        self.is_opened = False

    def open_camera(self):
        try:
            MvCamera.MV_CC_Initialize()

            device_list = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
            print(f"枚举设备: ret=0x{ret:08X}, 数量={device_list.nDeviceNum}")
            if ret != 0 or device_list.nDeviceNum == 0:
                return False

            self.camera = MvCamera()
            st_device = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

            ret = self.camera.MV_CC_CreateHandle(st_device)
            if ret != 0:
                return False

            ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                return False

            ret = self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print(f"⚠️ TriggerMode设置失败: 0x{ret:08X}")

            # 降低缓存减少积压
            self.camera.MV_CC_SetImageNodeNum(6)

            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                self.camera.MV_CC_CloseDevice()
                return False

            self.is_opened = True
            return True
        except Exception:
            return False

    def capture_frame_alternative(self):
        if not self.is_opened:
            return None

        stFrame = MV_FRAME_OUT()
        memset(byref(stFrame), 0, sizeof(stFrame))

        ret = self.camera.MV_CC_GetImageBuffer(stFrame, 5000)
        if ret != 0:
            return None

        buffer_size = stFrame.stFrameInfo.nFrameLen
        # Optimize: Use np.frombuffer directly without memmove
        # This avoids an extra memory copy operation
        image_array = np.frombuffer(
            (c_ubyte * buffer_size).from_address(stFrame.pBufAddr), 
            dtype=np.ubyte, 
            count=buffer_size
        ).copy()  # Still need a copy since we must free the buffer

        frame = None
        pixel_type = stFrame.stFrameInfo.enPixelType

        try:
            if pixel_type == PixelType_Gvsp_BGR8_Packed:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3))
            elif pixel_type == PixelType_Gvsp_RGB8_Packed:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif pixel_type == PixelType_Gvsp_Mono8:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif pixel_type == PixelType_Gvsp_BayerRG8:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
            elif pixel_type == PixelType_Gvsp_BayerGB8:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
            elif pixel_type == PixelType_Gvsp_BayerGR8:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
            elif pixel_type == PixelType_Gvsp_BayerBG8:
                frame = image_array.reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
        except Exception:
            frame = None

        self.camera.MV_CC_FreeImageBuffer(stFrame)
        return frame

    def close(self):
        if self.camera:
            try:
                if self.is_opened:
                    self.camera.MV_CC_StopGrabbing()
                    self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.is_opened = False

        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass


# ===== 增强追踪器（IOU + 平滑 + 丢失容忍 + 区域/方向约束）=====
class EnhancedTracker:
    def __init__(self,
                 iou_thresh=0.3,
                 max_frames_missing=15,
                 smoothing=0.5,
                 count_zone=None):
        """
        count_zone: (x1,y1,x2,y2) 计数区域，None 表示全帧
        """
        self.tracks = {}
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_frames_missing = max_frames_missing
        self.smoothing = smoothing
        self.frame_count = 0
        self.count_zone = count_zone

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _iou(self, b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        xi1, yi1 = max(x11, x21), max(y11, y21)
        xi2, yi2 = min(x12, x22), min(y12, y22)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _in_count_zone(self, center):
        if self.count_zone is None:
            return True
        x1, y1, x2, y2 = self.count_zone
        return (x1 <= center[0] <= x2) and (y1 <= center[1] <= y2)

    def update(self, detections):
        self.frame_count += 1

        # 1) 预测阶段：仅保留未超期 track
        active_tracks = {}
        for tid, t in self.tracks.items():
            if self.frame_count - t['last_seen'] <= self.max_frames_missing:
                active_tracks[tid] = t
        self.tracks = active_tracks

        if not detections:
            return []

        # 2) IOU 匹配
        unmatched_dets = set(range(len(detections)))
        for tid, t in list(self.tracks.items()):
            best_det = None
            best_iou = 0.0
            for i in unmatched_dets:
                iou_val = self._iou(t['bbox'], detections[i]['bbox'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = i
            if best_iou >= self.iou_thresh and best_det is not None:
                det = detections[best_det]
                # 平滑 bbox
                new_bbox = [
                    int(self.smoothing * det['bbox'][0] + (1 - self.smoothing) * t['bbox'][0]),
                    int(self.smoothing * det['bbox'][1] + (1 - self.smoothing) * t['bbox'][1]),
                    int(self.smoothing * det['bbox'][2] + (1 - self.smoothing) * t['bbox'][2]),
                    int(self.smoothing * det['bbox'][3] + (1 - self.smoothing) * t['bbox'][3]),
                ]
                new_center = self._bbox_center(new_bbox)
                t['bbox'] = new_bbox
                t['center'] = new_center
                t['last_seen'] = self.frame_count
                t['history'].append(new_center)
                if len(t['history']) > 30:
                    t['history'] = t['history'][-30:]
                # 更新方向
                if len(t['history']) >= 2:
                    prev_y = t['history'][-2][1]
                    curr_y = t['history'][-1][1]
                    t['direction'] = 'up' if curr_y < prev_y else 'down'
                unmatched_dets.remove(best_det)

        # 3) 新建 track
        for i in list(unmatched_dets):
            det = detections[i]
            center = self._bbox_center(det['bbox'])
            self.tracks[self.next_id] = {
                'bbox': det['bbox'],
                'center': center,
                'last_seen': self.frame_count,
                'history': [center],
                'direction': 'unknown',
                'counted': False
            }
            self.next_id += 1

        # 4) 输出
        updated_tracks = []
        for tid, t in self.tracks.items():
            updated_tracks.append({
                'track_id': tid,
                'bbox': t['bbox'],
                'center': t['center'],
                'history': t['history'],
                'direction': t.get('direction', 'unknown'),
                'counted': t.get('counted', False),
            })
        return updated_tracks


def check_crossing_strict_direction(tracked_objects, count_line_y, counted_objects, counting_direction, count_zone=None):
    new_counts = 0
    for obj in tracked_objects:
        track_id = obj['track_id']
        if track_id in counted_objects:
            continue
        history = obj.get('history', [])
        if len(history) < 2:
            continue
        prev_y = history[-2][1]
        curr_y = history[-1][1]
        obj_direction = obj.get('direction', 'unknown')
        center_now = history[-1]

        # 区域约束
        if count_zone:
            x1, y1, x2, y2 = count_zone
            if not (x1 <= center_now[0] <= x2 and y1 <= center_now[1] <= y2):
                continue

        if counting_direction == "up":
            if prev_y > count_line_y and curr_y <= count_line_y and obj_direction == 'up':
                counted_objects.add(track_id)
                obj['counted'] = True
                new_counts += 1
        elif counting_direction == "down":
            if prev_y < count_line_y and curr_y >= count_line_y and obj_direction == 'down':
                counted_objects.add(track_id)
                obj['counted'] = True
                new_counts += 1
    return new_counts, counted_objects


# ====== YOLO 推理：engine / onnx / pt 兼容 ======
class BaseDetector:
    def infer(self, frame_bgr, roi=None, downsample_ratio=1.0):
        raise NotImplementedError


class TRTDetector(BaseDetector):
    def __init__(self, engine_path, input_size=640, conf_thres=0.15, iou_thres=0.10, max_det=MAX_DETECTIONS):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT/pycuda 不可用，无法加载 .engine")
        self.engine_path = engine_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.input_binding = None

        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append((binding, host_mem, device_mem))
                if self.input_binding is None:
                    self.input_binding = binding
            else:
                self.outputs.append((binding, host_mem, device_mem))

    def _ensure_input_shape(self, inp_shape):
        """为动态shape的engine设置实际输入形状，避免执行时形状缺失导致的Cask报错。"""
        if self.input_binding is None:
            return
        try:
            # TensorRT 10.x
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(self.input_binding, inp_shape)
                return
            # TensorRT 8.x
            if hasattr(self.context, "set_binding_shape"):
                if hasattr(self.engine, "get_tensor_index"):
                    idx = self.engine.get_tensor_index(self.input_binding)
                else:
                    idx = self.engine.get_binding_index(self.input_binding)
                self.context.set_binding_shape(idx, inp_shape)
        except Exception as e:
            raise RuntimeError(f"设置TensorRT输入形状失败: {e}")

    def _preprocess(self, img_bgr):
        """优化的预处理：合并色彩空间转换和resize，减少内存拷贝"""
        # 使用cv2.resize的插值参数来加速（INTER_LINEAR比INTER_CUBIC快，但质量足够）
        # 先resize再转换色彩空间，可以减少处理的像素数量
        img = cv2.resize(img_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 合并归一化和转置操作
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(img)

    def infer(self, frame_bgr, roi=None, downsample_ratio=1.0):
        # 如果没有ROI，则全屏识别
        if roi is None:
            img = frame_bgr
            roi_top, roi_left = 0, 0
            # 不启用计数时，使用全屏，但为了统一，我们设置roi为全屏
            roi_width, roi_height = img.shape[1], img.shape[0]
        else:
            x, y, w, h = roi
            img = frame_bgr[y:y + h, x:x + w]
            roi_top, roi_left = y, x
            roi_width, roi_height = w, h
        
        # 应用下采样 - 优化：使用INTER_LINEAR以提升速度
        if downsample_ratio != 1.0 and downsample_ratio > 0:
            orig_h, orig_w = img.shape[:2]
            new_w = int(orig_w * downsample_ratio)
            new_h = int(orig_h * downsample_ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_factor = 1.0 / downsample_ratio
        else:
            scale_factor = 1.0
            
        if img is None or img.size == 0:
            return []

        inp = self._preprocess(img)
        # 对动态shape的engine显式设置输入形状，避免执行阶段抛出Cask convolution错误
        self._ensure_input_shape(inp.shape)
        _, in_host, in_dev = self.inputs[0]
        np.copyto(in_host, inp.ravel())
        cuda.memcpy_htod_async(in_dev, in_host, self.stream)
        
        # TensorRT版本自适应执行
        try:
            # 先尝试TensorRT 8.x的API
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        except AttributeError:
            try:
                # 尝试TensorRT 10.x的execute_v2（同步）
                self.context.execute_v2(bindings=self.bindings)
                # 同步执行后需要同步流
                self.stream.synchronize()
            except AttributeError:
                try:
                    # 尝试TensorRT 10.x的enqueue_v3（异步）
                    self.context.enqueue_v3(self.stream)
                except AttributeError:
                    # 最后尝试旧版本API
                    self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 无论哪种方式执行，都需要同步流以确保输出数据就绪
        for _, out_host, out_dev in self.outputs:
            cuda.memcpy_dtoh_async(out_host, out_dev, self.stream)
        self.stream.synchronize()

        out = np.array(self.outputs[0][1])

        dets = []
        # 处理YOLO26可能的多种输出格式
        # 格式1: (N, 6) [x1, y1, x2, y2, conf, cls] - 传统格式
        # 格式2: (N, 5+num_classes) [x1, y1, x2, y2, conf, cls0, cls1, ...] - YOLO26可能格式
        
        # 首先尝试传统格式
        if out.size % 6 == 0:
            out = out.reshape(-1, 6)
            out = out[:self.max_det]  # 限制检测数量
            for row in out:
                x1, y1, x2, y2, conf, cls = row
                conf = float(conf)
                if conf < self.conf_thres:
                    continue
                # 缩放回原始ROI尺寸
                x1 = int(x1 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                x2 = int(x2 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                y1 = int(y1 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                y2 = int(y2 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                dets.append({'bbox': [x1, y1, x2, y2], 'confidence': conf, 'class': int(cls), 'label': f'class_{int(cls)}'})
        # 尝试YOLO26格式 (5+num_classes)
        elif out.size > 0 and out.size % out.shape[-1] == 0:
            num_cols = out.shape[-1]
            if num_cols >= 6:  # 至少有5个框坐标+1个置信度
                out = out[:self.max_det]  # 限制检测数量
                for row in out.reshape(-1, num_cols):
                    x1, y1, x2, y2, obj_conf = row[0:5]
                    # 获取类别分数
                    cls_scores = row[5:]
                    cls_id = np.argmax(cls_scores)
                    cls_conf = float(cls_scores[cls_id])
                    # 综合置信度 = 对象置信度 * 类别置信度
                    final_conf = float(obj_conf) * cls_conf
                    if final_conf < self.conf_thres:
                        continue
                    # 缩放回原始ROI尺寸
                    x1 = int(x1 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                    x2 = int(x2 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                    y1 = int(y1 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                    y2 = int(y2 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                    dets.append({'bbox': [x1, y1, x2, y2], 'confidence': final_conf, 'class': int(cls_id), 'label': f'class_{int(cls_id)}'})
        
        # 可选的NMS处理（对于端到端模型可以关闭）
        if USE_NMS_FOR_YOLO26 and len(dets) > 0:
            dets = self._apply_nms(dets)
            
        return dets
    
    def _apply_nms(self, detections):
        """应用非极大值抑制"""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # 使用OpenCV的NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.conf_thres, self.iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []


class OnnxDetector(BaseDetector):
    def __init__(self, onnx_path, input_size=640, conf_thres=0.15, iou_thres=0.10, max_det=MAX_DETECTIONS):
        import onnxruntime as ort
        # 创建会话选项，设置日志级别减少输出
        so = ort.SessionOptions()
        so.log_severity_level = 3  # 3=ERROR, 2=WARNING, 1=INFO, 0=VERBOSE
        
        self.session = ort.InferenceSession(onnx_path, sess_options=so, 
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def _preprocess(self, img_bgr):
        """优化的预处理：先resize再转换色彩空间"""
        img = cv2.resize(img_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(img)

    def infer(self, frame_bgr, roi=None, downsample_ratio=1.0):
        # 如果没有ROI，则全屏识别
        if roi is None:
            img = frame_bgr
            roi_top, roi_left = 0, 0
        else:
            x, y, w, h = roi
            img = frame_bgr[y:y + h, x:x + w]
            roi_top, roi_left = y, x
        
        # 应用下采样 - 优化：使用INTER_LINEAR以提升速度
        if downsample_ratio != 1.0 and downsample_ratio > 0:
            orig_h, orig_w = img.shape[:2]
            new_w = int(orig_w * downsample_ratio)
            new_h = int(orig_h * downsample_ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_factor = 1.0 / downsample_ratio
        else:
            scale_factor = 1.0
            
        if img is None or img.size == 0:
            return []

        inp = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: inp})
        out = np.array(outputs[0])

        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        dets = []
        # 处理多种输出格式，支持YOLO26
        if out.size % 6 == 0:
            # 传统格式 (N, 6)
            out = out.reshape(-1, 6)
            out = out[:self.max_det]  # 限制检测数量
            for x1, y1, x2, y2, conf, cls in out:
                conf = float(conf)
                if conf < self.conf_thres:
                    continue
                # 缩放回原始ROI尺寸
                x1 = int(x1 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                x2 = int(x2 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                y1 = int(y1 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                y2 = int(y2 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                dets.append({'bbox': [x1, y1, x2, y2], 'confidence': conf, 'class': int(cls), 'label': f'class_{int(cls)}'})
        elif out.size > 0:
            # 尝试处理YOLO26格式 (N, 5+num_classes)
            try:
                num_cols = out.shape[-1]
                if num_cols >= 6:
                    out = out[:self.max_det]  # 限制检测数量
                    for row in out.reshape(-1, num_cols):
                        x1, y1, x2, y2, obj_conf = row[0:5]
                        cls_scores = row[5:]
                        cls_id = np.argmax(cls_scores)
                        cls_conf = float(cls_scores[cls_id])
                        final_conf = float(obj_conf) * cls_conf
                        if final_conf < self.conf_thres:
                            continue
                        # 缩放回原始ROI尺寸
                        x1 = int(x1 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                        x2 = int(x2 * (img.shape[1] / self.input_size) * scale_factor) + roi_left
                        y1 = int(y1 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                        y2 = int(y2 * (img.shape[0] / self.input_size) * scale_factor) + roi_top
                        dets.append({'bbox': [x1, y1, x2, y2], 'confidence': final_conf, 'class': int(cls_id), 'label': f'class_{int(cls_id)}'})
            except Exception as e:
                print(f"⚠️ 解析YOLO26输出时出错: {e}")
        
        # 可选的NMS处理
        if USE_NMS_FOR_YOLO26 and len(dets) > 0:
            dets = self._apply_nms(dets)
            
        return dets
    
    def _apply_nms(self, detections):
        """应用非极大值抑制"""
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.conf_thres, self.iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []


class PtDetector(BaseDetector):
    def __init__(self, pt_path, input_size=640, conf_thres=0.15, iou_thres=0.10, max_det=MAX_DETECTIONS, use_cpu=False):
        if not TORCH_AVAILABLE or not YOLOV5_CODE_AVAILABLE:
            raise RuntimeError("Torch 或 yolov5 代码不可用，无法加载 .pt")
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        device = torch.device('cpu') if use_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(pt_path, device=device)
        self.names = self.model.names

    def infer(self, frame_bgr, roi=None, downsample_ratio=1.0):
        # 如果没有ROI，则全屏识别
        if roi is None:
            img0 = frame_bgr
            roi_top, roi_left = 0, 0
        else:
            x, y, w, h = roi
            img0 = frame_bgr[y:y + h, x:x + w]
            roi_top, roi_left = y, x
            
        # 应用下采样 - 优化：使用INTER_LINEAR以提升速度
        if downsample_ratio != 1.0 and downsample_ratio > 0:
            orig_h, orig_w = img0.shape[:2]
            new_w = int(orig_w * downsample_ratio)
            new_h = int(orig_h * downsample_ratio)
            img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_factor = 1.0 / downsample_ratio
        else:
            scale_factor = 1.0
            
        if img0 is None or img0.size == 0:
            return []

        img_lb = letterbox(img0, self.input_size, stride=self.model.stride, auto=True)[0]
        img_chw = img_lb.transpose((2, 0, 1))[::-1]
        img_chw = np.ascontiguousarray(img_chw)

        im = torch.from_numpy(img_chw).to(self.model.device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)

        dets = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(conf)
                    cls = int(cls)
                    # 缩放回原始ROI尺寸
                    x1 = int(x1 * scale_factor) + roi_left
                    x2 = int(x2 * scale_factor) + roi_left
                    y1 = int(y1 * scale_factor) + roi_top
                    y2 = int(y2 * scale_factor) + roi_top
                    dets.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls,
                        'label': self.names[cls] if self.names else f'class_{cls}'
                    })
        return dets


def choose_model_file(default_path=DEFAULT_MODEL_PATH):
    if default_path and os.path.isfile(default_path):
        return default_path
    for p in MODEL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def create_detector(model_path):
    if model_path is None:
        raise RuntimeError("未找到模型文件，请放置 yolov12sbest.engine 或其他候选文件")
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".engine":
        print(f"🔧 使用 TensorRT Engine: {model_path}")
        return TRTDetector(model_path)
    if ext == ".onnx":
        print(f"🔧 使用 ONNXRuntime: {model_path}")
        return OnnxDetector(model_path)
    if ext == ".pt":
        print(f"🔧 使用 PyTorch(.pt): {model_path}")
        return PtDetector(model_path)
    raise RuntimeError(f"不支持的模型后缀: {ext}")


# ===== 密度识别 =====
class DensityDetector:
    def __init__(self, cols=16, rows=6, low_ratio=0.35):
        self.cols = cols
        self.rows = rows
        self.low_ratio = low_ratio
        self.in_low = False
        self.low_cnt = 0
        self.recover_cnt = 0
        self.current_start_s = None
        self.current_seed_sum = 0  # 低密度区间内的种子累计（用于重量换算）
        self.current_frames = 0

    def compute_density_map(self, detections, roi_rect):
        x, y, w, h = roi_rect
        dm = np.zeros((self.rows, self.cols), dtype=np.float32)
        if w <= 1 or h <= 1:
            return dm

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if cx < x or cx >= x + w or cy < y or cy >= y + h:
                continue
            gx = int((cx - x) / w * self.cols)
            gy = int((cy - y) / h * self.rows)
            gx = max(0, min(self.cols - 1, gx))
            gy = max(0, min(self.rows - 1, gy))
            dm[gy, gx] += 1.0
        return dm

    def low_density_columns(self, density_map):
        col_sum = density_map.sum(axis=0)
        mean = float(col_sum.mean()) if col_sum.size else 0.0
        if mean <= 1e-6:
            return [], col_sum, mean, 0.0
        thr = mean * self.low_ratio
        low = col_sum < thr

        segs = []
        start = None
        for i, is_low in enumerate(low):
            if is_low and start is None:
                start = i
            elif (not is_low) and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, self.cols - 1))
        return segs, col_sum, mean, thr

    def update_segment_state(self, has_significant_low: bool, s_cap: int, seed_count_frame: int):
        """
        返回一个完整区间：
          (s_start, s_end, seed_count_in_segment)
        """
        if has_significant_low:
            self.low_cnt += 1
            self.recover_cnt = 0
            if (not self.in_low) and self.low_cnt >= LOW_DENSITY_MIN_FRAMES:
                self.in_low = True
                self.current_start_s = s_cap
                self.current_seed_sum = 0
                self.current_frames = 0

            if self.in_low:
                self.current_seed_sum += int(seed_count_frame)
                self.current_frames += 1
            return None

        # not low
        self.low_cnt = 0
        if self.in_low:
            self.recover_cnt += 1
            if self.recover_cnt >= RECOVER_MIN_FRAMES:
                self.in_low = False
                self.recover_cnt = 0
                s_start = self.current_start_s
                s_end = s_cap
                seed_sum = self.current_seed_sum
                self.current_start_s = None
                self.current_seed_sum = 0
                self.current_frames = 0
                if s_start is not None and s_end is not None and s_end >= s_start:
                    return (int(s_start), int(s_end), int(seed_sum))
        return None

    def draw_density_map(self, frame, density_map, roi_rect):
        x, y, w, h = roi_rect
        dm = density_map
        dm_norm = dm.copy()
        if dm_norm.max() > 0:
            dm_norm = dm_norm / dm_norm.max()
        dm_img = (dm_norm * 255).astype(np.uint8)
        dm_img = cv2.resize(dm_img, (w, h), interpolation=cv2.INTER_NEAREST)
        dm_color = cv2.applyColorMap(dm_img, cv2.COLORMAP_JET)
        overlay = frame.copy()
        overlay[y:y + h, x:x + w] = cv2.addWeighted(frame[y:y + h, x:x + w], 0.5, dm_color, 0.5, 0)
        return overlay


# ===== 低密度区间缓存（异步写盘 + 轮转）=====
class SegmentCache(threading.Thread):
    def __init__(self, path, merge_gap_mm=80,
                 queue_max=CACHE_QUEUE_MAX,
                 flush_interval=CACHE_FLUSH_INTERVAL,
                 rotate_lines=CACHE_ROTATE_LINES,
                 rotate_keep=CACHE_ROTATE_KEEP):
        super().__init__(daemon=True)
        self.path = path
        self.merge_gap_mm = merge_gap_mm
        self.lock = threading.Lock()
        self._last_seg = None  # (s_start, s_end, seed_count_sum, weight_mg)
        self.q = queue.Queue(maxsize=queue_max)
        self.flush_interval = flush_interval
        self.rotate_lines = rotate_lines
        self.rotate_keep = rotate_keep
        self.stop_flag = threading.Event()
        self._pending = []
        self._line_count = 0
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._line_count = sum(1 for _ in f)
            except Exception:
                self._line_count = 0

    def rotate_if_needed(self):
        if self.rotate_lines <= 0:
            return
        if self._line_count < self.rotate_lines:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        new_name = f"{self.path}.{ts}"
        try:
            os.rename(self.path, new_name)
            print(f"🌀 cache rotate -> {new_name}")
        except Exception:
            pass
        self._line_count = 0
        try:
            files = sorted([f for f in os.listdir(".") if f.startswith(os.path.basename(self.path) + ".")])
            if len(files) > self.rotate_keep:
                for f in files[:-self.rotate_keep]:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass

    def add_segment(self, s_start, s_end, seed_count_sum: int, weight_mg: int, meta=None):
        if s_end < s_start:
            return

        with self.lock:
            seg = (int(s_start), int(s_end), int(seed_count_sum), int(weight_mg))
            if self._last_seg is None:
                self._last_seg = seg
                self._enqueue(seg, meta)
                return

            last_start, last_end, last_seed, last_wmg = self._last_seg
            if seg[0] <= last_end + self.merge_gap_mm:
                new_seg = (last_start, max(last_end, seg[1]), last_seed + seg[2], last_wmg + seg[3])
                self._last_seg = new_seg
                self._enqueue(new_seg, {**(meta or {}), "merged": True})
            else:
                self._last_seg = seg
                self._enqueue(seg, meta)

    def _enqueue(self, seg, meta=None):
        s_start, s_end, seed_count_sum, weight_mg = seg
        record = {
            "ts_host": time.time(),
            "s_start_mm": s_start,
            "s_end_mm": s_end,
            "seed_count": seed_count_sum,
            "weight_g": weight_mg / 1000.0,
            "tkw_g_per_1000": SEED_TKW_GRAMS,
        }
        if meta:
            record.update(meta)
        try:
            self.q.put(record, timeout=0.01)
        except queue.Full:
            print("⚠️ cache 队列已满，丢弃一条记录。")

    def run(self):
        last_flush = time.time()
        while not self.stop_flag.is_set():
            try:
                rec = self.q.get(timeout=0.2)
                self._pending.append(rec)
                self.q.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if (now - last_flush) >= self.flush_interval or len(self._pending) >= 100:
                self.flush()
                last_flush = now

        self.flush()

    def flush(self):
        if not self._pending:
            return
        try:
            self.rotate_if_needed()
            with open(self.path, "a", encoding="utf-8") as f:
                for rec in self._pending:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._line_count += 1
            self._pending.clear()
        except Exception as e:
            print(f"⚠️ cache 写入失败: {e}")

    def stop(self):
        self.stop_flag.set()


# ===== UI 线程 =====
class CameraThread(QThread):
    """相机采集和处理的线程"""
    frame_processed = pyqtSignal(np.ndarray, dict)  # 发送处理后的帧和统计数据
    long_status_changed = pyqtSignal(str)          # 长图状态变化（用于UI显示）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cam = None
        self.detector = None
        self.tracker = None
        self.ctrl_buf = None
        self.density_detector = None
        self.event_sender = None
        self.uart_rx = None
        self.seg_cache = None
        
        # 可配置参数
        self.enable_detection = True
        self.enable_tracking = True
        self.enable_density_detection = DENSITY_ENABLED_DEFAULT
        self.enable_uart_rx = True
        self.enable_uart_tx = True
        self.enable_counting = True
        
        # 计数线参数
        self.count_line_percent = COUNT_LINE_PERCENT_DEFAULT  # 计数线位置百分比
        self.count_line_top_extend = COUNT_LINE_TOP_EXTEND_DEFAULT  # 向上扩展像素数
        self.count_line_bottom_extend = COUNT_LINE_BOTTOM_EXTEND_DEFAULT  # 向下扩展像素数
        
        # ROI区域
        self.roi_rect = None  # 启用计数时为ROI区域，不启用时为None（全屏识别）
        
        # 下采样参数
        self.downsample_ratio = DOWNSAMPLE_RATIO_DEFAULT
        
        # 帧跳过参数（用于极端性能优化）
        self.frame_skip = 0  # 0=处理所有帧，1=跳过1帧处理1帧，2=跳过2帧处理1帧
        self.frame_skip_counter = 0
        
        # 性能监控（可选）
        self.enable_profiling = False
        self.profile_times = {
            'capture': 0.0,
            'detection': 0.0,
            'tracking': 0.0,
            'drawing': 0.0,
            'total': 0.0,
        }
        
        # 统计信息
        self.stats = {
            'fps': 0,
            'detections': 0,
            'tracked': 0,
            'counted': 0,
            'low_density': False,
            'gate_state': False,
            's_mm': 0,
            'v_mmps': 0,
            'uart_rx_status': '未连接',
            'uart_tx_status': '未连接',
            'long_status': '空闲',
            'count_line_y': 0,
            'roi_info': '全屏',
        }
        
        # 其他变量
        self.counted_objects = set()
        self.event_id = 1
        self.count_line_y = 0  # 计数线像素位置
        self.counting_direction = "down"
        self.count_zone = None
        self.max_detections = MAX_DETECTIONS  # 最大检测数量
        
        # 性能监控
        self.frame_count = 0
        self.last_fps_time = time.time()

        # 长图拼接
        self.long_capturing = False
        self.long_frames = []
        self.base_save_dir = DEFAULT_SAVE_DIR
        self.long_reason = ""
        self.manual_allowed = True  # 当 UART 关闭时由 UI 控制
        self._prev_speed_positive = False
        
        # 原始帧尺寸
        self.original_frame_height = 720
        self.original_frame_width = 1280
    
    def set_base_save_dir(self, path: str):
        if path:
            self.base_save_dir = path
        else:
            self.base_save_dir = DEFAULT_SAVE_DIR
    
    def update_count_line_and_roi(self, frame):
        """根据计数线百分比和扩展像素数更新计数线位置和ROI区域"""
        if frame is None:
            return
        
        self.original_frame_height, self.original_frame_width = frame.shape[:2]
        
        # 计算计数线像素位置（从上到下百分比）
        self.count_line_y = int(self.original_frame_height * (self.count_line_percent / 100.0))
        
        if self.enable_counting:
            # 启用计数时，计算ROI区域
            roi_y = max(0, self.count_line_y - self.count_line_top_extend)
            roi_h = min(self.original_frame_height - roi_y, 
                       self.count_line_top_extend + self.count_line_bottom_extend)
            roi_x = 0
            roi_w = self.original_frame_width
            
            # 确保ROI区域有效
            if roi_h > 0 and roi_w > 0:
                self.roi_rect = (roi_x, roi_y, roi_w, roi_h)
                # 设置计数区域为ROI区域
                self.count_zone = self.roi_rect
                self.stats['roi_info'] = f"{roi_w}x{roi_h}"
            else:
                self.roi_rect = None
                self.count_zone = None
                self.stats['roi_info'] = '无效'
        else:
            # 不启用计数时，ROI为None（全屏识别）
            self.roi_rect = None
            self.count_zone = None
            self.stats['roi_info'] = '全屏'
        
        self.stats['count_line_y'] = self.count_line_y
    
    def start_manual_long_capture(self):
        if self.enable_uart_rx and SERIAL_AVAILABLE:
            return  # UART 开启时不允许手动
        self._start_long_capture(reason="manual")

    def stop_manual_long_capture(self):
        if self.enable_uart_rx and SERIAL_AVAILABLE:
            return  # UART 开启时不允许手动
        self._finalize_long_capture(trigger="manual_stop")
    
    def _start_long_capture(self, reason: str):
        if self.long_capturing:
            return
        self.long_capturing = True
        self.long_reason = reason
        self.long_frames = []
        self.long_status_changed.emit(f"拼接中 ({reason})")
        self.stats['long_status'] = f"拼接中 ({reason})"

    def _append_long_frame(self, frame):
        if not self.long_capturing:
            return
        if len(self.long_frames) >= MAX_LONG_FRAMES:
            return
        self.long_frames.append(frame.copy())

    def _finalize_long_capture(self, trigger: str):
        if not self.long_capturing:
            return
        self.long_capturing = False
        frames = self.long_frames
        self.long_frames = []
        if not frames:
            self.long_status_changed.emit("空闲")
            self.stats['long_status'] = "空闲"
            return
        try:
            long_img = np.vstack(frames)
        except Exception as e:
            print(f"⚠️ 长图拼接失败: {e}")
            self.long_status_changed.emit("空闲")
            self.stats['long_status'] = "空闲"
            return
        if long_img.shape[0] < MIN_HEIGHT_FOR_SAVE:
            self.long_status_changed.emit("空闲")
            self.stats['long_status'] = "空闲"
            return

        detections = []
        annotated = long_img.copy()
        try:
            if self.detector:
                # 长图识别时使用全屏识别
                detections = self.detector.infer(long_img, roi=None, downsample_ratio=self.downsample_ratio)
                annotated = self._draw_simple_dets(long_img.copy(), detections)
        except Exception as e:
            print(f"⚠️ 长图整体识别失败: {e}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self.base_save_dir, ts)
        os.makedirs(folder, exist_ok=True)
        long_path = os.path.join(folder, "long.jpg")
        det_path = os.path.join(folder, "long_det.jpg")
        meta_path = os.path.join(folder, "result.json")
        try:
            cv2.imwrite(long_path, long_img)
            cv2.imwrite(det_path, annotated)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"trigger": trigger, "detections": detections}, f, ensure_ascii=False, indent=2)
            print(f"📸 长图已保存: {long_path}")
            print(f"✅ 识别结果已保存: {det_path}")
        except Exception as e:
            print(f"⚠️ 保存长图或结果失败: {e}")

        self.long_status_changed.emit("空闲")
        self.stats['long_status'] = "空闲"

    def _draw_simple_dets(self, img, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det.get('confidence', 0.0)
            label = det.get('label', 'obj')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img
    
    def initialize(self, model_path):
        """初始化所有组件"""
        try:
            # 确保之前的相机资源已释放
            if self.cam and hasattr(self.cam, 'is_opened') and self.cam.is_opened:
                self.cam.close()
                time.sleep(0.5)  # 等待资源释放
            
            # 初始化相机
            self.cam = MVS_Camera()
            if not self.cam.open_camera():
                print("⚠️ 相机打开失败，重试...")
                time.sleep(1)
                if not self.cam.open_camera():
                    raise RuntimeError("相机打开失败")
            print("✅ 相机初始化成功")
            
            # 初始化检测器
            if model_path:
                self.detector = create_detector(model_path)
                print(f"✅ 检测器初始化成功: {model_path}")
            else:
                raise RuntimeError("未指定模型路径")
            
            # 初始化追踪器
            self.tracker = EnhancedTracker(iou_thresh=0.3, max_frames_missing=15, 
                                          smoothing=0.5, count_zone=self.count_zone)
            print("✅ 追踪器初始化成功")
            
            # 初始化控制缓冲区
            self.ctrl_buf = CtrlBuffer(keep_seconds=CTRL_BUFFER_SECONDS)
            print("✅ 控制缓冲区初始化成功")
            
            # 初始化UART接收器
            self.uart_rx = None
            if self.enable_uart_rx and SERIAL_AVAILABLE:
                try:
                    self.uart_rx = UartCtrlReceiver(UART_PORT, UART_BAUD, self.ctrl_buf)
                    self.uart_rx.start()
                    self.stats['uart_rx_status'] = '运行中'
                    print("✅ UART接收器已启动")
                except Exception as e:
                    print(f"⚠️ UART接收器启动失败: {e}")
                    self.stats['uart_rx_status'] = '启动失败'
            elif self.enable_uart_rx and (not SERIAL_AVAILABLE):
                self.stats['uart_rx_status'] = '不可用'
            
            # 初始化UART发送器
            self.event_sender = None
            if self.enable_uart_tx and SERIAL_AVAILABLE:
                try:
                    self.event_sender = UartEventSender(UART_PORT, UART_BAUD)
                    self.event_sender.start()
                    self.stats['uart_tx_status'] = '运行中'
                    print("✅ UART发送器已启动")
                except Exception as e:
                    print(f"⚠️ UART发送器启动失败: {e}")
                    self.stats['uart_tx_status'] = '启动失败'
            elif self.enable_uart_tx and (not SERIAL_AVAILABLE):
                self.stats['uart_tx_status'] = '不可用'
            
            # 初始化密度检测器
            self.density_detector = None
            if self.enable_density_detection:
                self.density_detector = DensityDetector(cols=DENSITY_GRID_COLS, 
                                                       rows=DENSITY_GRID_ROWS, 
                                                       low_ratio=LOW_DENSITY_RATIO)
                print("✅ 密度检测器已初始化")
            
            # 初始化缓存
            self.seg_cache = None
            try:
                self.seg_cache = SegmentCache(EVENT_CACHE_FILE, 
                                             merge_gap_mm=EVENT_MERGE_GAP_MM,
                                             queue_max=CACHE_QUEUE_MAX,
                                             flush_interval=CACHE_FLUSH_INTERVAL,
                                             rotate_lines=CACHE_ROTATE_LINES,
                                             rotate_keep=CACHE_ROTATE_KEEP)
                self.seg_cache.start()
                print("✅ 缓存系统已启动")
            except Exception as e:
                print(f"⚠️ 缓存系统启动失败: {e}")
            
            # 重置计数
            self.counted_objects.clear()
            self.event_id = 1
            
            print("✅ 系统初始化完成")
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            # 清理已创建的资源
            self._cleanup_resources()
            return False
    
    def _cleanup_resources(self):
        """清理资源"""
        try:
            if self.cam:
                self.cam.close()
            if hasattr(self, 'uart_rx') and self.uart_rx:
                self.uart_rx.stop()
            if hasattr(self, 'event_sender') and self.event_sender:
                self.event_sender.stop()
            if self.seg_cache:
                self.seg_cache.stop()
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def run(self):
        """线程主循环"""
        self.running = True
        
        while self.running:
            try:
                t_loop_start = time.perf_counter() if self.enable_profiling else 0
                
                # 采集帧
                t_capture_start = time.perf_counter() if self.enable_profiling else 0
                if not self.cam:
                    time.sleep(0.01)  # Reduced from 0.1s to 10ms
                    continue
                    
                frame = self.cam.capture_frame_alternative()
                if frame is None:
                    time.sleep(0.001)  # Reduced from 0.01s to 1ms
                    continue
                
                if self.enable_profiling:
                    self.profile_times['capture'] = time.perf_counter() - t_capture_start
                
                # 帧跳过逻辑（仅在检测和追踪时跳过，始终更新显示）
                skip_processing = False
                if self.frame_skip > 0:
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter <= self.frame_skip:
                        skip_processing = True
                    else:
                        self.frame_skip_counter = 0
                
                # 更新原始帧尺寸
                self.original_frame_height, self.original_frame_width = frame.shape[:2]
                
                # 更新计数线位置和ROI区域
                self.update_count_line_and_roi(frame)
                
                # 更新FPS - 计算实际处理帧率
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                if elapsed >= 1.0:
                    # 计算每秒实际处理帧数
                    self.stats['fps'] = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                t_cap_host = time.monotonic()
                
                # 时间对齐 - 安全访问 ctrl_buf
                s_cap, v_cap, age = (0, 0, 999)
                if self.ctrl_buf:
                    result = self.ctrl_buf.query_s_at_host_time(t_cap_host)
                    if result[0] is not None:
                        s_cap, v_cap, age = result
                
                # 门控状态：
                # - 如果 UART 接收关闭或串口不可用，则默认门控开启（允许识别）
                # - 否则按原有门控逻辑基于最新 CTRL 数据
                if (not self.enable_uart_rx) or (not SERIAL_AVAILABLE):
                    gate_state = True
                else:
                    gate_state = False
                    if age <= GATE_MAX_AGE and self.ctrl_buf:
                        gate_state = self.ctrl_buf.gate_enabled(max_age=GATE_MAX_AGE)
                
                self.stats['gate_state'] = gate_state
                self.stats['s_mm'] = s_cap
                self.stats['v_mmps'] = v_cap
                
                # 长图拼接逻辑
                if self.enable_uart_rx and SERIAL_AVAILABLE:
                    # UART 模式：速度>0 开始/继续拼接；<=0 停止并识别
                    if v_cap > 0:
                        if not self.long_capturing:
                            self._start_long_capture(reason="uart_positive_speed")
                        self._append_long_frame(frame)
                    else:
                        if self.long_capturing and self.long_reason == "uart_positive_speed":
                            self._finalize_long_capture(trigger="uart_speed_non_positive")
                # UART 关闭时不自动控制，由快捷键/按钮触发

                # 检测 - 关键修改：启用计数时只识别ROI区域，不启用时全屏识别
                detections = []
                t_detect_start = time.perf_counter() if self.enable_profiling else 0
                
                if not skip_processing and self.enable_detection and gate_state and self.detector:
                    try:
                        # 根据是否启用计数决定识别区域
                        detections = self.detector.infer(frame, self.roi_rect, self.downsample_ratio)
                        # 应用最大检测数量限制
                        if len(detections) > self.max_detections:
                            detections = detections[:self.max_detections]
                    except Exception as e:
                        print(f"⚠️ 检测失败: {e}")
                
                if self.enable_profiling:
                    self.profile_times['detection'] = time.perf_counter() - t_detect_start
                
                self.stats['detections'] = len(detections)
                
                # 追踪
                tracked = []
                t_track_start = time.perf_counter() if self.enable_profiling else 0
                
                if not skip_processing and self.enable_tracking and self.tracker:
                    tracked = self.tracker.update(detections)
                
                if self.enable_profiling:
                    self.profile_times['tracking'] = time.perf_counter() - t_track_start
                
                self.stats['tracked'] = len(tracked)
                
                # 过线计数
                new_cnt = 0
                if self.enable_counting and self.enable_tracking:
                    new_cnt, self.counted_objects = check_crossing_strict_direction(
                        tracked, self.count_line_y, self.counted_objects, 
                        self.counting_direction, count_zone=self.count_zone
                    )
                    self.stats['counted'] = len(self.counted_objects)
                
                # 密度检测
                if self.enable_density_detection and self.density_detector:
                    total_dets = len(detections)
                    if total_dets >= MIN_DETS_FOR_DENSITY:
                        density_map = self.density_detector.compute_density_map(detections, self.roi_rect)
                        segs, col_sum, mean, thr = self.density_detector.low_density_columns(density_map)
                        has_low = len(segs) > 0
                    else:
                        has_low = False
                    
                    self.stats['low_density'] = has_low
                    
                    seg = self.density_detector.update_segment_state(has_low, s_cap, 
                                                                    seed_count_frame=len(detections))
                    if seg and self.event_sender:
                        try:
                            s_start, s_end, seed_sum = seg
                            weight_g = seed_count_to_grams(seed_sum, SEED_TKW_GRAMS)
                            weight_mg = grams_to_mg_u32(weight_g)
                            
                            if self.seg_cache:
                                self.seg_cache.add_segment(
                                    s_start=s_start,
                                    s_end=s_end,
                                    seed_count_sum=seed_sum,
                                    weight_mg=weight_mg
                                )
                            
                            self.event_sender.send_low_density_segment(
                                event_id=self.event_id,
                                severity=255,
                                s_start_mm=s_start,
                                s_end_mm=s_end,
                                seed_count=seed_sum,
                                weight_mg=weight_mg
                            )
                            self.event_id = (self.event_id + 1) & 0xFFFF
                        except Exception as e:
                            print(f"⚠️ 发送密度事件失败: {e}")
                
                # 绘制检测结果（不显示统计信息）
                t_draw_start = time.perf_counter() if self.enable_profiling else 0
                display_frame = self.draw_detections_simple(frame, detections, tracked, gate_state)
                if self.enable_profiling:
                    self.profile_times['drawing'] = time.perf_counter() - t_draw_start
                
                # 发送处理后的帧和统计信息
                self.frame_processed.emit(display_frame, self.stats.copy())
                
                # 记录完整循环时间
                if self.enable_profiling:
                    self.profile_times['total'] = time.perf_counter() - t_loop_start
                    # 可选：定期打印性能数据
                    if self.frame_count % 30 == 0:  # 每30帧打印一次
                        print(f"[性能分析] 捕获:{self.profile_times['capture']*1000:.1f}ms "
                              f"检测:{self.profile_times['detection']*1000:.1f}ms "
                              f"追踪:{self.profile_times['tracking']*1000:.1f}ms "
                              f"绘制:{self.profile_times['drawing']*1000:.1f}ms "
                              f"总计:{self.profile_times['total']*1000:.1f}ms "
                              f"FPS:{self.stats['fps']:.1f}")
                
            except Exception as e:
                print(f"处理帧时出错: {e}")
                time.sleep(0.01)  # Reduced from 0.1s
    
    def draw_detections_simple(self, frame, detections, tracked, gate_state):
        """绘制检测和追踪结果（不显示统计信息）"""
        # Optimize: Only copy frame if we need to preserve original
        # Since we're just drawing overlays for display, work directly on the frame
        display_frame = frame
        
        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det.get('label', 'object')
            
            # 绘制边界框
            color = (0, 255, 0) if gate_state else (128, 128, 128)  # 绿色或灰色
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            text = f"{label} {conf:.2f}"
            cv2.putText(display_frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制追踪轨迹
        for track in tracked:
            track_id = track['track_id']
            center = track['center']
            history = track.get('history', [])
            counted = track.get('counted', False)
            
            # 绘制轨迹点
            for i in range(1, len(history)):
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                cv2.line(display_frame, pt1, pt2, (0, 255, 255), 2)
            
            # 绘制ID和中心点
            color = (0, 0, 255) if counted else (255, 0, 0)  # 红色=已计数，蓝色=未计数
            cv2.circle(display_frame, (int(center[0]), int(center[1])), 5, color, -1)
            cv2.putText(display_frame, f"ID:{track_id}", 
                       (int(center[0]) + 10, int(center[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 如果启用计数，绘制ROI区域
        if self.enable_counting and self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            roi_info = f"ROI {w}x{h}"
            cv2.putText(display_frame, roi_info, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 如果启用计数，绘制计数线
        if self.enable_counting:
            if self.roi_rect is not None:
                x, _, w, _ = self.roi_rect
            else:
                x, w = 0, self.original_frame_width
            
            cv2.line(display_frame, (x, self.count_line_y), 
                    (x + w, self.count_line_y), 
                    (0, 0, 255), 2)
            cv2.putText(display_frame, f"Count Line ({self.count_line_percent}%)", 
                       (x + 10, self.count_line_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 绘制计数线扩展信息
            extend_info = f"上扩:{self.count_line_top_extend}px 下扩:{self.count_line_bottom_extend}px"
            if self.roi_rect is not None:
                x, y, w, h = self.roi_rect
                cv2.putText(display_frame, extend_info, (x + 10, y + h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 绘制下采样信息（只在画面右下角显示）
        if self.downsample_ratio != 1.0:
            downsample_text = f"下采样:{self.downsample_ratio}x"
            cv2.putText(display_frame, downsample_text, 
                       (self.original_frame_width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return display_frame
    
    def stop(self):
        """停止线程"""
        self.running = False
        
        # 清理资源 - 使用安全访问方式
        try:
            if self.cam:
                self.cam.close()
        except Exception as e:
            print(f"关闭相机时出错: {e}")
        
        # 安全停止UART接收器
        try:
            if hasattr(self, 'uart_rx') and self.uart_rx:
                self.uart_rx.stop()
        except Exception as e:
            print(f"停止UART接收器时出错: {e}")
        
        # 安全停止UART发送器
        try:
            if hasattr(self, 'event_sender') and self.event_sender:
                self.event_sender.stop()
        except Exception as e:
            print(f"停止UART发送器时出错: {e}")
        
        # 安全停止缓存线程
        try:
            if self.seg_cache:
                self.seg_cache.stop()
        except Exception as e:
            print(f"停止缓存线程时出错: {e}")
        
        self.wait()


# ===== 主窗口 =====
class MainWindow(QMainWindow):
    """主窗口，包含三个页面"""
    
    def __init__(self):
        super().__init__()
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        
        print(f"📺 屏幕尺寸: {self.screen_width}x{self.screen_height}")
        
        # 根据屏幕尺寸设置窗口大小（留出边距）
        window_width = min(1400, self.screen_width - 50)
        window_height = min(900, self.screen_height - 50)
        
        # 窗口设置
        self.setWindowTitle("种子检测系统 - MVS相机 + YOLO检测")
        self.setGeometry(50, 50, window_width, window_height)
        
        # 检测系统组件
        self.camera_thread = None
        self.current_model_path = None
        
        # 配置参数
        self.base_save_dir = DEFAULT_SAVE_DIR
        self.hotkey_start = DEFAULT_HOTKEY_START
        self.hotkey_stop = DEFAULT_HOTKEY_STOP
        
        # 视频显示相关
        self.video_width = 1280  # 默认视频宽度
        self.video_height = 720  # 默认视频高度
        self.display_scale = 1.0  # 显示缩放比例
        self.fit_to_window = True  # 默认适应窗口模式
        
        # 计数线参数
        self.count_line_percent = COUNT_LINE_PERCENT_DEFAULT
        self.count_line_top_extend = COUNT_LINE_TOP_EXTEND_DEFAULT
        self.count_line_bottom_extend = COUNT_LINE_BOTTOM_EXTEND_DEFAULT
        
        # 下采样参数
        self.downsample_ratio = DOWNSAMPLE_RATIO_DEFAULT
        
        # 性能优化参数
        self.frame_skip = 0
        self.enable_profiling = False
        
        # 初始化UI
        self.init_ui()
        
        # 状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)
        
        # 定时器更新UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(UI_UPDATE_INTERVAL)
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 工具栏
        self.toolbar = self.addToolBar("工具栏")
        self.toolbar.setMovable(False)
        
        # 启动/停止按钮
        self.start_btn = QPushButton("启动系统")
        self.start_btn.clicked.connect(self.toggle_system)
        self.toolbar.addWidget(self.start_btn)
        
        # 添加分隔线
        self.toolbar.addSeparator()
        
        # 全屏按钮
        self.fullscreen_btn = QPushButton("全屏显示")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.toolbar.addWidget(self.fullscreen_btn)
        
        # 截屏按钮（新增）
        self.screenshot_btn = QPushButton("截屏")
        self.screenshot_btn.clicked.connect(self.capture_screen)
        self.screenshot_btn.setToolTip("截取程序全屏界面")
        self.toolbar.addWidget(self.screenshot_btn)
        
        # 添加分隔线
        self.toolbar.addSeparator()
        
        # 标签页
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 创建配置页
        self.config_tab = self.create_config_tab()
        self.tab_widget.addTab(self.config_tab, "系统配置")
        
        # 创建视频页
        self.video_tab = self.create_video_tab()
        self.tab_widget.addTab(self.video_tab, "视频显示")
        
        # 创建长图识别页
        self.long_tab = self.create_long_tab()
        self.tab_widget.addTab(self.long_tab, "长图识别")
        
        # 创建状态页
        self.status_tab = self.create_status_tab()
        self.tab_widget.addTab(self.status_tab, "系统状态")
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_btn.setText("全屏显示")
        else:
            self.showFullScreen()
            self.fullscreen_btn.setText("退出全屏")
    
    def capture_screen(self):
        """截取程序全屏界面"""
        try:
            # 使用QScreen.grabWindow截取当前窗口
            screen = QApplication.primaryScreen()
            pixmap = screen.grabWindow(self.winId())
            
            # 确保保存目录存在
            save_dir = self.base_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")
            
            # 保存截图
            if pixmap.save(filename, "PNG"):
                self.log_event(f"截屏已保存: {filename}")
                self.status_label.setText(f"截屏已保存: {os.path.basename(filename)}")
                
                # 显示成功消息
                QMessageBox.information(self, "截屏成功", 
                                      f"截屏已保存到:\n{filename}\n\n保存目录: {save_dir}")
            else:
                self.log_event("截屏保存失败")
                QMessageBox.warning(self, "截屏失败", "截屏保存失败")
                
        except Exception as e:
            self.log_event(f"截屏失败: {e}")
            QMessageBox.critical(self, "截屏错误", f"截屏时发生错误: {e}")
    
    def create_config_tab(self):
        """创建配置页面"""
        config_tab = QWidget()
        layout = QVBoxLayout(config_tab)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # ===== 模型配置组 =====
        model_group = QGroupBox("模型配置")
        model_layout = QGridLayout()
        
        # 模型选择
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.model_combo = QComboBox()
        for model in MODEL_CANDIDATES:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(os.path.basename(DEFAULT_MODEL_PATH))
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # 模型手动输入
        self.model_edit = QLineEdit(DEFAULT_MODEL_PATH)
        self.model_edit.setPlaceholderText("或输入模型文件路径...")
        model_layout.addWidget(self.model_edit, 0, 2)
        
        # 浏览按钮
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn, 0, 3)
        
        # YOLO参数
        model_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.15)
        self.conf_spin.setSingleStep(0.05)
        model_layout.addWidget(self.conf_spin, 1, 1)
        
        model_layout.addWidget(QLabel("IOU阈值:"), 1, 2)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.10)
        self.iou_spin.setSingleStep(0.05)
        model_layout.addWidget(self.iou_spin, 1, 3)
        
        # 最大检测数量
        model_layout.addWidget(QLabel("最大检测数:"), 2, 0)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 50000)
        self.max_det_spin.setValue(MAX_DETECTIONS)
        self.max_det_spin.setSingleStep(100)
        model_layout.addWidget(self.max_det_spin, 2, 1)
        
        # YOLO26 NMS开关
        model_layout.addWidget(QLabel("启用NMS(YOLO26):"), 2, 2)
        self.nms_check = QCheckBox()
        self.nms_check.setChecked(USE_NMS_FOR_YOLO26)
        model_layout.addWidget(self.nms_check, 2, 3)
        
        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)
        
        # ===== 功能开关组 =====
        func_group = QGroupBox("功能开关")
        func_layout = QGridLayout()
        
        # 检测开关
        self.detection_check = QCheckBox("启用检测")
        self.detection_check.setChecked(True)
        func_layout.addWidget(self.detection_check, 0, 0)
        
        # 追踪开关
        self.tracking_check = QCheckBox("启用追踪")
        self.tracking_check.setChecked(True)
        func_layout.addWidget(self.tracking_check, 0, 1)
        
        # 计数开关
        self.counting_check = QCheckBox("启用过线计数")
        self.counting_check.setChecked(True)
        func_layout.addWidget(self.counting_check, 0, 2)
        
        # 密度检测开关
        self.density_check = QCheckBox("启用密度检测")
        self.density_check.setChecked(DENSITY_ENABLED_DEFAULT)
        func_layout.addWidget(self.density_check, 0, 3)
        
        # UART接收开关
        self.uart_rx_check = QCheckBox("启用UART接收")
        self.uart_rx_check.setChecked(True)
        func_layout.addWidget(self.uart_rx_check, 1, 0)
        
        # UART发送开关
        self.uart_tx_check = QCheckBox("启用UART发送")
        self.uart_tx_check.setChecked(True)
        func_layout.addWidget(self.uart_tx_check, 1, 1)
        
        func_group.setLayout(func_layout)
        scroll_layout.addWidget(func_group)
        
        # ===== 计数配置组 =====
        count_group = QGroupBox("计数配置")
        count_layout = QGridLayout()
        
        # 计数线位置百分比
        count_layout.addWidget(QLabel("计数线位置(%):"), 0, 0)
        self.count_line_percent_spin = QDoubleSpinBox()
        self.count_line_percent_spin.setRange(0.1, 100.0)
        self.count_line_percent_spin.setValue(self.count_line_percent)
        self.count_line_percent_spin.setSingleStep(0.5)
        count_layout.addWidget(self.count_line_percent_spin, 0, 1)
        
        # 计数方向
        count_layout.addWidget(QLabel("计数方向:"), 0, 2)
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["向上", "向下"])
        self.direction_combo.setCurrentText("向下")
        count_layout.addWidget(self.direction_combo, 0, 3)
        
        # ROI向上扩展像素数
        count_layout.addWidget(QLabel("ROI向上扩展(px):"), 1, 0)
        self.roi_top_spin = QSpinBox()
        self.roi_top_spin.setRange(1, 1000)
        self.roi_top_spin.setValue(self.count_line_top_extend)
        count_layout.addWidget(self.roi_top_spin, 1, 1)
        
        # ROI向下扩展像素数
        count_layout.addWidget(QLabel("ROI向下扩展(px):"), 1, 2)
        self.roi_bottom_spin = QSpinBox()
        self.roi_bottom_spin.setRange(1, 1000)
        self.roi_bottom_spin.setValue(self.count_line_bottom_extend)
        count_layout.addWidget(self.roi_bottom_spin, 1, 3)
        
        count_group.setLayout(count_layout)
        scroll_layout.addWidget(count_group)
        
        # ===== 下采样配置组 =====
        downsample_group = QGroupBox("画面下采样配置")
        downsample_layout = QGridLayout()
        
        # 下采样比例选择
        downsample_layout.addWidget(QLabel("下采样比例:"), 0, 0)
        self.downsample_combo = QComboBox()
        for ratio in DOWNSAMPLE_OPTIONS:
            self.downsample_combo.addItem(f"{ratio:.2f}")
        self.downsample_combo.setCurrentText(f"{self.downsample_ratio:.2f}")
        downsample_layout.addWidget(self.downsample_combo, 0, 1)
        
        # 下采样说明
        downsample_layout.addWidget(QLabel("说明: 1.0=不下采样, 0.5=缩小一半, 0.25=缩小到1/4"), 0, 2, 1, 2)
        
        downsample_group.setLayout(downsample_layout)
        scroll_layout.addWidget(downsample_group)
        
        # ===== 性能优化配置组 =====
        perf_group = QGroupBox("性能优化配置")
        perf_layout = QGridLayout()
        
        # 帧跳过配置
        perf_layout.addWidget(QLabel("帧跳过数:"), 0, 0)
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(0, 10)
        self.frame_skip_spin.setValue(0)
        self.frame_skip_spin.setToolTip("0=处理所有帧, 1=跳过1帧处理1帧, 2=跳过2帧处理1帧")
        perf_layout.addWidget(self.frame_skip_spin, 0, 1)
        
        # 性能分析开关
        self.enable_profiling_check = QCheckBox("启用性能分析")
        self.enable_profiling_check.setChecked(False)
        self.enable_profiling_check.setToolTip("启用后会在控制台打印详细的性能数据")
        perf_layout.addWidget(self.enable_profiling_check, 0, 2, 1, 2)
        
        # 说明
        perf_layout.addWidget(QLabel("说明: 帧跳过可以在极端资源受限时使用，以牺牲检测连续性换取FPS"), 1, 0, 1, 4)
        
        perf_group.setLayout(perf_layout)
        scroll_layout.addWidget(perf_group)
        
        # ===== 千粒重配置组 =====
        tkw_group = QGroupBox("千粒重配置")
        tkw_layout = QGridLayout()
        
        # 千粒重配置
        tkw_layout.addWidget(QLabel("千粒重(g):"), 0, 0)
        self.tkw_spin = QDoubleSpinBox()
        self.tkw_spin.setRange(0.1, 1000.0)
        self.tkw_spin.setValue(SEED_TKW_GRAMS)
        self.tkw_spin.setSingleStep(1.0)
        tkw_layout.addWidget(self.tkw_spin, 0, 1)
        
        tkw_group.setLayout(tkw_layout)
        scroll_layout.addWidget(tkw_group)
        
        # ===== 密度检测配置组 =====
        density_group = QGroupBox("密度检测配置")
        density_layout = QGridLayout()
        
        # 网格列数
        density_layout.addWidget(QLabel("网格列数:"), 0, 0)
        self.grid_cols_spin = QSpinBox()
        self.grid_cols_spin.setRange(1, 50)
        self.grid_cols_spin.setValue(DENSITY_GRID_COLS)
        density_layout.addWidget(self.grid_cols_spin, 0, 1)
        
        # 网格行数
        density_layout.addWidget(QLabel("网格行数:"), 0, 2)
        self.grid_rows_spin = QSpinBox()
        self.grid_rows_spin.setRange(1, 50)
        self.grid_rows_spin.setValue(DENSITY_GRID_ROWS)
        density_layout.addWidget(self.grid_rows_spin, 0, 3)
        
        # 低密度阈值
        density_layout.addWidget(QLabel("低密度比例:"), 1, 0)
        self.low_ratio_spin = QDoubleSpinBox()
        self.low_ratio_spin.setRange(0.01, 1.0)
        self.low_ratio_spin.setValue(LOW_DENSITY_RATIO)
        self.low_ratio_spin.setSingleStep(0.05)
        density_layout.addWidget(self.low_ratio_spin, 1, 1)
        
        # 最小检测数
        density_layout.addWidget(QLabel("最小检测数:"), 1, 2)
        self.min_dets_spin = QSpinBox()
        self.min_dets_spin.setRange(1, 100)
        self.min_dets_spin.setValue(MIN_DETS_FOR_DENSITY)
        density_layout.addWidget(self.min_dets_spin, 1, 3)
        
        density_group.setLayout(density_layout)
        scroll_layout.addWidget(density_group)
        
        # ===== UART配置组 =====
        uart_group = QGroupBox("串口配置")
        uart_layout = QGridLayout()
        
        # 串口端口
        uart_layout.addWidget(QLabel("串口端口:"), 0, 0)
        self.uart_port_edit = QLineEdit(UART_PORT)
        uart_layout.addWidget(self.uart_port_edit, 0, 1)
        
        # 波特率
        uart_layout.addWidget(QLabel("波特率:"), 0, 2)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "921600"])
        self.baud_combo.setCurrentText(str(UART_BAUD))
        uart_layout.addWidget(self.baud_combo, 0, 3)
        
        # 门控最大年龄
        uart_layout.addWidget(QLabel("门控超时(秒):"), 1, 0)
        self.gate_age_spin = QDoubleSpinBox()
        self.gate_age_spin.setRange(0.1, 10.0)
        self.gate_age_spin.setValue(GATE_MAX_AGE)
        self.gate_age_spin.setSingleStep(0.1)
        uart_layout.addWidget(self.gate_age_spin, 1, 1)
        
        uart_group.setLayout(uart_layout)
        scroll_layout.addWidget(uart_group)

        # ===== 保存与快捷键配置组 =====
        save_group = QGroupBox("长图与快捷键配置")
        save_layout = QGridLayout()
        save_layout.addWidget(QLabel("保存根目录:"), 0, 0)
        self.save_dir_edit = QLineEdit(self.base_save_dir)
        save_layout.addWidget(self.save_dir_edit, 0, 1, 1, 3)
        save_layout.addWidget(QLabel("开始拼接快捷键:"), 1, 0)
        self.hotkey_start_edit = QLineEdit(self.hotkey_start)
        self.hotkey_start_edit.setMaxLength(1)
        save_layout.addWidget(self.hotkey_start_edit, 1, 1)
        save_layout.addWidget(QLabel("结束拼接快捷键:"), 1, 2)
        self.hotkey_stop_edit = QLineEdit(self.hotkey_stop)
        self.hotkey_stop_edit.setMaxLength(1)
        save_layout.addWidget(self.hotkey_stop_edit, 1, 3)
        save_group.setLayout(save_layout)
        scroll_layout.addWidget(save_group)
        
        # ===== 应用按钮 =====
        button_layout = QHBoxLayout()
        apply_btn = QPushButton("应用配置")
        apply_btn.clicked.connect(self.apply_config)
        apply_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(apply_btn)
        
        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self.reset_config)
        reset_btn.setStyleSheet("background-color: #f44336; color: white;")
        button_layout.addWidget(reset_btn)
        
        scroll_layout.addLayout(button_layout)
        
        # 设置滚动区域内容
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return config_tab
    
    def create_video_tab(self):
        """创建视频显示页面"""
        video_tab = QWidget()
        layout = QVBoxLayout(video_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建滚动区域用于视频显示
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 视频显示区域
        self.video_container = QWidget()
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(5)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 计算适合屏幕的初始最小尺寸
        min_width = min(640, self.screen_width - 100)
        min_height = min(360, self.screen_height - 200)
        self.video_label.setMinimumSize(min_width, min_height)
        
        self.video_label.setScaledContents(False)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 1px solid #333;
            }
        """)
        
        # 添加尺寸标签
        self.size_label = QLabel("原始尺寸: 1280x720")
        self.size_label.setAlignment(Qt.AlignCenter)
        self.size_label.setStyleSheet("color: #888; font-size: 9pt; padding: 2px;")
        
        self.video_layout.addWidget(self.video_label, 0, Qt.AlignCenter)
        self.video_layout.addWidget(self.size_label, 0, Qt.AlignCenter)
        self.video_layout.addStretch()
        
        # 设置滚动区域的内容
        scroll_area.setWidget(self.video_container)
        layout.addWidget(scroll_area)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        # 显示模式切换按钮
        self.display_mode_btn = QPushButton("适应窗口")
        self.display_mode_btn.setFixedWidth(100)
        self.display_mode_btn.clicked.connect(self.toggle_display_mode)
        self.display_mode_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        control_layout.addWidget(self.display_mode_btn)
        
        # 原始大小按钮
        self.original_size_btn = QPushButton("原始大小")
        self.original_size_btn.setFixedWidth(100)
        self.original_size_btn.clicked.connect(self.show_original_size)
        self.original_size_btn.setStyleSheet("background-color: #2196F3; color: white;")
        control_layout.addWidget(self.original_size_btn)
        
        # 添加分隔符
        control_layout.addSpacing(20)
        
        # 其他功能按钮
        self.record_btn = QPushButton("开始录制")
        self.record_btn.setFixedWidth(100)
        self.record_btn.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_btn)
        
        self.snapshot_btn = QPushButton("截图")
        self.snapshot_btn.setFixedWidth(80)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        control_layout.addWidget(self.snapshot_btn)
        
        self.clear_count_btn = QPushButton("清零计数")
        self.clear_count_btn.setFixedWidth(100)
        self.clear_count_btn.clicked.connect(self.clear_counting)
        control_layout.addWidget(self.clear_count_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        return video_tab

    def create_long_tab(self):
        """创建长图识别页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info = QLabel(
            "UART 开启：速度>0 自动拼接，速度≤0 自动停止并识别。\n"
            "UART 关闭：使用快捷键或下方按钮开始/结束拼接，结束后自动识别。\n"
            "结果（原图+识别图+json）保存到根目录下按年月日时分秒创建的子目录。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        btn_layout = QHBoxLayout()
        self.long_start_btn = QPushButton("手动开始拼接")
        self.long_stop_btn = QPushButton("手动结束并识别")
        self.long_start_btn.clicked.connect(self.manual_start_long)
        self.long_stop_btn.clicked.connect(self.manual_stop_long)
        btn_layout.addWidget(self.long_start_btn)
        btn_layout.addWidget(self.long_stop_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("长图状态:"))
        self.long_status_label = QLabel("空闲")
        self.long_status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.long_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        return tab
    
    def create_status_tab(self):
        """创建系统状态页面"""
        status_tab = QWidget()
        layout = QVBoxLayout(status_tab)
        
        # 系统状态组
        sys_group = QGroupBox("系统状态")
        sys_layout = QGridLayout()
        
        # 相机状态
        sys_layout.addWidget(QLabel("相机状态:"), 0, 0)
        self.camera_status = QLabel("未连接")
        self.camera_status.setStyleSheet("color: red; font-weight: bold;")
        sys_layout.addWidget(self.camera_status, 0, 1)
        
        # 模型状态
        sys_layout.addWidget(QLabel("模型状态:"), 0, 2)
        self.model_status = QLabel("未加载")
        self.model_status.setStyleSheet("color: red; font-weight: bold;")
        sys_layout.addWidget(self.model_status, 0, 3)
        
        # UART接收状态
        sys_layout.addWidget(QLabel("UART接收:"), 1, 0)
        self.uart_rx_status = QLabel("未连接")
        self.uart_rx_status.setStyleSheet("color: red; font-weight: bold;")
        sys_layout.addWidget(self.uart_rx_status, 1, 1)
        
        # UART发送状态
        sys_layout.addWidget(QLabel("UART发送:"), 1, 2)
        self.uart_tx_status = QLabel("未连接")
        self.uart_tx_status.setStyleSheet("color: red; font-weight: bold;")
        sys_layout.addWidget(self.uart_tx_status, 1, 3)
        
        # 系统运行时间
        sys_layout.addWidget(QLabel("运行时间:"), 2, 0)
        self.uptime_label = QLabel("00:00:00")
        sys_layout.addWidget(self.uptime_label, 2, 1)
        
        # 帧率显示
        sys_layout.addWidget(QLabel("处理帧率:"), 2, 2)
        self.fps_label = QLabel("0")
        sys_layout.addWidget(self.fps_label, 2, 3)
        
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # 检测统计组
        stats_group = QGroupBox("检测统计")
        stats_layout = QGridLayout()
        
        # 检测数量
        stats_layout.addWidget(QLabel("当前检测数:"), 0, 0)
        self.detections_label = QLabel("0")
        stats_layout.addWidget(self.detections_label, 0, 1)
        
        # 追踪数量
        stats_layout.addWidget(QLabel("追踪目标数:"), 0, 2)
        self.tracked_label = QLabel("0")
        stats_layout.addWidget(self.tracked_label, 0, 3)
        
        # 计数数量
        stats_layout.addWidget(QLabel("累计计数:"), 1, 0)
        self.counted_label = QLabel("0")
        stats_layout.addWidget(self.counted_label, 1, 1)
        
        # 低密度状态
        stats_layout.addWidget(QLabel("低密度状态:"), 1, 2)
        self.low_density_label = QLabel("正常")
        self.low_density_label.setStyleSheet("color: green; font-weight: bold;")
        stats_layout.addWidget(self.low_density_label, 1, 3)
        
        # 门控状态
        stats_layout.addWidget(QLabel("门控状态:"), 2, 0)
        self.gate_label = QLabel("关闭")
        self.gate_label.setStyleSheet("color: red; font-weight: bold;")
        stats_layout.addWidget(self.gate_label, 2, 1)
        
        # 速度显示
        stats_layout.addWidget(QLabel("当前速度:"), 2, 2)
        self.speed_label = QLabel("0 mm/s")
        stats_layout.addWidget(self.speed_label, 2, 3)
        
        # 位置显示
        stats_layout.addWidget(QLabel("当前位置:"), 3, 0)
        self.position_label = QLabel("0 mm")
        stats_layout.addWidget(self.position_label, 3, 1)

        # 长图状态
        stats_layout.addWidget(QLabel("长图状态:"), 3, 2)
        self.long_status_small = QLabel("空闲")
        self.long_status_small.setStyleSheet("color: green; font-weight: bold;")
        stats_layout.addWidget(self.long_status_small, 3, 3)
        
        # 计数线状态
        stats_layout.addWidget(QLabel("计数线位置:"), 4, 0)
        self.count_line_label = QLabel("0 px (0.0%)")
        self.count_line_label.setStyleSheet("color: blue; font-weight: bold;")
        stats_layout.addWidget(self.count_line_label, 4, 1)
        
        # ROI状态
        stats_layout.addWidget(QLabel("ROI区域:"), 4, 2)
        self.roi_label = QLabel("全屏")
        self.roi_label.setStyleSheet("color: blue; font-weight: bold;")
        stats_layout.addWidget(self.roi_label, 4, 3)
        
        # 下采样状态
        stats_layout.addWidget(QLabel("下采样:"), 5, 0)
        self.downsample_label = QLabel("1.0x")
        self.downsample_label.setStyleSheet("color: blue; font-weight: bold;")
        stats_layout.addWidget(self.downsample_label, 5, 1)
        
        # 计数方向
        stats_layout.addWidget(QLabel("计数方向:"), 5, 2)
        self.counting_direction_label = QLabel("向下")
        self.counting_direction_label.setStyleSheet("color: blue; font-weight: bold;")
        stats_layout.addWidget(self.counting_direction_label, 5, 3)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 事件日志组
        log_group = QGroupBox("事件日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        # 日志控制
        log_control_layout = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_control_layout.addWidget(clear_log_btn)
        
        log_control_layout.addStretch()
        log_layout.addLayout(log_control_layout)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return status_tab
    
    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.engine *.onnx *.pt);;所有文件 (*.*)"
        )
        if file_path:
            self.model_edit.setText(file_path)
    
    def apply_config(self):
        """应用配置"""
        # 获取模型路径
        model_path = self.model_edit.text().strip()
        if not model_path:
            model_path = self.model_combo.currentText()
        
        # 检查模型文件是否存在
        if not os.path.isfile(model_path):
            QMessageBox.warning(self, "警告", f"模型文件不存在: {model_path}")
            return
        
        # 保存当前模型路径
        self.current_model_path = model_path
        
        # 更新全局配置
        global USE_NMS_FOR_YOLO26
        global MAX_DETECTIONS
        USE_NMS_FOR_YOLO26 = self.nms_check.isChecked()
        MAX_DETECTIONS = self.max_det_spin.value()
        
        # 更新计数线参数
        self.count_line_percent = self.count_line_percent_spin.value()
        self.count_line_top_extend = self.roi_top_spin.value()
        self.count_line_bottom_extend = self.roi_bottom_spin.value()
        
        # 更新下采样参数
        self.downsample_ratio = float(self.downsample_combo.currentText())
        
        # 更新性能优化参数
        self.frame_skip = self.frame_skip_spin.value()
        self.enable_profiling = self.enable_profiling_check.isChecked()
        
        # 保存路径与快捷键
        self.base_save_dir = self.save_dir_edit.text().strip() or DEFAULT_SAVE_DIR
        self.hotkey_start = (self.hotkey_start_edit.text().strip() or DEFAULT_HOTKEY_START).upper()[0:1]
        self.hotkey_stop = (self.hotkey_stop_edit.text().strip() or DEFAULT_HOTKEY_STOP).upper()[0:1]

        # 记录配置变更
        self.log_event(f"应用配置: 模型={os.path.basename(model_path)}")
        self.log_event(f"  置信度={self.conf_spin.value()}, IOU={self.iou_spin.value()}")
        self.log_event(f"  最大检测数={MAX_DETECTIONS}")
        self.log_event(f"  计数线位置={self.count_line_percent}%")
        self.log_event(f"  ROI上扩={self.count_line_top_extend}px, ROI下扩={self.count_line_bottom_extend}px")
        self.log_event(f"  计数方向={self.direction_combo.currentText()}")
        self.log_event(f"  下采样比例={self.downsample_ratio:.2f}x")
        self.log_event(f"  帧跳过数={self.frame_skip}")
        self.log_event(f"  性能分析={'启用' if self.enable_profiling else '禁用'}")
        self.log_event(f"  YOLO26 NMS={'启用' if USE_NMS_FOR_YOLO26 else '禁用'}")
        self.log_event(f"  保存目录={self.base_save_dir}")
        self.log_event(f"  快捷键 开始={self.hotkey_start}, 结束={self.hotkey_stop}")
        
        # 如果系统正在运行，需要重启
        if self.camera_thread and self.camera_thread.running:
            reply = QMessageBox.question(self, "重启系统", 
                                        "配置已更改，需要重启系统才能生效。是否现在重启?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_system()
                self.start_system()
    
    def reset_config(self):
        """恢复默认配置"""
        reply = QMessageBox.question(self, "确认", 
                                    "是否恢复所有配置为默认值?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 重置所有控件
            self.model_combo.setCurrentText(os.path.basename(DEFAULT_MODEL_PATH))
            self.model_edit.setText(DEFAULT_MODEL_PATH)
            self.conf_spin.setValue(0.15)
            self.iou_spin.setValue(0.10)
            self.max_det_spin.setValue(MAX_DETECTIONS)
            self.nms_check.setChecked(USE_NMS_FOR_YOLO26)
            self.detection_check.setChecked(True)
            self.tracking_check.setChecked(True)
            self.counting_check.setChecked(True)
            self.density_check.setChecked(DENSITY_ENABLED_DEFAULT)
            self.uart_rx_check.setChecked(True)
            self.uart_tx_check.setChecked(True)
            
            # 计数线参数
            self.count_line_percent_spin.setValue(COUNT_LINE_PERCENT_DEFAULT)
            self.direction_combo.setCurrentText("向下")
            self.roi_top_spin.setValue(COUNT_LINE_TOP_EXTEND_DEFAULT)
            self.roi_bottom_spin.setValue(COUNT_LINE_BOTTOM_EXTEND_DEFAULT)
            
            # 下采样参数
            self.downsample_combo.setCurrentText(f"{DOWNSAMPLE_RATIO_DEFAULT:.2f}")
            
            self.tkw_spin.setValue(SEED_TKW_GRAMS)
            self.grid_cols_spin.setValue(DENSITY_GRID_COLS)
            self.grid_rows_spin.setValue(DENSITY_GRID_ROWS)
            self.low_ratio_spin.setValue(LOW_DENSITY_RATIO)
            self.min_dets_spin.setValue(MIN_DETS_FOR_DENSITY)
            self.uart_port_edit.setText(UART_PORT)
            self.baud_combo.setCurrentText(str(UART_BAUD))
            self.gate_age_spin.setValue(GATE_MAX_AGE)
            self.save_dir_edit.setText(DEFAULT_SAVE_DIR)
            self.hotkey_start_edit.setText(DEFAULT_HOTKEY_START)
            self.hotkey_stop_edit.setText(DEFAULT_HOTKEY_STOP)
            
            self.log_event("配置已恢复为默认值")
    
    def toggle_system(self):
        """启动/停止系统"""
        if self.camera_thread and self.camera_thread.running:
            self.stop_system()
        else:
            self.start_system()
    
    def start_system(self):
        """启动系统"""
        # 获取模型路径
        model_path = self.model_edit.text().strip()
        if not model_path:
            model_path = self.model_combo.currentText()
        
        # 检查模型文件
        if not os.path.isfile(model_path):
            QMessageBox.critical(self, "错误", f"模型文件不存在: {model_path}")
            return
        
        # 创建并启动相机线程
        self.camera_thread = CameraThread()
        
        # 设置配置参数
        self.camera_thread.enable_detection = self.detection_check.isChecked()
        self.camera_thread.enable_tracking = self.tracking_check.isChecked()
        self.camera_thread.enable_counting = self.counting_check.isChecked()
        self.camera_thread.enable_density_detection = self.density_check.isChecked()
        self.camera_thread.enable_uart_rx = self.uart_rx_check.isChecked()
        self.camera_thread.enable_uart_tx = self.uart_tx_check.isChecked()
        
        # 计数线参数
        self.camera_thread.count_line_percent = self.count_line_percent_spin.value()
        self.camera_thread.count_line_top_extend = self.roi_top_spin.value()
        self.camera_thread.count_line_bottom_extend = self.roi_bottom_spin.value()
        
        # 下采样参数
        self.camera_thread.downsample_ratio = float(self.downsample_combo.currentText())
        
        # 性能优化参数
        self.camera_thread.frame_skip = self.frame_skip_spin.value()
        self.camera_thread.enable_profiling = self.enable_profiling_check.isChecked()
        
        # 其他参数
        self.camera_thread.counting_direction = "up" if self.direction_combo.currentText() == "向上" else "down"
        self.camera_thread.max_detections = self.max_det_spin.value()
        self.camera_thread.set_base_save_dir(self.base_save_dir)
        
        # 连接信号
        self.camera_thread.frame_processed.connect(self.update_video_display)
        self.camera_thread.long_status_changed.connect(self.on_long_status_changed)
        
        # 初始化系统
        if not self.camera_thread.initialize(model_path):
            QMessageBox.critical(self, "错误", "系统初始化失败")
            return
        
        # 启动线程
        self.camera_thread.start()
        
        # 更新UI状态
        self.start_btn.setText("停止系统")
        self.start_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        
        # 记录启动时间
        self.start_time = time.time()
        
        # 更新状态
        self.camera_status.setText("运行中")
        self.camera_status.setStyleSheet("color: green; font-weight: bold;")
        self.model_status.setText(f"已加载: {os.path.basename(model_path)}")
        self.model_status.setStyleSheet("color: green; font-weight: bold;")
        self.count_line_label.setText(f"0 px (0.0%)")
        self.roi_label.setText("全屏")
        self.downsample_label.setText(f"{self.downsample_ratio:.2f}x")
        self.counting_direction_label.setText(self.direction_combo.currentText())
        
        self.log_event(f"系统启动 - 模型: {os.path.basename(model_path)}")
        self.log_event(f"  计数线位置: {self.count_line_percent}%")
        self.log_event(f"  ROI扩展: 上{self.count_line_top_extend}px, 下{self.count_line_bottom_extend}px")
        self.log_event(f"  下采样: {self.downsample_ratio:.2f}x")
        self.log_event(f"  计数方向: {self.direction_combo.currentText()}")
        self.log_event(f"  计数模式: {'启用' if self.counting_check.isChecked() else '禁用'}")
    
    def stop_system(self):
        """停止系统"""
        if self.camera_thread:
            try:
                self.camera_thread.stop()
                self.camera_thread = None
            except Exception as e:
                print(f"停止相机线程时出错: {e}")
        
        # 更新UI状态
        self.start_btn.setText("启动系统")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        # 清空视频显示
        self.video_label.clear()
        self.video_label.setText("系统已停止")
        
        # 更新状态
        self.camera_status.setText("未连接")
        self.camera_status.setStyleSheet("color: red; font-weight: bold;")
        self.model_status.setText("未加载")
        self.model_status.setStyleSheet("color: red; font-weight: bold;")
        self.uart_rx_status.setText("未连接")
        self.uart_tx_status.setText("未连接")
        self.long_status_label.setText("空闲")
        self.long_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.long_status_small.setText("空闲")
        self.long_status_small.setStyleSheet("color: green; font-weight: bold;")
        self.count_line_label.setText("0 px (0.0%)")
        self.roi_label.setText("全屏")
        self.downsample_label.setText("1.0x")
        self.counting_direction_label.setText("向下")
        
        self.log_event("系统已停止")
    
    def update_video_display(self, frame, stats):
        """更新视频显示，确保不会超出屏幕"""
        try:
            # 保存原始帧尺寸
            self.video_height, self.video_width = frame.shape[:2]
            
            # 转换OpenCV帧为Qt图像
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # 创建Pixmap并保存
            self.current_pixmap = QPixmap.fromImage(q_img)
            
            if self.fit_to_window:
                # 获取当前可用空间（考虑滚动区域和边距）
                scroll_widget = self.video_container.parentWidget()
                if scroll_widget:
                    # 获取滚动区域的可用大小
                    scroll_area = scroll_widget.parentWidget()
                    if scroll_area:
                        available_width = scroll_area.width() - 30  # 减去滚动条宽度和边距
                        available_height = scroll_area.height() - 50  # 减去状态标签和边距
                    else:
                        available_width = self.video_label.width()
                        available_height = self.video_label.height()
                else:
                    available_width = self.video_label.width()
                    available_height = self.video_label.height()
                
                # 确保可用尺寸有效
                available_width = max(100, available_width)
                available_height = max(100, available_height)
                
                # 关键改进：独立检查宽度和高度是否超出，而不是统一缩放
                
                # 初始值
                new_width = width
                new_height = height
                
                # 检查宽度是否超出可用宽度
                if width > available_width:
                    # 宽度超出，按宽度缩放
                    scale_w = available_width / width
                    new_width = available_width
                    new_height = int(height * scale_w)
                
                # 检查缩放后的高度是否超出可用高度
                if new_height > available_height:
                    # 高度超出，按高度缩放
                    scale_h = available_height / new_height
                    new_height = available_height
                    new_width = int(new_width * scale_h)
                
                # 应用最终的缩放
                if new_width != width or new_height != height:
                    # 需要缩放
                    scaled_pixmap = self.current_pixmap.scaled(
                        new_width, new_height, 
                        Qt.KeepAspectRatio,  # 保持宽高比
                        Qt.SmoothTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
                    
                    # 计算实际的缩放比例
                    actual_scale_w = new_width / width
                    actual_scale_h = new_height / height
                    actual_scale = min(actual_scale_w, actual_scale_h)
                    
                    self.size_label.setText(f"显示尺寸: {new_width}x{new_height} (缩放: {actual_scale:.2f}x)")
                else:
                    # 不需要缩放，显示原始大小
                    self.video_label.setPixmap(self.current_pixmap)
                    self.size_label.setText(f"原始尺寸: {width}x{height}")
                
            else:
                # 显示模式：原始大小
                self.video_label.setPixmap(self.current_pixmap)
                self.video_label.adjustSize()
                self.size_label.setText(f"原始尺寸: {width}x{height} (原始大小)")
            
            # 更新统计信息
            self.stats = stats
            
        except Exception as e:
            print(f"更新视频显示时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def toggle_display_mode(self):
        """切换显示模式"""
        self.fit_to_window = not self.fit_to_window
        if self.fit_to_window:
            self.display_mode_btn.setText("适应窗口")
            self.display_mode_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.display_mode_btn.setText("原始大小")
            self.display_mode_btn.setStyleSheet("background-color: #2196F3; color: white;")
        
        # 立即重新显示当前帧
        if hasattr(self, 'current_pixmap') and self.current_pixmap:
            self.update_video_display_from_cache()
    
    def show_original_size(self):
        """显示原始大小"""
        self.fit_to_window = False
        self.display_mode_btn.setText("原始大小")
        self.display_mode_btn.setStyleSheet("background-color: #2196F3; color: white;")
        
        if hasattr(self, 'current_pixmap') and self.current_pixmap:
            self.video_label.setPixmap(self.current_pixmap)
            self.video_label.adjustSize()
            pixmap_size = self.current_pixmap.size()
            self.size_label.setText(f"原始尺寸: {pixmap_size.width()}x{pixmap_size.height()}")
    
    def update_video_display_from_cache(self):
        """从缓存的Pixmap更新显示"""
        if hasattr(self, 'current_pixmap') and self.current_pixmap:
            if self.fit_to_window:
                # 重新计算适应窗口的显示
                pixmap_size = self.current_pixmap.size()
                width = pixmap_size.width()
                height = pixmap_size.height()
                
                # 获取当前可用空间
                scroll_widget = self.video_container.parentWidget()
                if scroll_widget:
                    scroll_area = scroll_widget.parentWidget()
                    if scroll_area:
                        available_width = scroll_area.width() - 30
                        available_height = scroll_area.height() - 50
                    else:
                        available_width = self.video_label.width()
                        available_height = self.video_label.height()
                else:
                    available_width = self.video_label.width()
                    available_height = self.video_label.height()
                
                # 确保有效尺寸
                available_width = max(100, available_width)
                available_height = max(100, available_height)
                
                # 独立的宽高检查
                new_width = width
                new_height = height
                
                if width > available_width:
                    scale_w = available_width / width
                    new_width = available_width
                    new_height = int(height * scale_w)
                
                if new_height > available_height:
                    scale_h = available_height / new_height
                    new_height = available_height
                    new_width = int(new_width * scale_h)
                
                # 应用缩放
                if new_width != width or new_height != height:
                    scaled_pixmap = self.current_pixmap.scaled(
                        new_width, new_height, 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.video_label.setPixmap(scaled_pixmap)
                    
                    actual_scale_w = new_width / width
                    actual_scale_h = new_height / height
                    actual_scale = min(actual_scale_w, actual_scale_h)
                    
                    self.size_label.setText(f"显示尺寸: {new_width}x{new_height} (缩放: {actual_scale:.2f}x)")
                else:
                    self.video_label.setPixmap(self.current_pixmap)
                    self.size_label.setText(f"原始尺寸: {width}x{height}")
            else:
                # 原始大小模式
                self.video_label.setPixmap(self.current_pixmap)
                self.video_label.adjustSize()
                pixmap_size = self.current_pixmap.size()
                self.size_label.setText(f"原始尺寸: {pixmap_size.width()}x{pixmap_size.height()}")
    
    def resizeEvent(self, event):
        """窗口大小改变时调整视频显示"""
        super().resizeEvent(event)
        
        # 如果当前是适应窗口模式，重新计算显示尺寸
        if self.fit_to_window and hasattr(self, 'current_pixmap') and self.current_pixmap:
            # 使用定时器延迟更新，避免频繁重绘
            QTimer.singleShot(100, self.update_video_display_from_cache)
    
    def update_status(self):
        """更新状态信息"""
        if self.camera_thread and self.camera_thread.running:
            # 更新运行时间
            if hasattr(self, 'start_time'):
                uptime = int(time.time() - self.start_time)
                hours = uptime // 3600
                minutes = (uptime % 3600) // 60
                seconds = uptime % 60
                self.uptime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 更新统计信息
            if hasattr(self, 'stats'):
                self.fps_label.setText(str(self.stats.get('fps', 0)))
                self.detections_label.setText(str(self.stats.get('detections', 0)))
                self.tracked_label.setText(str(self.stats.get('tracked', 0)))
                self.counted_label.setText(str(self.stats.get('counted', 0)))
                
                # 低密度状态
                low_density = self.stats.get('low_density', False)
                self.low_density_label.setText("低密度" if low_density else "正常")
                self.low_density_label.setStyleSheet(
                    "color: orange; font-weight: bold;" if low_density else "color: green; font-weight: bold;"
                )
                
                # 门控状态
                gate_state = self.stats.get('gate_state', False)
                self.gate_label.setText("开启" if gate_state else "关闭")
                self.gate_label.setStyleSheet(
                    "color: green; font-weight: bold;" if gate_state else "color: red; font-weight: bold;"
                )
                
                # 速度和位置
                self.speed_label.setText(f"{self.stats.get('v_mmps', 0)} mm/s")
                self.position_label.setText(f"{self.stats.get('s_mm', 0)} mm")
                
                # UART状态
                self.uart_rx_status.setText(self.stats.get('uart_rx_status', '未连接'))
                self.uart_tx_status.setText(self.stats.get('uart_tx_status', '未连接'))
                
                # 更新状态颜色
                rx_color = "green" if self.stats.get('uart_rx_status') == '运行中' else "red"
                tx_color = "green" if self.stats.get('uart_tx_status') == '运行中' else "red"
                self.uart_rx_status.setStyleSheet(f"color: {rx_color}; font-weight: bold;")
                self.uart_tx_status.setStyleSheet(f"color: {tx_color}; font-weight: bold;")

                # 长图状态
                long_text = self.stats.get('long_status', '空闲')
                self.long_status_small.setText(long_text)
                self.long_status_label.setText(long_text)
                color = "green" if long_text == "空闲" else "orange"
                self.long_status_small.setStyleSheet(f"color: {color}; font-weight: bold;")
                self.long_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
                
                # 计数线和ROI状态
                count_line_y = self.stats.get('count_line_y', 0)
                count_line_percent = 0
                if count_line_y > 0 and self.camera_thread and hasattr(self.camera_thread, 'original_frame_height'):
                    count_line_percent = (count_line_y / self.camera_thread.original_frame_height) * 100
                self.count_line_label.setText(f"{count_line_y} px ({count_line_percent:.1f}%)")
                
                roi_info = self.stats.get('roi_info', '全屏')
                self.roi_label.setText(roi_info)
                
                # 下采样状态
                self.downsample_label.setText(f"{self.downsample_ratio:.2f}x")
                
                # 计数方向
                self.counting_direction_label.setText(self.direction_combo.currentText())
        
        # 更新状态栏
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_label.setText(f"就绪 | {current_time}")
    
    def toggle_recording(self):
        """切换录制状态"""
        if self.record_btn.text() == "开始录制":
            self.record_btn.setText("停止录制")
            self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
            self.log_event("开始录制视频")
        else:
            self.record_btn.setText("开始录制")
            self.record_btn.setStyleSheet("")
            self.log_event("停止录制视频")
    
    def take_snapshot(self):
        """截图"""
        if self.camera_thread and self.camera_thread.running:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.base_save_dir, f"snapshot_{timestamp}.jpg")
            try:
                # 保存当前显示的视频帧
                if hasattr(self, 'current_pixmap') and self.current_pixmap:
                    self.current_pixmap.save(filename, "JPG")
                    self.log_event(f"截图已保存: {filename}")
                else:
                    self.log_event("截图失败：无可用图像")
            except Exception as e:
                self.log_event(f"截图失败: {e}")
    
    def clear_counting(self):
        """清零计数"""
        if self.camera_thread:
            self.camera_thread.counted_objects.clear()
            self.camera_thread.stats['counted'] = 0
            self.log_event("计数已清零")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def log_event(self, message):
        """记录事件到日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        if self.camera_thread and self.camera_thread.running:
            reply = QMessageBox.question(self, "确认", 
                                        "系统正在运行，确定要退出吗?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_system()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ===== 长图手动控制 =====
    def manual_start_long(self):
        if not (self.camera_thread and self.camera_thread.running):
            return
        if self.camera_thread.enable_uart_rx and SERIAL_AVAILABLE:
            QMessageBox.information(self, "提示", "UART 开启时由速度自动控制，不支持手动。")
            return
        self.camera_thread.start_manual_long_capture()
        self.log_event("手动开始长图拼接")

    def manual_stop_long(self):
        if not (self.camera_thread and self.camera_thread.running):
            return
        if self.camera_thread.enable_uart_rx and SERIAL_AVAILABLE:
            QMessageBox.information(self, "提示", "UART 开启时由速度自动控制，不支持手动。")
            return
        self.camera_thread.stop_manual_long_capture()
        self.log_event("手动结束长图拼接并识别")

    def on_long_status_changed(self, text):
        self.long_status_label.setText(text)
        self.long_status_small.setText(text)
        color = "green" if text == "空闲" else "orange"
        self.long_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.long_status_small.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ===== 键盘快捷键 =====
    def keyPressEvent(self, event):
        key_text = event.text().upper()
        if not key_text:
            return super().keyPressEvent(event)
        if not (self.camera_thread and self.camera_thread.running):
            return super().keyPressEvent(event)
        # 仅在 UART 关闭或不可用时允许快捷键
        if self.camera_thread.enable_uart_rx and SERIAL_AVAILABLE:
            return super().keyPressEvent(event)
        if key_text == self.hotkey_start:
            self.manual_start_long()
            return
        if key_text == self.hotkey_stop:
            self.manual_stop_long()
            return
        return super().keyPressEvent(event)


# ===== 主程序 =====
def main():
    print("🎯 seedyolo.py：时间对齐 + 门控识别 + 计数 + 密度区间缓存 + 克重换算 + UI界面")
    print(f"🔧 支持YOLO26模型，NMS开关: {'启用' if USE_NMS_FOR_YOLO26 else '禁用'}")
    print(f"🔧 最大检测数量: {MAX_DETECTIONS}")
    print(f"🔧 默认模型路径: {DEFAULT_MODEL_PATH}")
    print(f"🔧 截图保存路径: {DEFAULT_SAVE_DIR}")
    print(f"🔧 计数线位置: {COUNT_LINE_PERCENT_DEFAULT}%")
    print(f"🔧 ROI扩展: 上{COUNT_LINE_TOP_EXTEND_DEFAULT}px, 下{COUNT_LINE_BOTTOM_EXTEND_DEFAULT}px")
    print(f"🔧 下采样比例: {DOWNSAMPLE_RATIO_DEFAULT}")
    print(f"🔧 计数模式: 启用时只识别ROI区域，禁用时全屏识别")
    print(f"🔧 新增功能: 程序全屏截屏，保存路径可配置")
    
    # 检查PyQt5是否可用
    if not PYQT_AVAILABLE:
        print("❌ PyQt5不可用，无法启动UI界面")
        print("请安装: pip install PyQt5")
        return
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 启动应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
