#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5 GUI for Linker Hand vision interaction.

The camera and MediaPipe pipeline run in a QThread. The main Qt thread owns the
LinkerHand SDK object and decides when it is safe to send poses.
"""

import os
import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
GUI_CONTROL_DIR = os.path.join(PROJECT_ROOT, "example", "gui_control")

for path in (PROJECT_ROOT, GUI_CONTROL_DIR, CURRENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import mediapipe as mp
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import vision_control as vision


class VisionWorker(QThread):
    frame_ready = pyqtSignal(object)
    result_ready = pyqtSignal(bool, object, object, object, float)
    log = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int,
        hand_joint: str,
        open_pose: List[int],
        fist_pose: List[int],
        smoothing: float,
        deadband: int,
        max_step: int,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.hand_joint = hand_joint
        self.open_pose = list(open_pose)
        self.fist_pose = list(fist_pose)
        self.smoothing = smoothing
        self.smoother = PoseSmoother(alpha=smoothing, deadband=deadband, max_step=max_step)
        self._running = False

    def set_smoothing(self, smoothing: float) -> None:
        self.smoothing = smoothing
        self.smoother.set_alpha(smoothing)

    def set_filter_params(
        self,
        alpha: Optional[float] = None,
        deadband: Optional[int] = None,
        max_step: Optional[int] = None,
    ) -> None:
        self.smoother.update_params(alpha=alpha, deadband=deadband, max_step=max_step)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        cap = None
        frame_count = 0
        fps = 0.0
        fps_time = time.time()

        try:
            cap = vision.open_camera(self.camera_index)
            if not cap or not cap.isOpened():
                self.log.emit("摄像头打开失败: index={}".format(self.camera_index))
                return

            self.log.emit("摄像头已打开: index={}".format(self.camera_index))
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            ) as hands:
                while self._running:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.log.emit("摄像头读取帧失败，视觉线程停止")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = hands.process(rgb)
                    rgb.flags.writeable = True

                    detected = bool(result.multi_hand_landmarks)
                    raw_pose = None
                    smooth_pose = None
                    curls = None

                    if detected:
                        hand_landmarks = result.multi_hand_landmarks[0]
                        curls = vision.get_finger_curls(hand_landmarks)
                        raw_pose = vision.pose_from_curls(
                            self.hand_joint,
                            curls,
                            self.open_pose,
                            self.fist_pose,
                            None,
                            0.0,
                        )
                        smooth_pose = self.smoother.smooth(raw_pose)

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                    else:
                        self.smoother.reset()

                    frame_count += 1
                    now = time.time()
                    if now - fps_time >= 1.0:
                        fps = frame_count / (now - fps_time)
                        frame_count = 0
                        fps_time = now

                    cv2.putText(
                        frame,
                        "FPS: {:.1f}".format(fps),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Hand: {}".format("YES" if detected else "NO"),
                        (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if detected else (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    self.frame_ready.emit(frame)
                    self.result_ready.emit(detected, raw_pose, smooth_pose, curls, fps)
                    self.msleep(1)

        except Exception as exc:
            self.log.emit("视觉线程异常: {}".format(exc))
        finally:
            if cap is not None:
                cap.release()
            self.log.emit("摄像头资源已释放")


class PoseSmoother:
    """Smooth Linker Hand pose arrays before sending them to the SDK."""

    def __init__(self, alpha: float = 0.25, deadband: int = 5, max_step: int = 8):
        self.alpha = self._clamp_float(alpha, 0.01, 1.0)
        self.deadband = max(0, int(deadband))
        self.max_step = max(1, int(max_step))
        self._state = None

    @staticmethod
    def _clamp_float(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _clamp_int(value: float, low: int = 0, high: int = 255) -> int:
        return int(max(low, min(high, round(value))))

    def set_alpha(self, alpha: float) -> None:
        self.alpha = self._clamp_float(alpha, 0.01, 1.0)

    def set_deadband(self, deadband: int) -> None:
        self.deadband = max(0, int(deadband))

    def set_max_step(self, max_step: int) -> None:
        self.max_step = max(1, int(max_step))

    def update_params(
        self,
        alpha: Optional[float] = None,
        deadband: Optional[int] = None,
        max_step: Optional[int] = None,
    ) -> None:
        if alpha is not None:
            self.set_alpha(alpha)
        if deadband is not None:
            self.set_deadband(deadband)
        if max_step is not None:
            self.set_max_step(max_step)

    def reset(self, pose: Optional[List[int]] = None) -> None:
        self._state = list(pose) if pose else None

    def smooth(self, raw_pose: Optional[List[int]]) -> Optional[List[int]]:
        if not raw_pose:
            return None

        raw = [self._clamp_int(v) for v in raw_pose]
        if self._state is None or len(self._state) != len(raw):
            self._state = list(raw)
            return list(raw)

        next_state = []
        for previous, target in zip(self._state, raw):
            delta = target - previous

            if abs(delta) <= self.deadband:
                value = previous
            else:
                deadband_target = target
                ema_value = (1.0 - self.alpha) * previous + self.alpha * deadband_target
                step = ema_value - previous
                if step > self.max_step:
                    value = previous + self.max_step
                elif step < -self.max_step:
                    value = previous - self.max_step
                else:
                    value = ema_value

            next_state.append(self._clamp_int(value))

        self._state = next_state
        return list(next_state)


@dataclass
class RehabAction:
    name: str
    target_robot_pose: List[int]
    target_finger_scores: Dict[str, int]
    description: str
    hit_threshold: float = 80.0
    reset_threshold: float = 60.0
    hold_frames: int = 5


FINGER_SCORE_KEYS = ("thumb", "index", "middle", "ring", "little")
FINGER_SCORE_ALIASES = {"pinky": "little"}
REHAB_TEMPLATE_PATH = os.path.join(CURRENT_DIR, "rehab_action_templates.json")


REHAB_ACTION_CONFIGS = [
    {
        "name": "张开手训练",
        "pose": "open",
        "target_finger_scores": {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0},
        "description": "训练五指伸展能力",
        "hit_threshold": 80.0,
        "reset_threshold": 60.0,
        "hold_frames": 5,
    },
    {
        "name": "握拳训练",
        "pose": "fist",
        "target_finger_scores": {"thumb": 43, "index": 85, "middle": 92, "ring": 84, "pinky": 74},
        "description": "训练五指屈曲能力",
        "hit_threshold": 78.0,
        "reset_threshold": 58.0,
        "hold_frames": 5,
    },
    {
        "name": "拇指对食指训练",
        "fingers": ["thumb", "index"],
        "ratio": 0.75,
        "target_finger_scores": {"thumb": 75, "index": 75, "middle": 0, "ring": 0, "pinky": 0},
        "description": "训练拇指与食指精细配合能力",
        "hit_threshold": 75.0,
        "reset_threshold": 55.0,
        "hold_frames": 4,
    },
    {
        "name": "两指捏合训练",
        "fingers": ["thumb", "index"],
        "ratio": 0.9,
        "target_finger_scores": {"thumb": 90, "index": 90, "middle": 0, "ring": 0, "pinky": 0},
        "description": "训练两指捏取能力",
        "hit_threshold": 75.0,
        "reset_threshold": 55.0,
        "hold_frames": 4,
    },
    {
        "name": "食指点击训练",
        "fingers": ["index"],
        "ratio": 1.0,
        "target_finger_scores": {"thumb": 0, "index": 100, "middle": 0, "ring": 0, "pinky": 0},
        "description": "训练单指伸展与点击能力",
        "hit_threshold": 75.0,
        "reset_threshold": 50.0,
        "hold_frames": 3,
    },
    {
        "name": "五指依次弯曲训练",
        "fingers": ["thumb", "index", "middle", "ring", "little"],
        "ratio": 0.65,
        "target_finger_scores": {"thumb": 65, "index": 65, "middle": 65, "ring": 65, "pinky": 65},
        "description": "训练手指独立控制能力",
        "hit_threshold": 70.0,
        "reset_threshold": 50.0,
        "hold_frames": 4,
    },
]


class RehabTrainer:
    WAIT_TARGET = "WAIT_TARGET"
    HOLDING_TARGET = "HOLDING_TARGET"
    WAIT_LEAVE_TARGET = "WAIT_LEAVE_TARGET"

    def __init__(self, target_reps: int = 30, hold_frames: int = 8):
        self.target_reps = target_reps
        self.hold_frames = hold_frames
        self.action = None
        self.start_pose = []
        self.rep_count = 0
        self.state = "stopped"
        self.hold_count = 0
        self.running = False
        self.paused = False
        self.started_at = None
        self.completed_scores = []
        self.records = []
        self.pose_history = deque(maxlen=20)
        self.last_completion = 0.0
        self.last_stability = "无数据"
        self._state_before_pause = self.WAIT_TARGET

    @staticmethod
    def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def normalize_finger_scores(scores: Optional[Dict[str, int]]) -> Dict[str, int]:
        normalized = {key: 0 for key in FINGER_SCORE_KEYS}
        if not scores:
            return normalized
        for key, value in scores.items():
            normalized_key = FINGER_SCORE_ALIASES.get(str(key), str(key))
            if normalized_key in normalized:
                normalized[normalized_key] = int(RehabTrainer.clamp(float(value)))
        return normalized

    @staticmethod
    def completion_from_finger_scores(
        current_scores: Optional[Dict[str, int]],
        target_scores: Optional[Dict[str, int]],
    ) -> float:
        current = RehabTrainer.normalize_finger_scores(current_scores)
        target = RehabTrainer.normalize_finger_scores(target_scores)
        mean_error = sum(abs(current[key] - target[key]) for key in FINGER_SCORE_KEYS) / len(FINGER_SCORE_KEYS)
        return RehabTrainer.clamp(100.0 - mean_error)

    def start(self, action: RehabAction, start_pose: List[int], target_reps: int) -> None:
        self.action = action
        self.start_pose = list(start_pose)
        self.target_reps = max(1, int(target_reps))
        self.hold_frames = max(1, int(action.hold_frames))
        self.rep_count = 0
        self.state = self.WAIT_TARGET
        self.hold_count = 0
        self.running = True
        self.paused = False
        self.started_at = time.time()
        self.completed_scores = []
        self.records = []
        self.pose_history.clear()
        self.last_completion = 0.0
        self.last_stability = "无数据"

    def pause(self) -> None:
        if self.running:
            self._state_before_pause = self.state
            self.paused = True
            self.state = "已暂停"

    def resume(self) -> None:
        if self.running:
            self.paused = False
            self.state = self._state_before_pause or self.WAIT_TARGET

    def finish(self) -> None:
        self.running = False
        self.paused = False
        self.state = "已结束"

    def reset_count(self) -> None:
        self.rep_count = 0
        self.hold_count = 0
        self.completed_scores = []
        self.records = []
        self.pose_history.clear()
        if self.running:
            self.state = self.WAIT_TARGET

    def stability(self, pose: Optional[List[int]]) -> str:
        if pose:
            self.pose_history.append(list(pose))
        if len(self.pose_history) < 5:
            self.last_stability = "无数据"
            return self.last_stability

        total = 0.0
        samples = 0
        history = list(self.pose_history)
        for previous, current in zip(history[:-1], history[1:]):
            count = min(len(previous), len(current))
            if count <= 0:
                continue
            total += sum(abs(current[i] - previous[i]) for i in range(count)) / count
            samples += 1
        avg_delta = total / samples if samples else 0.0
        if avg_delta < 3:
            self.last_stability = "良好"
        elif avg_delta < 8:
            self.last_stability = "一般"
        else:
            self.last_stability = "较差"
        return self.last_stability

    def update(self, pose: Optional[List[int]], finger_scores: Dict[str, int]) -> Dict[str, object]:
        counted = False
        if not self.action or not pose or not finger_scores:
            if self.running and not self.paused and self.state == self.HOLDING_TARGET:
                self.state = self.WAIT_TARGET
                self.hold_count = 0
            return {
                "completion": 0.0,
                "start_completion": 0.0,
                "stability": self.last_stability,
                "counted": False,
                "state": self.state,
            }

        current_finger_scores = self.normalize_finger_scores(finger_scores)
        target_finger_scores = self.normalize_finger_scores(self.action.target_finger_scores)
        completion = self.completion_from_finger_scores(current_finger_scores, target_finger_scores)
        start_completion = 0.0
        stability = self.stability(pose)
        self.last_completion = completion

        if self.running and not self.paused:
            hit_threshold = self.clamp(float(self.action.hit_threshold))
            reset_threshold = min(self.clamp(float(self.action.reset_threshold)), hit_threshold)
            hold_frames = max(1, int(self.action.hold_frames))
            self.hold_frames = hold_frames

            if self.state not in (self.WAIT_TARGET, self.HOLDING_TARGET, self.WAIT_LEAVE_TARGET):
                self.state = self.WAIT_TARGET

            if self.state == self.WAIT_TARGET:
                if completion >= hit_threshold:
                    self.state = self.HOLDING_TARGET
                    self.hold_count += 1
                else:
                    self.hold_count = 0
            elif self.state == self.HOLDING_TARGET:
                if completion >= hit_threshold:
                    self.hold_count += 1
                else:
                    self.state = self.WAIT_TARGET
                    self.hold_count = 0

                if self.hold_count >= hold_frames:
                    self.rep_count += 1
                    counted = True
                    self.completed_scores.append(completion)
                    self.state = self.WAIT_LEAVE_TARGET
                    self.hold_count = 0
            elif self.state == self.WAIT_LEAVE_TARGET:
                if completion <= reset_threshold:
                    self.state = self.WAIT_TARGET
                    self.hold_count = 0

            if self.rep_count >= self.target_reps:
                self.finish()

        if self.running and not self.paused:
            self.records.append(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "action": self.action.name,
                    "pose": list(pose),
                    "target_robot_pose": list(self.action.target_robot_pose),
                    "target_finger_scores": dict(target_finger_scores),
                    "completion": round(completion, 2),
                    "hit_threshold": float(self.action.hit_threshold),
                    "reset_threshold": float(self.action.reset_threshold),
                    "hold_frames": int(self.action.hold_frames),
                    "finger_scores": dict(current_finger_scores),
                    "stability": stability,
                    "counted": counted,
                }
            )

        return {
            "completion": completion,
            "start_completion": start_completion,
            "stability": stability,
            "counted": counted,
            "state": self.state,
        }

    def summary(self) -> Dict[str, object]:
        scores = [float(record["completion"]) for record in self.records]
        duration = 0.0 if not self.started_at else time.time() - self.started_at
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        return {
            "action": self.action.name if self.action else "-",
            "target_reps": self.target_reps,
            "rep_count": self.rep_count,
            "avg_completion": avg_score,
            "max_completion": max_score,
            "min_completion": min_score,
            "stability": self.last_stability,
            "duration": duration,
        }


class LinkerHandController:
    def __init__(self):
        self.api = None
        self.setting = None
        self.hand_type = ""
        self.hand_joint = ""
        self.can = ""
        self.modbus = ""
        self.open_pose = []
        self.fist_pose = []
        self.connected = False
        self.sending_enabled = False

    def connect(self) -> str:
        if self.connected:
            return "机械手已经连接"

        self.setting = vision.load_hand_setting()
        self.hand_type = str(self.setting["hand_type"])
        self.hand_joint = str(self.setting["hand_joint"])
        self.can = str(self.setting["can"])
        self.modbus = str(self.setting["modbus"])
        self.open_pose, self.fist_pose = vision.choose_open_and_fist_poses(self.hand_joint)

        self.api = vision.LinkerHandApi(
            hand_joint=self.hand_joint,
            hand_type=self.hand_type,
            modbus=self.modbus,
            can=self.can,
        )
        self.connected = True
        return "机械手连接成功: {} {}, CAN={}, MODBUS={}".format(
            self.hand_type, self.hand_joint, self.can, self.modbus
        )

    def disconnect(self) -> str:
        self.sending_enabled = False
        if self.api is not None:
            self.api.close_can()
        self.api = None
        self.connected = False
        return "机械手已断开"

    def send_pose(self, pose: Optional[List[int]]) -> None:
        if not self.connected or not self.sending_enabled or self.api is None or not pose:
            return
        self.api.finger_move(pose)

    def send_manual_pose(self, pose: List[int]) -> None:
        if not self.connected or self.api is None:
            raise RuntimeError("机械手未连接")
        self.api.finger_move(pose)

    def stop_sending(self) -> None:
        self.sending_enabled = False


class VisionControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linker Hand 视觉交互控制")
        self.resize(1280, 760)

        self.controller = LinkerHandController()
        self.worker = None
        self.pose_smoother = PoseSmoother(alpha=0.25, deadband=5, max_step=8)
        self.rehab_trainer = RehabTrainer(target_reps=30)
        self.rehab_actions: Dict[str, RehabAction] = {}
        self.current_raw_pose = None
        self.current_pose = None
        self.current_curls = {}
        self.current_finger_scores = {}
        self.current_mode = "未连接"
        self.last_send_time = 0.0
        self.last_no_hand_log_time = 0.0
        self.last_send_log_time = 0.0
        self.send_interval = 0.08
        self.startup_self_check_done = False
        self.is_training = False
        self.enable_robot_control = False
        self.training_status = "IDLE"
        self.rehab_template_path = REHAB_TEMPLATE_PATH
        self.rehab_template_overrides = self.load_rehab_template_overrides()

        self._build_ui()
        self._connect_signals()
        self._update_state()
        self._log("程序已启动。将执行自检；不会自动连接或控制机械手。")
        QTimer.singleShot(100, self.run_startup_self_check)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        self.video_labels = []
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        vision_page = QWidget()
        vision_root = QHBoxLayout(vision_page)
        vision_root.setContentsMargins(0, 0, 0, 0)
        vision_root.setSpacing(12)
        self.tabs.addTab(vision_page, "视觉控制")

        self.video_label = QLabel("摄像头画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 520)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
        self.video_labels.append(self.video_label)
        vision_root.addWidget(self.video_label, stretch=3)

        side = QVBoxLayout()
        side.setSpacing(10)
        vision_root.addLayout(side, stretch=1)

        control_group = QGroupBox("控制")
        control_layout = QGridLayout(control_group)
        side.addWidget(control_group)

        self.connect_btn = QPushButton("连接机械手")
        self.disconnect_btn = QPushButton("断开连接")
        self.start_btn = QPushButton("开始视觉控制")
        self.stop_btn = QPushButton("停止视觉控制")
        self.open_btn = QPushButton("张开手")
        self.fist_btn = QPushButton("握拳")
        self.estop_btn = QPushButton("急停/停止发送")

        control_layout.addWidget(self.connect_btn, 0, 0)
        control_layout.addWidget(self.disconnect_btn, 0, 1)
        control_layout.addWidget(self.start_btn, 1, 0)
        control_layout.addWidget(self.stop_btn, 1, 1)
        control_layout.addWidget(self.open_btn, 2, 0)
        control_layout.addWidget(self.fist_btn, 2, 1)
        control_layout.addWidget(self.estop_btn, 3, 0, 1, 2)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(0)

        control_layout.addWidget(QLabel("摄像头编号"), 4, 0)
        control_layout.addWidget(self.camera_spin, 4, 1)

        filter_group = QGroupBox("实时滤波调节")
        filter_layout = QGridLayout(filter_group)
        side.addWidget(filter_group)

        self.alpha_value_label = QLabel("alpha: 0.25")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 100)
        self.alpha_slider.setValue(25)

        self.deadband_value_label = QLabel("deadband: 5")
        self.deadband_slider = QSlider(Qt.Horizontal)
        self.deadband_slider.setRange(0, 30)
        self.deadband_slider.setValue(5)

        self.max_step_value_label = QLabel("max_step: 8")
        self.max_step_slider = QSlider(Qt.Horizontal)
        self.max_step_slider.setRange(1, 50)
        self.max_step_slider.setValue(8)

        filter_layout.addWidget(self.alpha_value_label, 0, 0)
        filter_layout.addWidget(self.alpha_slider, 1, 0)
        filter_layout.addWidget(self.deadband_value_label, 2, 0)
        filter_layout.addWidget(self.deadband_slider, 3, 0)
        filter_layout.addWidget(self.max_step_value_label, 4, 0)
        filter_layout.addWidget(self.max_step_slider, 5, 0)

        status_group = QGroupBox("状态")
        status_layout = QGridLayout(status_group)
        side.addWidget(status_group)

        self.sdk_status_label = QLabel("未连接")
        self.can_label = QLabel("-")
        self.mode_label = QLabel("未连接")
        self.hand_detected_label = QLabel("否")
        self.fps_label = QLabel("0.0")
        self.pose_label = QLabel("[]")
        self.pose_label.setWordWrap(True)

        status_layout.addWidget(QLabel("SDK连接状态"), 0, 0)
        status_layout.addWidget(self.sdk_status_label, 0, 1)
        status_layout.addWidget(QLabel("CAN通道"), 1, 0)
        status_layout.addWidget(self.can_label, 1, 1)
        status_layout.addWidget(QLabel("当前模式"), 2, 0)
        status_layout.addWidget(self.mode_label, 2, 1)
        status_layout.addWidget(QLabel("检测到手"), 3, 0)
        status_layout.addWidget(self.hand_detected_label, 3, 1)
        status_layout.addWidget(QLabel("FPS"), 4, 0)
        status_layout.addWidget(self.fps_label, 4, 1)
        status_layout.addWidget(QLabel("控制数组"), 5, 0, Qt.AlignTop)
        status_layout.addWidget(self.pose_label, 5, 1)

        rehab_page = QWidget()
        rehab_root = QHBoxLayout(rehab_page)
        rehab_root.setContentsMargins(0, 0, 0, 0)
        rehab_root.setSpacing(12)
        self.tabs.addTab(rehab_page, "康复训练")

        self.rehab_video_label = QLabel("摄像头画面")
        self.rehab_video_label.setAlignment(Qt.AlignCenter)
        self.rehab_video_label.setMinimumSize(760, 520)
        self.rehab_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rehab_video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
        self.video_labels.append(self.rehab_video_label)
        rehab_root.addWidget(self.rehab_video_label, stretch=3)

        rehab_side = QVBoxLayout()
        rehab_side.setSpacing(10)
        rehab_root.addLayout(rehab_side, stretch=1)

        rehab_group = QGroupBox("康复训练模式")
        rehab_layout = QGridLayout(rehab_group)
        rehab_side.addWidget(rehab_group)

        self.rehab_action_combo = QComboBox()
        self.rehab_completion_label = QLabel("0%")
        self.rehab_target_completion_label = QLabel("80%")
        self.rehab_count_label = QLabel("0 / 30")
        self.rehab_target_spin = QSpinBox()
        self.rehab_target_spin.setRange(1, 999)
        self.rehab_target_spin.setValue(30)
        self.rehab_stability_label = QLabel("无数据")
        self.rehab_state_label = QLabel("未开始")
        self.rehab_description_label = QLabel("-")
        self.rehab_description_label.setWordWrap(True)
        self.current_finger_scores_label = QLabel("-")
        self.current_finger_scores_label.setWordWrap(True)
        self.target_finger_scores_label = QLabel("-")
        self.target_finger_scores_label.setWordWrap(True)

        rehab_layout.addWidget(QLabel("当前训练动作"), 0, 0)
        rehab_layout.addWidget(self.rehab_action_combo, 0, 1)
        rehab_layout.addWidget(QLabel("当前完成度"), 1, 0)
        rehab_layout.addWidget(self.rehab_completion_label, 1, 1)
        rehab_layout.addWidget(QLabel("目标完成度"), 2, 0)
        rehab_layout.addWidget(self.rehab_target_completion_label, 2, 1)
        rehab_layout.addWidget(QLabel("有效训练次数"), 3, 0)
        rehab_layout.addWidget(self.rehab_count_label, 3, 1)
        rehab_layout.addWidget(QLabel("目标训练次数"), 4, 0)
        rehab_layout.addWidget(self.rehab_target_spin, 4, 1)
        rehab_layout.addWidget(QLabel("动作稳定性"), 5, 0)
        rehab_layout.addWidget(self.rehab_stability_label, 5, 1)
        rehab_layout.addWidget(QLabel("当前训练状态"), 6, 0)
        rehab_layout.addWidget(self.rehab_state_label, 6, 1)
        rehab_layout.addWidget(QLabel("动作说明"), 7, 0, Qt.AlignTop)
        rehab_layout.addWidget(self.rehab_description_label, 7, 1)
        rehab_layout.addWidget(QLabel("当前五指活动度"), 8, 0, Qt.AlignTop)
        rehab_layout.addWidget(self.current_finger_scores_label, 8, 1)
        rehab_layout.addWidget(QLabel("目标五指活动度"), 9, 0, Qt.AlignTop)
        rehab_layout.addWidget(self.target_finger_scores_label, 9, 1)

        self.start_rehab_btn = QPushButton("开始训练")
        self.pause_rehab_btn = QPushButton("暂停训练")
        self.finish_rehab_btn = QPushButton("结束训练")
        self.reset_rehab_btn = QPushButton("重置计数")
        self.demo_rehab_btn = QPushButton("灵巧手演示标准动作")
        self.save_template_btn = QPushButton("保存当前姿态为该动作模板")
        self.report_rehab_btn = QPushButton("生成训练报告")
        self.save_rehab_btn = QPushButton("保存训练数据")

        rehab_layout.addWidget(self.start_rehab_btn, 10, 0)
        rehab_layout.addWidget(self.pause_rehab_btn, 10, 1)
        rehab_layout.addWidget(self.finish_rehab_btn, 11, 0)
        rehab_layout.addWidget(self.reset_rehab_btn, 11, 1)
        rehab_layout.addWidget(self.demo_rehab_btn, 12, 0, 1, 2)
        rehab_layout.addWidget(self.save_template_btn, 13, 0, 1, 2)
        rehab_layout.addWidget(self.report_rehab_btn, 14, 0)
        rehab_layout.addWidget(self.save_rehab_btn, 14, 1)

        self.finger_bars = {}
        finger_labels = [
            ("thumb", "拇指活动度"),
            ("index", "食指活动度"),
            ("middle", "中指活动度"),
            ("ring", "无名指活动度"),
            ("little", "小指活动度"),
        ]
        row = 15
        for key, text in finger_labels:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self.finger_bars[key] = bar
            rehab_layout.addWidget(QLabel(text), row, 0)
            rehab_layout.addWidget(bar, row, 1)
            row += 1

        rehab_side.addStretch(1)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(140)
        log_layout.addWidget(self.log_edit)
        root.addWidget(log_group)

    def _connect_signals(self) -> None:
        self.connect_btn.clicked.connect(self.connect_hand)
        self.disconnect_btn.clicked.connect(self.disconnect_hand)
        self.start_btn.clicked.connect(self.start_vision)
        self.stop_btn.clicked.connect(self.stop_vision)
        self.open_btn.clicked.connect(self.send_open_pose)
        self.fist_btn.clicked.connect(self.send_fist_pose)
        self.estop_btn.clicked.connect(self.emergency_stop)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.deadband_slider.valueChanged.connect(self.on_deadband_changed)
        self.max_step_slider.valueChanged.connect(self.on_max_step_changed)
        self.rehab_action_combo.currentTextChanged.connect(self.update_selected_rehab_action)
        self.rehab_target_spin.valueChanged.connect(self.update_rehab_labels)
        self.start_rehab_btn.clicked.connect(self.start_rehab_training)
        self.pause_rehab_btn.clicked.connect(self.pause_rehab_training)
        self.finish_rehab_btn.clicked.connect(self.finish_rehab_training)
        self.reset_rehab_btn.clicked.connect(self.reset_rehab_count)
        self.demo_rehab_btn.clicked.connect(self.demo_rehab_action)
        self.save_template_btn.clicked.connect(self.save_current_pose_as_rehab_template)
        self.report_rehab_btn.clicked.connect(self.generate_rehab_report)
        self.save_rehab_btn.clicked.connect(self.save_rehab_records)

    def _pose_for_fingers(
        self,
        hand_joint: str,
        open_pose: List[int],
        fist_pose: List[int],
        fingers: List[str],
        ratio: float = 1.0,
    ) -> List[int]:
        pose = list(open_pose)
        mapping = vision.FINGER_JOINT_INDEXES.get(hand_joint.upper(), {})
        for finger in fingers:
            for index in mapping.get(finger, []):
                if 0 <= index < len(pose) and index < len(fist_pose):
                    pose[index] = int(round(open_pose[index] + ratio * (fist_pose[index] - open_pose[index])))
        return [int(max(0, min(255, value))) for value in pose]

    def load_rehab_template_overrides(self) -> Dict[str, Dict[str, int]]:
        if not os.path.exists(self.rehab_template_path):
            return {}
        try:
            with open(self.rehab_template_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        templates = data.get("templates", data)
        if not isinstance(templates, dict):
            return {}
        return {
            str(name): RehabTrainer.normalize_finger_scores(scores)
            for name, scores in templates.items()
            if isinstance(scores, dict)
        }

    def save_rehab_template_overrides(self) -> None:
        templates = {
            name: self.finger_scores_for_json(scores)
            for name, scores in self.rehab_template_overrides.items()
        }
        data = {"templates": templates}
        with open(self.rehab_template_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def finger_scores_for_json(self, scores: Dict[str, int]) -> Dict[str, int]:
        normalized = RehabTrainer.normalize_finger_scores(scores)
        return {
            "thumb": normalized["thumb"],
            "index": normalized["index"],
            "middle": normalized["middle"],
            "ring": normalized["ring"],
            "pinky": normalized["little"],
        }

    def format_finger_scores(self, scores: Dict[str, int]) -> str:
        normalized = RehabTrainer.normalize_finger_scores(scores)
        return "拇指 {thumb}% / 食指 {index}% / 中指 {middle}% / 无名指 {ring}% / 小指 {little}%".format(
            **normalized
        )

    def build_rehab_actions(
        self,
        hand_joint: str,
        open_pose: List[int],
        fist_pose: List[int],
    ) -> Dict[str, RehabAction]:
        actions = {}
        for config in REHAB_ACTION_CONFIGS:
            if config.get("pose") == "open":
                target_pose = list(open_pose)
            elif config.get("pose") == "fist":
                target_pose = list(fist_pose)
            else:
                target_pose = self._pose_for_fingers(
                    hand_joint,
                    open_pose,
                    fist_pose,
                    list(config["fingers"]),
                    float(config["ratio"]),
                )
            default_scores = RehabTrainer.normalize_finger_scores(config["target_finger_scores"])
            target_scores = self.rehab_template_overrides.get(str(config["name"]), default_scores)

            action = RehabAction(
                name=str(config["name"]),
                target_robot_pose=target_pose,
                target_finger_scores=dict(target_scores),
                description=str(config["description"]),
                hit_threshold=float(config["hit_threshold"]),
                reset_threshold=float(config["reset_threshold"]),
                hold_frames=int(config["hold_frames"]),
            )
            actions[action.name] = action
        return actions

    def current_rehab_action(self) -> Optional[RehabAction]:
        return self.rehab_actions.get(self.rehab_action_combo.currentText())

    def rehab_state_text(self, state: str) -> str:
        state_text = {
            RehabTrainer.WAIT_TARGET: "WAIT_TARGET：等待达到目标动作",
            RehabTrainer.HOLDING_TARGET: "HOLDING_TARGET：正在保持目标动作",
            RehabTrainer.WAIT_LEAVE_TARGET: "WAIT_LEAVE_TARGET：等待离开目标动作",
        }
        return state_text.get(state, state)

    def rehab_threshold_text(self, action: RehabAction) -> str:
        return "命中 {:.0f}% / 复位 {:.0f}% / 保持 {} 帧".format(
            action.hit_threshold,
            action.reset_threshold,
            action.hold_frames,
        )

    def update_selected_rehab_action(self) -> None:
        action = self.current_rehab_action()
        if not action:
            self.rehab_description_label.setText("-")
            self.rehab_target_completion_label.setText("80%")
            self.target_finger_scores_label.setText("-")
            return
        self.rehab_description_label.setText(action.description)
        self.rehab_target_completion_label.setText(self.rehab_threshold_text(action))
        self.target_finger_scores_label.setText(self.format_finger_scores(action.target_finger_scores))
        self.update_rehab_labels()

    def update_rehab_labels(self) -> None:
        action = self.current_rehab_action()
        target = self.rehab_target_spin.value()
        self.rehab_count_label.setText("{} / {}".format(self.rehab_trainer.rep_count, target))
        self.rehab_state_label.setText(self.rehab_state_text(self.rehab_trainer.state))
        if action:
            self.rehab_target_completion_label.setText(self.rehab_threshold_text(action))
            self.target_finger_scores_label.setText(self.format_finger_scores(action.target_finger_scores))

    def finger_scores_from_curls(self, curls) -> Dict[str, int]:
        if not curls:
            return {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0}
        scores = {}
        for key in FINGER_SCORE_KEYS:
            scores[key] = int(max(0, min(100, round(float(curls.get(key, 0.0)) * 100))))
        return scores

    def update_finger_bars(self, finger_scores: Dict[str, int]) -> None:
        for key, bar in self.finger_bars.items():
            bar.setValue(int(finger_scores.get(key, 0)))

    def start_rehab_training(self) -> None:
        if self.is_training:
            return

        self.is_training = False
        self.enable_robot_control = False
        if not self.controller.connected and not self.ensure_hand_connected(show_message=True):
            self.training_status = "IDLE"
            self._set_mode("训练未开始")
            self._update_state()
            return

        action = self.current_rehab_action()
        if not action:
            self._log("请先选择康复训练动作")
            self.training_status = "IDLE"
            self._set_mode("训练未开始")
            self._update_state()
            return

        start_pose = self.controller.open_pose or [255] * len(action.target_robot_pose)
        self.rehab_trainer.start(action, start_pose, self.rehab_target_spin.value())
        self.is_training = True
        self.training_status = "RUNNING"
        self.enable_robot_control = False

        if self.worker is None or not self.worker.isRunning():
            self.start_vision()
        if self.worker is None or not self.worker.isRunning():
            self.stop_rehab_training("视觉识别启动失败，训练未开始", auto_save=False)
            return

        self.enable_robot_control = self.controller.connected
        self.controller.sending_enabled = self.enable_robot_control
        self.pause_rehab_btn.setText("暂停训练")
        self._set_mode("训练中")
        self.rehab_state_label.setText("RUNNING：训练中")
        self.update_rehab_labels()
        self._log("训练开始: {}".format(action.name))
        self._update_state()

    def pause_rehab_training(self) -> None:
        if self.rehab_trainer.paused:
            self.rehab_trainer.resume()
            self.pause_rehab_btn.setText("暂停训练")
            self._log("训练继续")
        else:
            self.rehab_trainer.pause()
            self.pause_rehab_btn.setText("继续训练")
            self._log("训练暂停")
        self.update_rehab_labels()

    def finish_rehab_training(self) -> None:
        self.stop_rehab_training("用户结束训练", auto_save=True)

    def stop_rehab_training(self, reason: str, auto_save: bool = True) -> None:
        was_training = self.is_training or self.rehab_trainer.running
        self.is_training = False
        self.training_status = "FINISHED"
        self.enable_robot_control = False
        self.controller.stop_sending()
        self.rehab_trainer.finish()
        self.pause_rehab_btn.setText("暂停训练")
        self.stop_vision()
        self._set_mode("训练已结束")
        if reason:
            self._log(reason)
        if auto_save and self.rehab_trainer.records:
            self.save_rehab_records()
        elif auto_save and was_training:
            self._log("暂无训练数据可保存")
        self.update_rehab_labels()
        self.rehab_state_label.setText("FINISHED：训练已结束")
        self._update_state()

    def on_target_count_reached(self) -> None:
        self.stop_rehab_training("已达到目标次数，训练结束", auto_save=True)

    def reset_rehab_count(self) -> None:
        self.rehab_trainer.reset_count()
        self.update_rehab_labels()
        self._log("训练计数已重置")

    def demo_rehab_action(self) -> None:
        action = self.current_rehab_action()
        if not action:
            self._log("请先选择康复训练动作")
            return
        if not self.controller.connected:
            self._log("机械手未连接，不能演示标准动作")
            return
        try:
            self.controller.send_manual_pose(action.target_robot_pose)
            self._log("已演示标准动作: {}".format(action.name))
        except Exception as exc:
            self._log("演示标准动作失败: {}".format(exc))

    def save_current_pose_as_rehab_template(self) -> None:
        action = self.current_rehab_action()
        if not action:
            self._log("请先选择康复训练动作")
            return
        if not self.current_finger_scores:
            self._log("当前没有可保存的五指活动度，请先启动视觉识别并检测到手")
            return
        scores = RehabTrainer.normalize_finger_scores(self.current_finger_scores)
        action.target_finger_scores = dict(scores)
        self.rehab_template_overrides[action.name] = dict(scores)
        try:
            self.save_rehab_template_overrides()
            self.target_finger_scores_label.setText(self.format_finger_scores(scores))
            self._log("已保存当前姿态为动作模板: {} {}".format(action.name, self.format_finger_scores(scores)))
        except Exception as exc:
            self._log("保存动作模板失败: {}".format(exc))

    def _ensure_output_dir(self, folder_name: str) -> str:
        path = os.path.join(CURRENT_DIR, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    def save_rehab_records(self) -> None:
        records = self.rehab_trainer.records
        if not records:
            self._log("暂无训练数据可保存")
            return
        output_dir = self._ensure_output_dir("rehab_records")
        file_path = os.path.join(output_dir, "rehab_record_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "action",
                        "pose",
                        "target_robot_pose",
                        "target_finger_scores",
                        "completion",
                        "hit_threshold",
                        "reset_threshold",
                        "hold_frames",
                        "finger_scores",
                        "stability",
                        "counted",
                    ],
                )
                writer.writeheader()
                for record in records:
                    row = dict(record)
                    row["pose"] = json.dumps(row["pose"], ensure_ascii=False)
                    row["target_robot_pose"] = json.dumps(row["target_robot_pose"], ensure_ascii=False)
                    row["target_finger_scores"] = json.dumps(row["target_finger_scores"], ensure_ascii=False)
                    row["finger_scores"] = json.dumps(row["finger_scores"], ensure_ascii=False)
                    writer.writerow(row)
            self._log("训练数据已保存: {}".format(file_path))
        except Exception as exc:
            self._log("保存训练数据失败: {}".format(exc))

    def generate_rehab_report(self) -> None:
        summary = self.rehab_trainer.summary()
        output_dir = self._ensure_output_dir("rehab_reports")
        file_path = os.path.join(output_dir, "rehab_report_{}.txt".format(time.strftime("%Y%m%d_%H%M%S")))
        avg_completion = float(summary["avg_completion"])
        if avg_completion >= 85:
            suggestion = "动作完成较好，可适当增加训练难度。"
        elif avg_completion >= 70:
            suggestion = "动作完成基本达标，建议继续保持训练。"
        else:
            suggestion = "动作完成度偏低，建议降低目标难度或延长训练时间。"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("手功能康复训练辅助报告\n")
                f.write("训练日期: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("训练动作: {}\n".format(summary["action"]))
                f.write("目标次数: {}\n".format(summary["target_reps"]))
                f.write("实际有效次数: {}\n".format(summary["rep_count"]))
                f.write("平均完成度: {:.2f}%\n".format(summary["avg_completion"]))
                f.write("最高完成度: {:.2f}%\n".format(summary["max_completion"]))
                f.write("最低完成度: {:.2f}%\n".format(summary["min_completion"]))
                f.write("平均稳定性: {}\n".format(summary["stability"]))
                f.write("训练总时长: {:.1f} 秒\n".format(summary["duration"]))
                f.write("建议: {}\n".format(suggestion))
            self._log("训练报告已保存: {}".format(file_path))
        except Exception as exc:
            self._log("生成训练报告失败: {}".format(exc))

    def _log(self, message: str) -> None:
        self.log_edit.append("[{}] {}".format(time.strftime("%H:%M:%S"), message))

    def run_startup_self_check(self) -> None:
        if self.startup_self_check_done:
            return
        self.startup_self_check_done = True

        self._log("开始启动自检")
        self._log("Python: {}".format(sys.version.split()[0]))
        self._log("项目根目录: {}".format(PROJECT_ROOT))

        try:
            setting = vision.load_hand_setting()
            hand_joint = str(setting["hand_joint"])
            open_pose, fist_pose = vision.choose_open_and_fist_poses(hand_joint)
            expected_len = vision.expected_joint_count(hand_joint)
            self._log(
                "配置检查通过: {} {}, CAN={}, MODBUS={}".format(
                    setting["hand_type"], hand_joint, setting["can"], setting["modbus"]
                )
            )
            self._log(
                "位姿长度检查: open={}, fist={}, expected={}".format(
                    len(open_pose), len(fist_pose), expected_len
                )
            )
            if len(open_pose) != expected_len or len(fist_pose) != expected_len:
                self._log("位姿长度不匹配，请检查 gui_control/config/constants.py")
        except Exception as exc:
            self._log("配置自检失败: {}".format(exc))

        cap = None
        try:
            camera_index = self.camera_spin.value()
            cap = vision.open_camera(camera_index)
            if cap is not None and cap.isOpened():
                self._log("摄像头自检通过: index={}".format(camera_index))
            else:
                self._log("摄像头自检失败: index={} 无法打开".format(camera_index))
        except Exception as exc:
            self._log("摄像头自检异常: {}".format(exc))
        finally:
            if cap is not None:
                cap.release()

        try:
            setting = vision.load_hand_setting()
            hand_joint = str(setting["hand_joint"])
            open_pose, fist_pose = vision.choose_open_and_fist_poses(hand_joint)
            self.rehab_actions = self.build_rehab_actions(hand_joint, open_pose, fist_pose)
            self.rehab_action_combo.clear()
            self.rehab_action_combo.addItems(list(self.rehab_actions.keys()))
            self.update_selected_rehab_action()
            self._log("康复动作库已加载: {} 个动作".format(len(self.rehab_actions)))
        except Exception as exc:
            self._log("康复动作库加载失败: {}".format(exc))

        self._log("启动自检完成。请按需点击“连接机械手”后再发送控制指令。")
        self._update_state()

    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        self.mode_label.setText(mode)

    def _update_state(self) -> None:
        connected = self.controller.connected
        running = self.worker is not None and self.worker.isRunning()

        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.open_btn.setEnabled(connected)
        self.fist_btn.setEnabled(connected)
        self.estop_btn.setEnabled(connected or running)
        self.camera_spin.setEnabled(not running)

        self.sdk_status_label.setText("已连接" if connected else "未连接")
        self.can_label.setText(self.controller.can if connected else "-")
        if hasattr(self, "start_rehab_btn"):
            self.start_rehab_btn.setEnabled(not self.is_training)
            self.finish_rehab_btn.setEnabled(self.is_training)
        if not connected and not running:
            self._set_mode("未连接")

    def ensure_hand_connected(self, show_message: bool = True) -> bool:
        try:
            message = self.controller.connect()
            self._log(message)
            self._log("张开位姿: {}".format(self.controller.open_pose))
            self._log("握拳位姿: {}".format(self.controller.fist_pose))
            self.rehab_actions = self.build_rehab_actions(
                self.controller.hand_joint,
                self.controller.open_pose,
                self.controller.fist_pose,
            )
            current_action = self.rehab_action_combo.currentText()
            self.rehab_action_combo.clear()
            self.rehab_action_combo.addItems(list(self.rehab_actions.keys()))
            if current_action in self.rehab_actions:
                self.rehab_action_combo.setCurrentText(current_action)
            self.update_selected_rehab_action()
            self._set_mode("已连接")
            return True
        except Exception as exc:
            self._log("机械手连接失败: {}".format(exc))
            if show_message:
                QMessageBox.warning(self, "连接失败", str(exc))
            return False

    @pyqtSlot()
    def connect_hand(self) -> None:
        self.ensure_hand_connected(show_message=True)
        self._update_state()

    @pyqtSlot()
    def disconnect_hand(self) -> None:
        self.emergency_stop()
        self.stop_vision()
        try:
            self._log(self.controller.disconnect())
        except Exception as exc:
            self._log("断开连接时出错: {}".format(exc))
        self._update_state()

    @pyqtSlot()
    def start_vision(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        if not self.rehab_actions:
            try:
                setting = vision.load_hand_setting()
                hand_joint = str(setting["hand_joint"])
                open_pose, fist_pose = vision.choose_open_and_fist_poses(hand_joint)
                self.rehab_actions = self.build_rehab_actions(hand_joint, open_pose, fist_pose)
                self.rehab_action_combo.clear()
                self.rehab_action_combo.addItems(list(self.rehab_actions.keys()))
                self.update_selected_rehab_action()
            except Exception as exc:
                self._log("视觉启动前加载动作库失败: {}".format(exc))

        hand_joint = self.controller.hand_joint
        open_pose = self.controller.open_pose
        fist_pose = self.controller.fist_pose
        if not self.controller.connected:
            try:
                setting = vision.load_hand_setting()
                hand_joint = str(setting["hand_joint"])
                open_pose, fist_pose = vision.choose_open_and_fist_poses(hand_joint)
            except Exception as exc:
                self._log("未连接机械手，且读取手型配置失败: {}".format(exc))
                self._update_state()
                return

        smoothing = self.alpha_slider.value() / 100.0
        self.pose_smoother.update_params(
            alpha=smoothing,
            deadband=self.deadband_slider.value(),
            max_step=self.max_step_slider.value(),
        )
        self.pose_smoother.reset()
        self.enable_robot_control = self.controller.connected
        self.controller.sending_enabled = self.enable_robot_control
        self.worker = VisionWorker(
            self.camera_spin.value(),
            hand_joint,
            open_pose,
            fist_pose,
            smoothing,
            self.deadband_slider.value(),
            self.max_step_slider.value(),
            self,
        )
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.result_ready.connect(self.update_result)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self._set_mode("视觉控制中" if self.controller.connected else "视觉识别中")
        if self.controller.connected:
            self._log("视觉控制已开始，检测到手后会调用 finger_move()")
        else:
            self._log("视觉识别已开始；未连接机械手，仅进行康复评分与记录")
        self._update_state()

    @pyqtSlot()
    def stop_vision(self) -> None:
        self.enable_robot_control = False
        self.controller.stop_sending()
        self.pose_smoother.reset()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            if self.worker.isRunning():
                self._log("视觉线程未能在 2 秒内退出")
            self.worker = None
        if self.controller.connected:
            self._set_mode("已停止")
        self._log("视觉控制已停止，实时发送已关闭")
        self._update_state()

    @pyqtSlot()
    def emergency_stop(self) -> None:
        if self.is_training or self.rehab_trainer.running:
            self.stop_rehab_training("急停：已停止训练和实时控制", auto_save=True)
            return
        self.enable_robot_control = False
        self.controller.stop_sending()
        self.pose_smoother.reset()
        if self.controller.connected:
            self._set_mode("已停止")
        self._log("急停：已停止发送实时控制指令")
        self._update_state()

    @pyqtSlot()
    def send_open_pose(self) -> None:
        try:
            self.controller.send_manual_pose(self.controller.open_pose)
            self.pose_smoother.reset(self.controller.open_pose)
            self.current_pose = list(self.controller.open_pose)
            self.pose_label.setText(str(self.current_pose))
            self._log("已发送张开手位姿")
        except Exception as exc:
            self._log("发送张开手失败: {}".format(exc))

    @pyqtSlot()
    def send_fist_pose(self) -> None:
        try:
            self.controller.send_manual_pose(self.controller.fist_pose)
            self.pose_smoother.reset(self.controller.fist_pose)
            self.current_pose = list(self.controller.fist_pose)
            self.pose_label.setText(str(self.current_pose))
            self._log("已发送握拳位姿")
        except Exception as exc:
            self._log("发送握拳失败: {}".format(exc))

    @pyqtSlot(int)
    def on_alpha_changed(self, value: int) -> None:
        alpha = value / 100.0
        self.alpha_value_label.setText("alpha: {:.2f}".format(alpha))
        self.pose_smoother.set_alpha(alpha)
        if self.worker is not None:
            self.worker.set_filter_params(alpha=alpha)

    @pyqtSlot(int)
    def on_deadband_changed(self, value: int) -> None:
        self.deadband_value_label.setText("deadband: {}".format(value))
        self.pose_smoother.set_deadband(value)
        if self.worker is not None:
            self.worker.set_filter_params(deadband=value)

    @pyqtSlot(int)
    def on_max_step_changed(self, value: int) -> None:
        self.max_step_value_label.setText("max_step: {}".format(value))
        self.pose_smoother.set_max_step(value)
        if self.worker is not None:
            self.worker.set_filter_params(max_step=value)

    @pyqtSlot(object)
    def update_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        for video_label in self.video_labels:
            pixmap = QPixmap.fromImage(image).scaled(
                video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            video_label.setPixmap(pixmap)

    @pyqtSlot(bool, object, object, object, float)
    def update_result(self, detected: bool, raw_pose, smooth_pose, curls, fps: float) -> None:
        self.hand_detected_label.setText("是" if detected else "否")
        self.fps_label.setText("{:.1f}".format(fps))

        if detected and raw_pose and smooth_pose:
            raw_pose = list(raw_pose)
            smooth_pose = list(smooth_pose)
            self.current_raw_pose = raw_pose
            self.current_pose = smooth_pose
            self.current_curls = curls or {}
            self.pose_label.setText(str(self.current_pose))
            finger_scores = self.finger_scores_from_curls(self.current_curls)
            self.current_finger_scores = dict(finger_scores)
            self.current_finger_scores_label.setText(self.format_finger_scores(finger_scores))
            self.update_finger_bars(finger_scores)
            action = self.current_rehab_action()
            completion = 0.0
            if action:
                completion = RehabTrainer.completion_from_finger_scores(finger_scores, action.target_finger_scores)
                self.rehab_completion_label.setText("{:.0f}%".format(completion))
            result = self.rehab_trainer.update(smooth_pose, finger_scores)
            self.rehab_completion_label.setText("{:.0f}%".format(result["completion"] if self.rehab_trainer.action else (completion if action else 0.0)))
            self.rehab_stability_label.setText(str(result["stability"]))
            self.rehab_state_label.setText(self.rehab_state_text(str(result["state"])))
            self.rehab_count_label.setText("{} / {}".format(self.rehab_trainer.rep_count, self.rehab_target_spin.value()))
            if result["counted"]:
                self._log("有效训练次数增加: {} / {}".format(self.rehab_trainer.rep_count, self.rehab_target_spin.value()))
            if self.is_training and self.rehab_trainer.rep_count >= self.rehab_trainer.target_reps:
                self.on_target_count_reached()
                return
            now = time.time()
            if now - self.last_send_log_time >= 2.0:
                self._log("raw_pose: {}".format(raw_pose))
                self._log("smooth_pose: {}".format(smooth_pose))
                self.last_send_log_time = now
            if (
                self.enable_robot_control
                and self.controller.sending_enabled
                and now - self.last_send_time >= self.send_interval
            ):
                try:
                    self.controller.send_pose(smooth_pose)
                    self.last_send_time = now
                except Exception as exc:
                    if self.is_training:
                        self.stop_rehab_training("发送控制指令失败，训练已停止: {}".format(exc), auto_save=True)
                    else:
                        self.enable_robot_control = False
                        self.controller.stop_sending()
                        self.pose_smoother.reset()
                        self._set_mode("已停止")
                        self._log("发送控制指令失败，已停止发送: {}".format(exc))
        else:
            self.current_raw_pose = None
            self.current_curls = {}
            self.current_finger_scores = {}
            self.current_finger_scores_label.setText("未检测到手")
            self.update_finger_bars({"thumb": 0, "index": 0, "middle": 0, "ring": 0, "little": 0})
            result = self.rehab_trainer.update(None, {})
            self.rehab_state_label.setText(self.rehab_state_text(str(result["state"])))
            self.rehab_count_label.setText("{} / {}".format(self.rehab_trainer.rep_count, self.rehab_target_spin.value()))
            self.pose_smoother.reset()
            if self.enable_robot_control and self.controller.sending_enabled:
                now = time.time()
                if now - self.last_no_hand_log_time >= 2.0:
                    self._log("未检测到手，已重置 pose 滤波器，本帧不发送控制指令")
                    self.last_no_hand_log_time = now

    @pyqtSlot()
    def on_worker_finished(self) -> None:
        if self.is_training:
            self.stop_rehab_training("视觉线程已停止，训练结束", auto_save=True)
            return
        self.controller.stop_sending()
        self.enable_robot_control = False
        self.worker = None
        if self.controller.connected:
            self._set_mode("已停止")
        self._update_state()

    def closeEvent(self, event) -> None:
        self.is_training = False
        self.training_status = "FINISHED"
        self.enable_robot_control = False
        self.controller.stop_sending()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        try:
            if self.controller.connected:
                self.controller.disconnect()
        except Exception as exc:
            self._log("关闭 SDK 时出错: {}".format(exc))
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = VisionControlWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
