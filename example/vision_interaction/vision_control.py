#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision interaction demo for Linker Hand.

Pipeline:
    camera -> MediaPipe Hands -> finger curl values -> LinkerHandApi.finger_move()

Press q in the OpenCV window to quit.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
GUI_CONTROL_DIR = os.path.join(PROJECT_ROOT, "example", "gui_control")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if GUI_CONTROL_DIR not in sys.path:
    sys.path.insert(0, GUI_CONTROL_DIR)

import cv2
import mediapipe as mp

from config.constants import _HAND_CONFIGS
from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml


FINGERS = ("thumb", "index", "middle", "ring", "little")

FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

# These indexes follow the same joint list used by example/gui_control/config/constants.py.
FINGER_JOINT_INDEXES = {
    "O6": {
        "thumb": [0, 1],
        "index": [2],
        "middle": [3],
        "ring": [4],
        "little": [5],
    },
    "L6": {
        "thumb": [0, 1],
        "index": [2],
        "middle": [3],
        "ring": [4],
        "little": [5],
    },
    "L7": {
        "thumb": [0, 1, 6],
        "index": [2],
        "middle": [3],
        "ring": [4],
        "little": [5],
    },
    "L10": {
        "thumb": [0, 1, 9],
        "index": [2, 6],
        "middle": [3],
        "ring": [4, 7],
        "little": [5, 8],
    },
    "L20": {
        "thumb": [0, 5, 10, 15],
        "index": [1, 6, 16],
        "middle": [2, 7, 17],
        "ring": [3, 8, 18],
        "little": [4, 9, 19],
    },
    "G20": {
        "thumb": [0, 5, 10, 15],
        "index": [1, 6, 16],
        "middle": [2, 7, 17],
        "ring": [3, 8, 18],
        "little": [4, 9, 19],
    },
    "L21": {
        "thumb": [0, 5, 10, 15, 20],
        "index": [1, 6, 21],
        "middle": [2, 7, 22],
        "ring": [3, 8, 23],
        "little": [4, 9, 24],
    },
    "L25": {
        "thumb": [0, 5, 10, 15, 20],
        "index": [1, 6, 21],
        "middle": [2, 7, 22],
        "ring": [3, 8, 23],
        "little": [4, 9, 24],
    },
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_degrees(a, b, c) -> float:
    ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    cb = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    norm_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    norm_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)
    if norm_ab <= 1e-6 or norm_cb <= 1e-6:
        return 180.0
    cosine = clamp(dot / (norm_ab * norm_cb), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def angle_to_curl(angle: float) -> float:
    # Straight fingers are close to 180 deg; curled fingers are usually 60-100 deg.
    return clamp((170.0 - angle) / 100.0, 0.0, 1.0)


def get_finger_curls(hand_landmarks) -> Dict[str, float]:
    lm = hand_landmarks.landmark
    curls = {}

    for finger, ids in FINGER_LANDMARKS.items():
        mcp, pip, dip, tip = ids
        if finger == "thumb":
            thumb_mcp_angle = angle_degrees(lm[mcp], lm[pip], lm[tip])
            thumb_tip_angle = angle_degrees(lm[pip], lm[dip], lm[tip])
            curls[finger] = clamp((angle_to_curl(thumb_mcp_angle) + angle_to_curl(thumb_tip_angle)) / 2.0, 0.0, 1.0)
        else:
            pip_angle = angle_degrees(lm[mcp], lm[pip], lm[dip])
            dip_angle = angle_degrees(lm[pip], lm[dip], lm[tip])
            curls[finger] = clamp((angle_to_curl(pip_angle) + angle_to_curl(dip_angle)) / 2.0, 0.0, 1.0)

    return curls


def load_hand_setting() -> Dict[str, object]:
    setting = LoadWriteYaml().load_setting_yaml()
    linker_hand = setting["LINKER_HAND"]
    left_exists = bool(linker_hand["LEFT_HAND"]["EXISTS"])
    right_exists = bool(linker_hand["RIGHT_HAND"]["EXISTS"])

    if left_exists:
        selected = linker_hand["LEFT_HAND"]
        hand_type = "left"
    elif right_exists:
        selected = linker_hand["RIGHT_HAND"]
        hand_type = "right"
    else:
        raise RuntimeError("setting.yaml does not enable LEFT_HAND or RIGHT_HAND")

    return {
        "hand_type": hand_type,
        "hand_joint": selected["JOINT"],
        "can": selected["CAN"],
        "modbus": selected["MODBUS"],
    }


def expected_joint_count(hand_joint: str) -> int:
    joint = hand_joint.upper()
    if joint in ("O6", "L6"):
        return 6
    if joint == "L7":
        return 7
    if joint == "L10":
        return 10
    if joint in ("L20", "G20"):
        return 20
    if joint in ("L21", "L25"):
        return 25
    raise RuntimeError("Unsupported hand joint type: {}".format(hand_joint))


def choose_open_and_fist_poses(hand_joint: str) -> Tuple[List[int], List[int]]:
    config = _HAND_CONFIGS[hand_joint]
    joint_count = expected_joint_count(hand_joint)
    presets = config.preset_actions or {}
    valid_presets = [list(pose) for pose in presets.values() if len(pose) == joint_count]

    if valid_presets:
        open_pose = max(valid_presets, key=sum)
        fist_pose = min(valid_presets, key=sum)
    else:
        open_pose = list(config.init_pos) if len(config.init_pos) == joint_count else [255] * joint_count
        fist_pose = [0] * joint_count

    return open_pose, fist_pose


def pose_from_curls(
    hand_joint: str,
    curls: Dict[str, float],
    open_pose: List[int],
    fist_pose: List[int],
    previous_pose: Optional[List[int]],
    smoothing: float,
) -> List[int]:
    pose = list(open_pose)
    joint = hand_joint.upper()
    mapping = FINGER_JOINT_INDEXES.get(joint)

    if mapping:
        for finger, indexes in mapping.items():
            curl = curls.get(finger, 0.0)
            for index in indexes:
                if 0 <= index < len(pose):
                    pose[index] = round(open_pose[index] + curl * (fist_pose[index] - open_pose[index]))
    else:
        average_curl = sum(curls.values()) / len(curls)
        pose = [round(o + average_curl * (f - o)) for o, f in zip(open_pose, fist_pose)]

    pose = [int(clamp(v, 0, 255)) for v in pose]
    if previous_pose:
        alpha = clamp(smoothing, 0.0, 0.95)
        pose = [
            int(round(alpha * old + (1.0 - alpha) * new))
            for old, new in zip(previous_pose, pose)
        ]
    return pose


def open_camera(camera_index: int):
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    return cap


def draw_status(frame, text: str, line: int = 0, color=(0, 255, 0)) -> None:
    y = 30 + line * 28
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control Linker Hand with camera hand gestures.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index, default: 0")
    parser.add_argument("--send-interval", type=float, default=0.08, help="Minimum seconds between SDK commands")
    parser.add_argument("--smoothing", type=float, default=0.55, help="Pose smoothing factor, 0-0.95")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the camera preview")
    parser.add_argument("--dry-run", action="store_true", help="Run vision only and print poses without controlling hardware")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = None
    cap = None

    try:
        setting = load_hand_setting()
        hand_joint = str(setting["hand_joint"])
        hand_type = str(setting["hand_type"])
        open_pose, fist_pose = choose_open_and_fist_poses(hand_joint)

        print("Hand config: {} {}, CAN={}, MODBUS={}".format(hand_type, hand_joint, setting["can"], setting["modbus"]))
        print("Open pose: {}".format(open_pose))
        print("Fist pose: {}".format(fist_pose))

        if not args.dry_run:
            api = LinkerHandApi(
                hand_joint=hand_joint,
                hand_type=hand_type,
                modbus=setting["modbus"],
                can=setting["can"],
            )
            api.finger_move(open_pose)
            time.sleep(0.3)

        cap = open_camera(args.camera)
        if not cap or not cap.isOpened():
            raise RuntimeError("Cannot open camera index {}".format(args.camera))

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles

        last_pose = list(open_pose)
        last_send_time = 0.0
        last_print_time = 0.0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera frame read failed.")
                    break

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    curls = get_finger_curls(hand_landmarks)
                    pose = pose_from_curls(hand_joint, curls, open_pose, fist_pose, last_pose, args.smoothing)

                    now = time.time()
                    if now - last_send_time >= args.send_interval:
                        if args.dry_run:
                            if now - last_print_time >= 0.5:
                                print("curls={}, pose={}".format({k: round(v, 2) for k, v in curls.items()}, pose))
                                last_print_time = now
                        else:
                            try:
                                api.finger_move(pose)
                            except Exception as exc:
                                print("SDK send failed: {}".format(exc))
                        last_pose = pose
                        last_send_time = now

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    curl_text = " ".join("{}:{:.2f}".format(k[:1], curls[k]) for k in FINGERS)
                    draw_status(frame, "Detected hand | {}".format(curl_text), 0, (0, 255, 0))
                    draw_status(frame, "Pose: {}".format(pose), 1, (0, 255, 255))
                else:
                    draw_status(frame, "No hand detected. Show one hand to the camera.", 0, (0, 200, 255))

                draw_status(frame, "Press q to quit", 2, (255, 255, 255))
                cv2.imshow("Linker Hand Vision Control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print("Error: {}".format(exc))
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if api is not None:
            try:
                api.close_can()
            except Exception as exc:
                print("SDK close failed: {}".format(exc))

    return 0


if __name__ == "__main__":
    sys.exit(main())
