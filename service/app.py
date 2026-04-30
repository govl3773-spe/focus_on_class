

# cd C:\Projects\focus_on_class
# uv pip install -r service\requirements-service.txt
# uv run uvicorn service.app:app --reload --host 127.0.0.1 --port 8000

from __future__ import annotations

import atexit
import json
import sqlite3
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from torch import nn
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models"
MP_MODEL_DIR = PROJECT_ROOT / "mp_model"
DB_PATH = BASE_DIR / "attention_logs.db"

REALTIME_MODEL_NAME = "efficientnet_b0"
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
FACE_MODEL_PATH = MP_MODEL_DIR / "face_landmarker.task"
FACE_PADDING_RATIO = 0.25

WINDOW_SIZE = 15
SAMPLE_INTERVAL_SEC = 0.2
MIN_FRAMES_FOR_DECISION = 5

ATTENTIVE_FRAME_THRESHOLD = 0.35
ATTENTIVE_MIN_COUNT = 5
LOOKING_AWAY_FRAME_THRESHOLD = 0.55
DROWSY_FRAME_THRESHOLD = 0.50
LOW_ATTENTIVE_AVG_THRESHOLD = 0.15
DROWSY_LOOKINGAWAY_MARGIN_THRESHOLD = 0.15
LOOKINGAWAY_BONUS = 0.08

NO_FACE_ALLOWED_COUNT = 5
UNCERTAIN_ALLOWED_COUNT = 5
RECENT_STATE_PRINT_COUNT = 3

CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10

CLASS_ALIASES = {
    "attentive": "Attentive",
    "lookingaway": "LookingAway",
    "looking_away": "LookingAway",
    "looking away": "LookingAway",
    "drowsy": "Drowsy",
    "noface": "NoFace",
    "no_face": "NoFace",
    "no face": "NoFace",
    "uncertain": "Uncertain",
    "unknown": "Unknown",
}
VALID_STATES = ["Attentive", "LookingAway", "Drowsy"]
STATE_ORDER = ["Attentive", "Drowsy", "LookingAway", "Unknown"]
STATE_LABELS_KO = {
    "Attentive": "집중",
    "Drowsy": "졸음",
    "LookingAway": "시선 이탈",
    "Unknown": "판단 불가",
    "Waiting": "대기",
}
STATE_COLORS = {
    "Attentive": "#2ca02c",
    "Drowsy": "#d62728",
    "LookingAway": "#ff7f0e",
    "Unknown": "#7f7f7f",
}
NON_ATTENTIVE_STATES = {"Drowsy", "LookingAway"}
TIMELINE_STATE_ORDER = ["Unknown", "Drowsy", "LookingAway", "Attentive"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def change_fc(model, class_count: int):
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def change_classifier(model, class_count: int):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


MODEL_LIST = {
    "resnet18": [models.resnet18, change_fc],
    "resnet34": [models.resnet34, change_fc],
    "resnet50": [models.resnet50, change_fc],
    "mobilenet_v3_large": [models.mobilenet_v3_large, change_classifier],
    "efficientnet_b0": [models.efficientnet_b0, change_classifier],
    "efficientnet_b2": [models.efficientnet_b2, change_classifier],
    "efficientnet_b3": [models.efficientnet_b3, change_classifier],
    "efficientnet_v2_s": [models.efficientnet_v2_s, change_classifier],
}


def infer_model_name(path: Path, checkpoint: dict):
    if "model_name" in checkpoint:
        return checkpoint["model_name"]
    return path.stem.removesuffix("_best")


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_class_names(checkpoint: dict):
    if "class_names" in checkpoint:
        return list(checkpoint["class_names"])
    if "class_to_idx" in checkpoint:
        return [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])]
    raise KeyError("checkpoint is missing class_names or class_to_idx")


def get_model_state(checkpoint: dict):
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    raise KeyError("checkpoint is missing model_state or model_state_dict")


def create_model(model_name: str, class_count: int):
    if model_name not in MODEL_LIST:
        raise ValueError(f"unsupported model name: {model_name}")
    model_func, change_last_layer = MODEL_LIST[model_name]
    model = model_func(weights=None)
    return change_last_layer(model, class_count)


def load_model_for_realtime(model_name: str):
    model_path = MODEL_DIR / f"{model_name}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    checkpoint = load_checkpoint(model_path)
    checkpoint_model_name = infer_model_name(model_path, checkpoint)
    class_names = get_class_names(checkpoint)
    image_size = int(checkpoint.get("image_size", 224))

    model = create_model(checkpoint_model_name, len(class_names)).to(device)
    model.load_state_dict(get_model_state(checkpoint))
    model.eval()

    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    transform = make_eval_transform(image_size)
    return model, class_names, idx_to_class, transform, image_size, model_path


def load_realtime_face_landmarker():
    if not FACE_MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe face model not found: {FACE_MODEL_PATH}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def normalize_class_name(name):
    key = str(name).strip().replace("-", "_").lower()
    return CLASS_ALIASES.get(key, str(name).strip())


def normalize_frame_probs(frame_prob):
    normalized = {state: 0.0 for state in VALID_STATES}
    invalid_state = None

    for key, value in dict(frame_prob).items():
        class_name = normalize_class_name(key)
        if class_name in VALID_STATES:
            normalized[class_name] = float(value)
        elif class_name in {"NoFace", "Uncertain", "Unknown"} and float(value) > 0:
            invalid_state = class_name

    return normalized, invalid_state


def decide_state(frame_probs, return_debug=False):
    frame_probs = list(frame_probs)[-WINDOW_SIZE:]

    if len(frame_probs) < MIN_FRAMES_FOR_DECISION:
        debug_info = {
            "attentive_avg": 0.0,
            "lookingaway_avg": 0.0,
            "drowsy_avg": 0.0,
            "attentive_count": 0,
            "lookingaway_count": 0,
            "drowsy_count": 0,
            "argmax_result": "Unknown",
            "final_state": "Attentive",
            "reason": "not enough samples: default Attentive",
            "valid_frame_count": 0,
            "no_face_count": 0,
            "uncertain_count": 0,
        }
        return ("Attentive", debug_info) if return_debug else "Attentive"

    valid_probs = []
    invalid_counter = Counter()
    argmax_votes = Counter()

    for frame_prob in frame_probs:
        normalized, invalid_state = normalize_frame_probs(frame_prob)
        if invalid_state in {"NoFace", "Uncertain", "Unknown"}:
            invalid_counter[invalid_state] += 1
            continue
        valid_probs.append(normalized)
        argmax_votes[max(VALID_STATES, key=lambda state: normalized[state])] += 1

    no_face_count = invalid_counter["NoFace"]
    uncertain_count = invalid_counter["Uncertain"] + invalid_counter["Unknown"]

    if no_face_count > NO_FACE_ALLOWED_COUNT or uncertain_count > UNCERTAIN_ALLOWED_COUNT:
        final_state = "Unknown"
        argmax_result = argmax_votes.most_common(1)[0][0] if argmax_votes else "Unknown"
        debug_info = {
            "attentive_avg": 0.0,
            "lookingaway_avg": 0.0,
            "drowsy_avg": 0.0,
            "attentive_count": 0,
            "lookingaway_count": 0,
            "drowsy_count": 0,
            "argmax_result": argmax_result,
            "final_state": final_state,
            "reason": "too many no-face or uncertain samples",
            "valid_frame_count": len(valid_probs),
            "no_face_count": no_face_count,
            "uncertain_count": uncertain_count,
        }
        return (final_state, debug_info) if return_debug else final_state

    if not valid_probs:
        final_state = "Unknown"
        debug_info = {
            "attentive_avg": 0.0,
            "lookingaway_avg": 0.0,
            "drowsy_avg": 0.0,
            "attentive_count": 0,
            "lookingaway_count": 0,
            "drowsy_count": 0,
            "argmax_result": "Unknown",
            "final_state": final_state,
            "reason": "no valid probability samples",
            "valid_frame_count": 0,
            "no_face_count": no_face_count,
            "uncertain_count": uncertain_count,
        }
        return (final_state, debug_info) if return_debug else final_state

    attentive_values = [prob["Attentive"] for prob in valid_probs]
    lookingaway_values = [prob["LookingAway"] for prob in valid_probs]
    drowsy_values = [prob["Drowsy"] for prob in valid_probs]

    attentive_avg = sum(attentive_values) / len(attentive_values)
    lookingaway_avg = sum(lookingaway_values) / len(lookingaway_values)
    drowsy_avg = sum(drowsy_values) / len(drowsy_values)

    attentive_count = sum(value >= ATTENTIVE_FRAME_THRESHOLD for value in attentive_values)
    lookingaway_count = sum(value >= LOOKING_AWAY_FRAME_THRESHOLD for value in lookingaway_values)
    drowsy_count = sum(value >= DROWSY_FRAME_THRESHOLD for value in drowsy_values)
    lookingaway_bonus_applied = (
        attentive_avg <= LOW_ATTENTIVE_AVG_THRESHOLD
        and abs(drowsy_avg - lookingaway_avg) <= DROWSY_LOOKINGAWAY_MARGIN_THRESHOLD
    )
    adjusted_lookingaway_avg = lookingaway_avg + LOOKINGAWAY_BONUS if lookingaway_bonus_applied else lookingaway_avg

    avg_by_state = {
        "Attentive": attentive_avg,
        "LookingAway": lookingaway_avg,
        "Drowsy": drowsy_avg,
    }
    argmax_result = max(avg_by_state, key=avg_by_state.get)

    if attentive_count >= ATTENTIVE_MIN_COUNT:
        final_state = "Attentive"
        reason = "at least 5 of 15 samples have Attentive probability >= 0.35"
    elif adjusted_lookingaway_avg >= drowsy_avg:
        final_state = "LookingAway"
        if lookingaway_bonus_applied:
            reason = "Low Attentive and Drowsy/LookingAway are close; LookingAway bonus applied"
        else:
            reason = "Attentive is not strong enough; LookingAway average is higher"
    else:
        final_state = "Drowsy"
        reason = "Attentive is not strong enough; Drowsy average is higher"

    debug_info = {
        "attentive_avg": attentive_avg,
        "lookingaway_avg": lookingaway_avg,
        "adjusted_lookingaway_avg": adjusted_lookingaway_avg,
        "drowsy_avg": drowsy_avg,
        "attentive_count": attentive_count,
        "lookingaway_count": lookingaway_count,
        "drowsy_count": drowsy_count,
        "lookingaway_bonus_applied": lookingaway_bonus_applied,
        "lookingaway_bonus": LOOKINGAWAY_BONUS if lookingaway_bonus_applied else 0.0,
        "argmax_result": argmax_result,
        "final_state": final_state,
        "reason": reason,
        "valid_frame_count": len(valid_probs),
        "no_face_count": no_face_count,
        "uncertain_count": uncertain_count,
    }
    return (final_state, debug_info) if return_debug else final_state


def init_db(db_path=DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attention_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                final_state TEXT NOT NULL,
                argmax_result TEXT,
                attentive_avg REAL,
                lookingaway_avg REAL,
                drowsy_avg REAL,
                attentive_count INTEGER,
                lookingaway_count INTEGER,
                drowsy_count INTEGER,
                reason TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS class_sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                total_duration_sec REAL
            )
            """
        )
    return db_path


def create_session(db_path=DB_PATH):
    init_db(db_path)
    session_id = uuid.uuid4().hex
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO class_sessions (session_id, started_at) VALUES (?, ?)",
            (session_id, utc_now_iso()),
        )
    return session_id


def save_attention_log(db_path, session_id, final_result, debug_info):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO attention_logs (
                session_id, timestamp, final_state, argmax_result,
                attentive_avg, lookingaway_avg, drowsy_avg,
                attentive_count, lookingaway_count, drowsy_count, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                utc_now_iso(),
                final_result,
                debug_info.get("argmax_result"),
                debug_info.get("attentive_avg"),
                debug_info.get("lookingaway_avg"),
                debug_info.get("drowsy_avg"),
                debug_info.get("attentive_count"),
                debug_info.get("lookingaway_count"),
                debug_info.get("drowsy_count"),
                debug_info.get("reason"),
            ),
        )


def get_recent_states(db_path, session_id, limit=RECENT_STATE_PRINT_COUNT):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT final_state
            FROM attention_logs
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [row[0] for row in reversed(rows)]


def check_warning(final_state, current_warning_message=None):
    if final_state == "Attentive":
        return None
    if final_state == "Drowsy":
        return "Drowsy state is continuing. Please sit up and refocus."
    if final_state == "LookingAway":
        return "Looking-away state is continuing. Please look back at the screen."
    return current_warning_message


def end_session(db_path, session_id):
    init_db(db_path)
    ended_at = utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT started_at FROM class_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        total_duration_sec = None
        if row:
            started_at = datetime.fromisoformat(row[0])
            ended_dt = datetime.fromisoformat(ended_at)
            total_duration_sec = max(0.0, (ended_dt - started_at).total_seconds())
        conn.execute(
            "UPDATE class_sessions SET ended_at = ?, total_duration_sec = ? WHERE session_id = ?",
            (ended_at, total_duration_sec, session_id),
        )
    return total_duration_sec


def realtime_face_landmarks_to_bbox(face_landmarks, frame_width: int, frame_height: int):
    xs = [int(lm.x * frame_width) for lm in face_landmarks]
    ys = [int(lm.y * frame_height) for lm in face_landmarks]
    return max(0, min(xs)), max(0, min(ys)), min(frame_width - 1, max(xs)), min(frame_height - 1, max(ys))


def crop_face_for_realtime(frame_bgr, face_box, output_size: int, padding_ratio: float = FACE_PADDING_RATIO):
    if face_box is None:
        return None

    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = face_box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width - 1, x2 + pad_x)
    crop_y2 = min(height - 1, y2 + pad_y)

    crop = frame_bgr[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def predict_realtime_crop(face_crop_bgr, model, transform):
    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_crop_rgb)
    inputs = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        return torch.softmax(outputs, dim=1).squeeze(0).cpu()


def draw_realtime_overlay(frame_bgr, face_box, state, state_prob, probs, fps, debug_info=None, warning_message=None):
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    lines = [
        f"Model: {REALTIME_MODEL_NAME}",
        f"FPS: {fps:.1f}",
        f"State: {state} ({state_prob:.2f})",
    ]
    for class_name in ["Attentive", "Drowsy", "LookingAway"]:
        lines.append(f"{class_name}: {float(probs.get(class_name, 0.0)):.2f}")
    if debug_info:
        lines.append(f"Argmax: {debug_info.get('argmax_result')}")
    if warning_message:
        lines.append(f"WARNING: {warning_message}")

    y = 32
    for line in lines:
        if line.startswith("WARNING:"):
            color = (0, 0, 255)
        elif line.startswith("State:"):
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)
        cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 28
    return frame_bgr


def encode_jpeg(frame_bgr) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return buffer.tobytes()


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes:02d}:{remain:04.1f}"


def load_session_logs_for_report(db_path=DB_PATH, target_session_id: str | None = None):
    db_path = Path(db_path)
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if target_session_id is None:
            row = conn.execute(
                """
                SELECT session_id
                FROM attention_logs
                GROUP BY session_id
                ORDER BY MAX(id) DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None, None, []
            target_session_id = row["session_id"]

        session = conn.execute(
            "SELECT * FROM class_sessions WHERE session_id = ?",
            (target_session_id,),
        ).fetchone()
        logs = conn.execute(
            """
            SELECT *
            FROM attention_logs
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (target_session_id,),
        ).fetchall()
    return target_session_id, session, logs


def build_state_runs(states: list[str], window_duration_sec: float) -> list[dict[str, Any]]:
    runs = []
    if not states:
        return runs

    start_index = 0
    current_state = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state == current_state:
            continue
        runs.append(
            {
                "state": current_state,
                "label": STATE_LABELS_KO.get(current_state, current_state),
                "start_index": start_index,
                "end_index": index - 1,
                "count": index - start_index,
                "start_sec": start_index * window_duration_sec,
                "end_sec": index * window_duration_sec,
                "time_range": f"{format_time(start_index * window_duration_sec)}-{format_time(index * window_duration_sec)}",
                "duration_sec": (index - start_index) * window_duration_sec,
            }
        )
        start_index = index
        current_state = state

    end_index = len(states)
    runs.append(
        {
            "state": current_state,
            "label": STATE_LABELS_KO.get(current_state, current_state),
            "start_index": start_index,
            "end_index": end_index - 1,
            "count": end_index - start_index,
            "start_sec": start_index * window_duration_sec,
            "end_sec": end_index * window_duration_sec,
            "time_range": f"{format_time(start_index * window_duration_sec)}-{format_time(end_index * window_duration_sec)}",
            "duration_sec": (end_index - start_index) * window_duration_sec,
        }
    )
    return runs


def build_session_analysis(session_id: str | None):
    init_db(DB_PATH)
    target_session_id, session, logs = load_session_logs_for_report(DB_PATH, session_id)
    if target_session_id is None:
        return None

    if not logs:
        return {
            "session_id": target_session_id,
            "session": dict(session) if session else None,
            "total_windows": 0,
            "summary": [],
            "runs": [],
            "events": [],
            "series": [],
            "longest_by_state": {},
            "chart_data": {},
        }

    states = [row["final_state"] for row in logs]
    counts = Counter(states)
    window_duration_sec = WINDOW_SIZE * SAMPLE_INTERVAL_SEC
    total_windows = len(states)
    total_sec = total_windows * window_duration_sec
    unknown_sec = counts.get("Unknown", 0) * window_duration_sec
    known_sec = max(0.0, total_sec - unknown_sec)
    attentive_sec = counts.get("Attentive", 0) * window_duration_sec
    drowsy_sec = counts.get("Drowsy", 0) * window_duration_sec
    lookingaway_sec = counts.get("LookingAway", 0) * window_duration_sec
    focus_ratio = attentive_sec / known_sec * 100.0 if known_sec else 0.0
    focus_score = max(
        0.0,
        min(
            100.0,
            focus_ratio
            - (drowsy_sec / known_sec * 120.0 if known_sec else 0.0)
            - (lookingaway_sec / known_sec * 80.0 if known_sec else 0.0),
        ),
    )
    transition_count = sum(1 for prev, curr in zip(states, states[1:]) if prev != curr)
    runs = build_state_runs(states, window_duration_sec)
    events = [
        {"index": index, **run}
        for index, run in enumerate((run for run in runs if run["state"] in NON_ATTENTIVE_STATES), start=1)
    ]
    longest_by_state = {
        state: max(
            (run for run in runs if run["state"] == state),
            key=lambda run: run["duration_sec"],
            default=None,
        )
        for state in STATE_ORDER
    }

    summary = []
    for state in STATE_ORDER:
        count = counts.get(state, 0)
        duration_sec = count * window_duration_sec
        summary.append(
            {
                "state": state,
                "label": STATE_LABELS_KO.get(state, state),
                "count": count,
                "duration_sec": duration_sec,
                "duration_label": f"{duration_sec:.1f}초",
                "ratio": count / total_windows * 100.0 if total_windows else 0.0,
                "color": STATE_COLORS.get(state),
            }
        )

    series = [
        {
            "index": idx,
            "elapsed_sec": idx * window_duration_sec,
            "state": row["final_state"],
            "attentive_avg": row["attentive_avg"] or 0.0,
            "drowsy_avg": row["drowsy_avg"] or 0.0,
            "lookingaway_avg": row["lookingaway_avg"] or 0.0,
        }
        for idx, row in enumerate(logs)
    ]
    timeline_state_to_y = {state: idx for idx, state in enumerate(TIMELINE_STATE_ORDER)}
    timeline = [
        {
            "elapsed_sec": idx * window_duration_sec,
            "state": state,
            "label": STATE_LABELS_KO.get(state, state),
            "y": timeline_state_to_y.get(state, timeline_state_to_y["Unknown"]),
            "color": STATE_COLORS.get(state, STATE_COLORS["Unknown"]),
        }
        for idx, state in enumerate(states)
    ]
    duration_bars = [
        {
            "state": item["state"],
            "label": item["label"],
            "duration_sec": item["duration_sec"],
            "color": item["color"],
        }
        for item in summary
        if item["count"] > 0
    ]
    probability_lines = [
        {
            "elapsed_sec": row["elapsed_sec"],
            "Attentive": row["attentive_avg"],
            "Drowsy": row["drowsy_avg"],
            "LookingAway": row["lookingaway_avg"],
        }
        for row in series
    ]
    longest_summary = {
        state: (
            {
                **run,
                "duration_label": f"{run['duration_sec']:.1f}초",
            }
            if run is not None
            else {
                "state": state,
                "label": STATE_LABELS_KO.get(state, state),
                "duration_sec": 0.0,
                "duration_label": "0.0초",
                "time_range": "-",
                "count": 0,
            }
        )
        for state, run in longest_by_state.items()
    }

    return {
        "session_id": target_session_id,
        "session": dict(session) if session else None,
        "total_windows": total_windows,
        "window_duration_sec": window_duration_sec,
        "total_sec": total_sec,
        "total_analyzed_sec": total_sec,
        "total_analyzed_label": f"{total_sec:.1f}초",
        "known_sec": known_sec,
        "unknown_sec": unknown_sec,
        "attentive_sec": attentive_sec,
        "drowsy_sec": drowsy_sec,
        "lookingaway_sec": lookingaway_sec,
        "focus_ratio": focus_ratio,
        "focus_score": focus_score,
        "transition_count": transition_count,
        "summary": summary,
        "runs": runs,
        "longest_by_state": longest_summary,
        "events": events,
        "series": series,
        "chart_data": {
            "timeline": timeline,
            "timeline_state_order": [
                {"state": state, "label": STATE_LABELS_KO[state], "y": idx}
                for idx, state in enumerate(TIMELINE_STATE_ORDER)
            ],
            "duration_bars": duration_bars,
            "probability_lines": probability_lines,
            "score_bars": [
                {"label": "집중 비율", "value": focus_ratio, "color": STATE_COLORS["Attentive"]},
                {"label": "가중 집중 점수", "value": focus_score, "color": "#1f77b4"},
            ],
        },
    }


def build_summary(session_id: str | None):
    analysis = build_session_analysis(session_id)
    if analysis is None:
        return None
    return {
        key: analysis[key]
        for key in [
            "session_id",
            "session",
            "total_windows",
            "window_duration_sec",
            "total_sec",
            "total_analyzed_sec",
            "focus_ratio",
            "focus_score",
            "transition_count",
            "summary",
            "events",
            "series",
        ]
        if key in analysis
    }


def build_realtime_focus_ratio(session_id: str | None) -> float:
    if session_id is None:
        return 0.0

    init_db(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT final_state, COUNT(*) AS state_count
            FROM attention_logs
            WHERE session_id = ?
            GROUP BY final_state
            """,
            (session_id,),
        ).fetchall()

    counts = {row[0]: int(row[1]) for row in rows}
    known_total = sum(count for state, count in counts.items() if state != "Unknown")
    if known_total == 0:
        return 0.0
    return counts.get("Attentive", 0) / known_total * 100.0


def render_analysis_plot_png(session_id: str | None) -> bytes:
    analysis = build_session_analysis(session_id)
    if analysis is None:
        raise ValueError("No saved session found in DB.")
    if not analysis.get("series"):
        raise ValueError(f"No attention_logs rows found for session: {analysis.get('session_id')}.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    chart_data = analysis["chart_data"]
    timeline = chart_data["timeline"]
    duration_bars = chart_data["duration_bars"]
    probability_lines = chart_data["probability_lines"]
    timeline_order = chart_data["timeline_state_order"]

    timeline_x = [item["elapsed_sec"] for item in timeline]
    timeline_y = [item["y"] for item in timeline]
    timeline_colors = [item["color"] for item in timeline]
    total_sec = max(float(analysis.get("total_analyzed_sec") or 0.0), timeline_x[-1] if timeline_x else 0.0)
    timeline_step_x = [*timeline_x, total_sec]
    timeline_step_y = [*timeline_y, timeline_y[-1]]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.75])
    ax_timeline = fig.add_subplot(grid[0, 0])
    ax_probability = fig.add_subplot(grid[1, 0], sharex=ax_timeline)
    ax_duration = fig.add_subplot(grid[2, 0])

    ax_timeline.step(timeline_step_x, timeline_step_y, where="post", color="#1f77b4", linewidth=2)
    ax_timeline.scatter(timeline_x, timeline_y, c=timeline_colors, s=30, zorder=3)
    ax_timeline.set_title("상태 변화 타임라인")
    ax_timeline.set_yticks([item["y"] for item in timeline_order])
    ax_timeline.set_yticklabels([item["label"] for item in timeline_order])
    ax_timeline.grid(True, axis="x", alpha=0.25)
    ax_timeline.set_xlim(0, total_sec)
    ax_timeline.tick_params(labelbottom=False)

    prob_x = [item["elapsed_sec"] for item in probability_lines]
    prob_step_x = [*prob_x, total_sec]
    ax_probability.step(prob_step_x, [*[item["Attentive"] for item in probability_lines], probability_lines[-1]["Attentive"]], where="post", label="집중", color=STATE_COLORS["Attentive"], linewidth=2)
    ax_probability.step(prob_step_x, [*[item["Drowsy"] for item in probability_lines], probability_lines[-1]["Drowsy"]], where="post", label="졸음", color=STATE_COLORS["Drowsy"], linewidth=2)
    ax_probability.step(prob_step_x, [*[item["LookingAway"] for item in probability_lines], probability_lines[-1]["LookingAway"]], where="post", label="시선 이탈", color=STATE_COLORS["LookingAway"], linewidth=2)
    ax_probability.set_title("구간별 평균 확률")
    ax_probability.set_xlabel("경과 시간(초)")
    ax_probability.set_ylabel("확률")
    ax_probability.set_ylim(0, 1)
    ax_probability.set_xlim(0, total_sec)
    ax_probability.grid(True, alpha=0.25)
    ax_probability.legend(loc="upper right", ncols=3)

    bar_labels = [item["label"] for item in duration_bars]
    bar_durations = [item["duration_sec"] for item in duration_bars]
    bar_colors = [item["color"] for item in duration_bars]
    ax_duration.bar(bar_labels, bar_durations, color=bar_colors)
    ax_duration.set_title("상태별 누적 시간")
    ax_duration.set_ylabel("시간(초)")
    ax_duration.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(bar_durations):
        ax_duration.text(index, value, f"{value:.1f}s", ha="center", va="bottom")

    title = f"Focus On Class 분석 - {analysis['session_id']}"
    fig.suptitle(title, fontsize=14)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


def render_analysis_summary_png(session_id: str | None) -> bytes:
    analysis = build_session_analysis(session_id)
    if analysis is None:
        raise ValueError("No saved session found in DB.")
    if not analysis.get("series"):
        raise ValueError(f"No attention_logs rows found for session: {analysis.get('session_id')}.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 2.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 1)
    ax_cards = fig.add_subplot(grid[0, :])

    ax_cards.axis("off")
    card_items = [
        ("총 분석 시간", analysis["total_analyzed_label"], "#111827"),
        ("집중 비율", f"{analysis['focus_ratio']:.1f}%", STATE_COLORS["Attentive"]),
        ("집중 점수", f"{analysis['focus_score']:.1f}/100", "#1f77b4"),
        ("상태 전환", f"{analysis['transition_count']}회", "#6b7280"),
    ]
    for index, (label, value, color) in enumerate(card_items):
        x = 0.02 + index * 0.245
        rect = plt.Rectangle((x, 0.15), 0.22, 0.7, transform=ax_cards.transAxes, color="#f9fafb", ec="#d1d5db")
        ax_cards.add_patch(rect)
        ax_cards.text(x + 0.11, 0.58, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color, transform=ax_cards.transAxes)
        ax_cards.text(x + 0.11, 0.32, label, ha="center", va="center", fontsize=11, color="#374151", transform=ax_cards.transAxes)

    fig.suptitle(f"Focus On Class 요약 - {analysis['session_id']}", fontsize=15)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    return buffer.getvalue()


class FocusService:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera_capture = None
        self.face_landmarker = None
        self.model = None
        self.class_names: list[str] = []
        self.idx_to_class = {}
        self.transform = None
        self.image_size = 224
        self.loaded_model_path = None

        self.session_id: str | None = None
        self.frame_window = deque(maxlen=WINDOW_SIZE)
        self.display_probs = {state: 0.0 for state in VALID_STATES}
        self.display_class = "Waiting"
        self.display_prob = 0.0
        self.latest_debug_info = None
        self.warning_message = None
        self.fps = 0.0
        self.last_jpeg: bytes | None = None
        self.recent_states: list[str] = []
        self.last_error: str | None = None
        self.log_count = 0

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.is_running():
            return
        self.stop_event.clear()
        self.last_error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.thread = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            status = {
                "running": self.is_running(),
                "session_id": self.session_id,
                "state": self.display_class,
                "state_label": STATE_LABELS_KO.get(self.display_class, self.display_class),
                "state_prob": self.display_prob,
                "probs": dict(self.display_probs),
                "fps": self.fps,
                "warning_message": self.warning_message,
                "debug_info": self.latest_debug_info,
                "recent_states": list(self.recent_states),
                "log_count": self.log_count,
                "sample_count": len(self.frame_window),
                "window_size": WINDOW_SIZE,
                "sample_interval_sec": SAMPLE_INTERVAL_SEC,
                "db_path": str(DB_PATH),
                "last_error": self.last_error,
            }
        status["focus_ratio"] = build_realtime_focus_ratio(status["session_id"])
        return status

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.last_jpeg

    def _close_face_landmarker(self):
        if self.face_landmarker is not None:
            close = getattr(self.face_landmarker, "close", None)
            if callable(close):
                close()
            self.face_landmarker = None

    def _ensure_resources(self):
        if self.model is None:
            (
                self.model,
                self.class_names,
                self.idx_to_class,
                self.transform,
                self.image_size,
                self.loaded_model_path,
            ) = load_model_for_realtime(REALTIME_MODEL_NAME)
            self.display_probs = {class_name: 0.0 for class_name in self.class_names}

    def _run(self):
        try:
            self._ensure_resources()
            self._close_face_landmarker()
            self.face_landmarker = load_realtime_face_landmarker()
            init_db(DB_PATH)
            self.session_id = create_session(DB_PATH)
            self.frame_window.clear()
            self.display_class = "Waiting"
            self.display_prob = 0.0
            self.latest_debug_info = None
            self.warning_message = None
            self.recent_states = []
            self.log_count = 0

            self.camera_capture = cv2.VideoCapture(CAMERA_INDEX)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not self.camera_capture.isOpened():
                raise RuntimeError("Camera open failed. Check CAMERA_INDEX and camera permission.")

            start_time = time.perf_counter()
            last_frame_time = time.perf_counter()
            last_sample_time = start_time - SAMPLE_INTERVAL_SEC

            while not self.stop_event.is_set():
                is_readable, frame_bgr = self.camera_capture.read()
                if not is_readable:
                    raise RuntimeError("Frame read failed.")

                frame_bgr = cv2.flip(frame_bgr, 1)
                current_time = time.perf_counter()
                timestamp_ms = int((current_time - start_time) * 1000)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
                face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
                face_landmarks_list = getattr(face_result, "face_landmarks", [])
                face_box = None
                sampled_probs = {"NoFace": 1.0}

                if face_landmarks_list:
                    face_box = realtime_face_landmarks_to_bbox(
                        face_landmarks_list[0], frame_bgr.shape[1], frame_bgr.shape[0]
                    )
                    face_crop_bgr = crop_face_for_realtime(frame_bgr, face_box, self.image_size)
                    if face_crop_bgr is not None:
                        probs = predict_realtime_crop(face_crop_bgr, self.model, self.transform)
                        current_frame_probs = {
                            self.idx_to_class[idx]: float(probs[idx].item())
                            for idx in range(len(self.idx_to_class))
                        }
                        self.display_probs = current_frame_probs
                        sampled_probs = current_frame_probs
                    else:
                        self.display_probs = {class_name: 0.0 for class_name in self.class_names}
                        sampled_probs = {"Uncertain": 1.0}
                else:
                    self.display_probs = {class_name: 0.0 for class_name in self.class_names}
                    sampled_probs = {"NoFace": 1.0}

                should_sample = current_time - last_sample_time >= SAMPLE_INTERVAL_SEC
                if should_sample:
                    self.frame_window.append(sampled_probs)
                    last_sample_time = current_time

                should_decide = should_sample and len(self.frame_window) >= WINDOW_SIZE
                if should_decide:
                    final_state, debug_info = decide_state(self.frame_window, return_debug=True)
                    save_attention_log(DB_PATH, self.session_id, final_state, debug_info)
                    self.recent_states = get_recent_states(DB_PATH, self.session_id)
                    self.warning_message = check_warning(final_state, self.warning_message)
                    self.display_class = final_state
                    self.display_prob = debug_info.get(f"{final_state.lower()}_avg", 0.0)
                    self.latest_debug_info = debug_info
                    self.frame_window.clear()
                    self.log_count += 1

                self.fps = 1.0 / max(1e-6, current_time - last_frame_time)
                last_frame_time = current_time
                display_frame = draw_realtime_overlay(
                    frame_bgr.copy(),
                    face_box,
                    self.display_class,
                    self.display_prob,
                    self.display_probs,
                    self.fps,
                    self.latest_debug_info,
                    self.warning_message,
                )
                jpeg = encode_jpeg(display_frame)
                with self.lock:
                    if jpeg is not None:
                        self.last_jpeg = jpeg
                    self.last_error = None
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
        finally:
            if self.camera_capture is not None:
                self.camera_capture.release()
                self.camera_capture = None
            self._close_face_landmarker()
            if self.session_id is not None:
                end_session(DB_PATH, self.session_id)


focus_service = FocusService()
app = FastAPI(title="Focus On Class Service")


HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Focus On Class Service</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; font-size: 14px; background: #f5f6f8; color: #1f2937; }
    header { padding: 10px 16px; background: #111827; color: white; display: flex; justify-content: space-between; align-items: center; }
    main { padding: 12px; display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 12px; align-items: start; }
    h3 { margin: 0 0 10px; font-size: 16px; }
    h4 { margin: 12px 0 8px; font-size: 13px; }
    .panel { background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; }
    .video-wrap { display: grid; grid-template-columns: minmax(0, 1fr) 56px; gap: 10px; align-items: stretch; }
    .score-gauge { min-height: 300px; display: grid; grid-template-rows: auto 1fr auto; gap: 6px; justify-items: center; }
    .score-gauge-title { font-size: 11px; font-weight: 700; color: #374151; text-align: center; }
    .score-track { position: relative; width: 22px; height: 100%; min-height: 230px; border-radius: 999px; background: #e5e7eb; overflow: hidden; border: 1px solid #d1d5db; }
    .score-fill { position: absolute; left: 0; right: 0; bottom: 0; height: 0%; background: linear-gradient(180deg, #22c55e, #2563eb); transition: height 0.3s ease; }
    .score-value { font-size: 15px; font-weight: 800; color: #111827; }
    .video { width: 100%; background: #111827; border-radius: 6px; min-height: 300px; max-height: 58vh; object-fit: contain; }
    button { border: 0; border-radius: 5px; padding: 7px 10px; font-size: 13px; font-weight: 700; cursor: pointer; margin-left: 6px; }
    .start { background: #16a34a; color: white; }
    .stop { background: #dc2626; color: white; }
    .metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #edf0f3; gap: 8px; }
    .metric strong { max-width: 170px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .warning { color: #dc2626; font-weight: 700; min-height: 20px; margin: 8px 0; }
    .bar { height: 6px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 3px 0 8px; }
    .fill { height: 100%; background: #2563eb; }
    .analysis-plot { display: block; width: 100%; max-width: 1120px; margin-top: 10px; border: 1px solid #e5e7eb; border-radius: 6px; background: white; }
    .hidden-json { display: none; }
    pre { white-space: pre-wrap; max-height: 180px; overflow: auto; background: #f9fafb; padding: 8px; border-radius: 6px; font-size: 12px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .video-wrap { grid-template-columns: minmax(0, 1fr) 48px; } }
  </style>
</head>
<body>
  <header>
    <div><strong>Focus On Class</strong> FastAPI MJPEG Service</div>
    <div>
      <button class="start" onclick="startService()">시작</button>
      <button class="stop" onclick="stopService()">종료</button>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="video-wrap">
        <img class="video" src="/video_feed" alt="video stream" />
        <div class="score-gauge">
          <div class="score-gauge-title">집중 비율</div>
          <div class="score-track"><div id="focusScoreFill" class="score-fill"></div></div>
          <div id="focusScoreValue" class="score-value">0%</div>
        </div>
      </div>
    </section>
    <aside class="panel">
      <h3>실시간 상태</h3>
      <div class="metric"><span>실행</span><strong id="running">-</strong></div>
      <div class="metric"><span>세션</span><strong id="session">-</strong></div>
      <div class="metric"><span>상태</span><strong id="state">-</strong></div>
      <div class="metric"><span>FPS</span><strong id="fps">-</strong></div>
      <div class="metric"><span>판단 로그</span><strong id="logs">-</strong></div>
      <p class="warning" id="warning"></p>
      <h4>프레임별 확률</h4>
      <div>집중 <span id="pAttentive">0.00</span><div class="bar"><div id="bAttentive" class="fill"></div></div></div>
      <div>졸음 <span id="pDrowsy">0.00</span><div class="bar"><div id="bDrowsy" class="fill"></div></div></div>
      <div>시선 이탈 <span id="pLookingAway">0.00</span><div class="bar"><div id="bLookingAway" class="fill"></div></div></div>
      <h4>마지막 판단</h4>
      <pre id="debug">-</pre>
    </aside>
    <section class="panel hidden-json" style="grid-column: 1 / -1;">
      <h3>최근 세션 요약</h3>
      <button onclick="loadSummary()">요약 새로고침</button>
      <pre id="summary">-</pre>
    </section>
    <section class="panel" style="grid-column: 1 / -1;">
      <h3>세션 상세 분석</h3>
      <button onclick="loadAnalysis()">분석 새로고침</button>
      <img id="summaryPlot" class="analysis-plot" alt="session summary plot" />
      <img id="analysisPlot" class="analysis-plot" alt="session analysis plot" />
      <pre id="analysis" class="hidden-json">-</pre>
    </section>
  </main>
  <script>
    async function post(url) {
      const res = await fetch(url, {method: 'POST'});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '요청 실패');
      return data;
    }
    function clearAnalysisImages() {
      document.getElementById('analysis').textContent = '-';
      for (const id of ['summaryPlot', 'analysisPlot']) {
        const img = document.getElementById(id);
        img.removeAttribute('src');
        img.style.display = 'none';
      }
    }
    async function startService() {
      try {
        clearAnalysisImages();
        await post('/api/start');
      } catch (error) {
        alert(error.message || '요청 실패');
      }
      await refresh();
    }
    async function stopService() {
      try {
        await post('/api/stop');
      } catch (error) {
        alert(error.message || '요청 실패');
      }
      await refresh();
      await loadSummary();
      await loadAnalysis();
    }
    function setProb(name, value) {
      const v = Number(value || 0);
      document.getElementById('p' + name).textContent = v.toFixed(2);
      document.getElementById('b' + name).style.width = Math.max(0, Math.min(100, v * 100)) + '%';
    }
    function setFocusScore(value) {
      const score = Math.max(0, Math.min(100, Number(value || 0)));
      document.getElementById('focusScoreFill').style.height = score + '%';
      document.getElementById('focusScoreValue').textContent = score.toFixed(0) + '%';
    }
    async function refresh() {
      const data = await (await fetch('/api/status')).json();
      document.getElementById('running').textContent = data.running ? '실행 중' : '정지';
      document.getElementById('session').textContent = data.session_id || '-';
      document.getElementById('state').textContent = data.state_label || data.state || '-';
      document.getElementById('fps').textContent = Number(data.fps || 0).toFixed(1);
      document.getElementById('logs').textContent = data.log_count || 0;
      document.getElementById('warning').textContent = data.warning_message || data.last_error || '';
      setFocusScore(data.focus_ratio);
      setProb('Attentive', data.probs.Attentive);
      setProb('Drowsy', data.probs.Drowsy);
      setProb('LookingAway', data.probs.LookingAway);
      document.getElementById('debug').textContent = JSON.stringify(data.debug_info || {}, null, 2);
    }
    async function loadSummary() {
      const data = await (await fetch('/api/summary')).json();
      document.getElementById('summary').textContent = JSON.stringify(data, null, 2);
    }
    async function loadAnalysis() {
      const data = await (await fetch('/api/analysis')).json();
      document.getElementById('analysis').textContent = JSON.stringify(data, null, 2);
      setFocusScore(data && data.focus_ratio);
      const summaryPlot = document.getElementById('summaryPlot');
      const plot = document.getElementById('analysisPlot');
      if (data && data.series && data.series.length) {
        summaryPlot.style.display = 'block';
        summaryPlot.src = '/api/analysis/summary-plot?t=' + Date.now();
        plot.style.display = 'block';
        plot.src = '/api/analysis/plot?t=' + Date.now();
      } else {
        summaryPlot.removeAttribute('src');
        summaryPlot.style.display = 'none';
        plot.removeAttribute('src');
        plot.style.display = 'none';
      }
    }
    setInterval(refresh, 700);
    refresh();
    clearAnalysisImages();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/start")
def start_service():
    try:
        focus_service.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "status": focus_service.status()}


@app.post("/api/stop")
def stop_service():
    focus_service.stop()
    return {"ok": True, "status": focus_service.status()}


@app.get("/api/status")
def get_status():
    return JSONResponse(focus_service.status())


@app.get("/api/summary")
def get_summary(session_id: str | None = None):
    target_session_id = session_id or focus_service.session_id
    return JSONResponse(build_summary(target_session_id))


@app.get("/api/analysis")
def get_analysis(session_id: str | None = None):
    target_session_id = session_id or focus_service.session_id
    return JSONResponse(build_session_analysis(target_session_id))


@app.get("/api/analysis/plot")
def get_analysis_plot(session_id: str | None = None):
    target_session_id = session_id or focus_service.session_id
    try:
        image = render_analysis_plot_png(target_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=image, media_type="image/png")


@app.get("/api/analysis/summary-plot")
def get_analysis_summary_plot(session_id: str | None = None):
    target_session_id = session_id or focus_service.session_id
    try:
        image = render_analysis_summary_png(target_session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=image, media_type="image/png")


def mjpeg_generator():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    try:
        while True:
            frame = focus_service.get_jpeg()
            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Press Start", (210, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                frame = encode_jpeg(blank)
            if frame is not None:
                yield boundary + frame + b"\r\n"
            time.sleep(0.05)
    except (GeneratorExit, BrokenPipeError, ConnectionError):
        return


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@atexit.register
def _cleanup():
    focus_service.stop()
