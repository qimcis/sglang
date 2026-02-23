# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

REALTIME_SUBDIR = "realtime"
EVENTS_FILE = "events.jsonl"
STATE_FILE = "state.json"
CONTROL_FILE = "control.json"


def get_realtime_root(output_path: str | None) -> str:
    base_dir = output_path or "outputs"
    return os.path.join(base_dir, REALTIME_SUBDIR)


def get_realtime_artifact_dir(request_id: str, output_path: str | None) -> str:
    return os.path.join(get_realtime_root(output_path), request_id)


def _state_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, STATE_FILE)


def _events_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, EVENTS_FILE)


def _control_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, CONTROL_FILE)


def _safe_json_load(path: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if not os.path.exists(path):
        return dict(fallback)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("Failed to parse JSON file: %s", path, exc_info=True)
    return dict(fallback)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def ensure_realtime_artifacts(request_id: str, output_path: str | None) -> str:
    artifact_dir = get_realtime_artifact_dir(
        request_id=request_id, output_path=output_path
    )
    os.makedirs(artifact_dir, exist_ok=True)

    control_path = _control_path(artifact_dir)
    if not os.path.exists(control_path):
        _atomic_write_json(control_path, {"cancel": False, "updated_at": time.time()})

    state_path = _state_path(artifact_dir)
    if not os.path.exists(state_path):
        _atomic_write_json(
            state_path,
            {
                "request_id": request_id,
                "status": "created",
                "created_at": time.time(),
            },
        )

    events_path = _events_path(artifact_dir)
    if not os.path.exists(events_path):
        open(events_path, "a", encoding="utf-8").close()

    return artifact_dir


def append_realtime_event(
    artifact_dir: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": event_type,
        "ts": time.time(),
    }
    if request_id is not None:
        event["id"] = request_id
    if payload:
        event.update(payload)

    line = json.dumps(event, ensure_ascii=False)
    with open(_events_path(artifact_dir), "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")

    return event


def read_realtime_state(artifact_dir: str) -> dict[str, Any]:
    return _safe_json_load(_state_path(artifact_dir), fallback={})


def update_realtime_state(artifact_dir: str, updates: dict[str, Any]) -> dict[str, Any]:
    state = read_realtime_state(artifact_dir)
    state.update(updates)
    state["updated_at"] = time.time()
    _atomic_write_json(_state_path(artifact_dir), state)
    return state


def update_realtime_control(
    artifact_dir: str, updates: dict[str, Any]
) -> dict[str, Any]:
    control = _safe_json_load(_control_path(artifact_dir), fallback={})
    control.update(updates)
    control["updated_at"] = time.time()
    _atomic_write_json(_control_path(artifact_dir), control)
    return control


def read_realtime_control_if_changed(
    artifact_dir: str, last_mtime: float | None
) -> tuple[dict[str, Any] | None, float | None]:
    control_path = _control_path(artifact_dir)
    if not os.path.exists(control_path):
        return None, last_mtime

    try:
        current_mtime = os.path.getmtime(control_path)
    except OSError:
        return None, last_mtime

    control = _safe_json_load(control_path, fallback={})
    updated_at = float(control.get("updated_at", 0.0))
    current_marker = max(current_mtime, updated_at)

    if last_mtime is not None and current_marker <= last_mtime:
        return None, last_mtime

    return control, current_marker


def read_realtime_events_page(
    artifact_dir: str,
    since_seq: int = 0,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], int, bool]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit!r}")

    events_path = _events_path(artifact_dir)
    if not os.path.exists(events_path):
        return [], since_seq + 1, False

    events: list[dict[str, Any]] = []
    has_more = False
    seq = 0
    since_seq = max(0, int(since_seq))

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            seq += 1
            if seq <= since_seq:
                continue

            try:
                parsed = json.loads(line)
            except Exception:
                logger.warning("Failed to parse realtime event line: %s", line[:200])
                continue
            if not isinstance(parsed, dict):
                continue

            event = dict(parsed)
            # Sequence is derived from file order so replay cursors stay deterministic.
            event["seq"] = seq
            events.append(event)

            if len(events) >= limit:
                for remaining in f:
                    if remaining.strip():
                        has_more = True
                        break
                break

    next_seq = (events[-1]["seq"] if events else since_seq) + 1
    return events, next_seq, has_more


@dataclass
class RealtimeEventTailReader:
    artifact_dir: str
    since_seq: int = 0
    offset: int = 0
    next_seq: int = 1
    initialized: bool = False

    def _initialize_offset(self) -> None:
        if self.initialized:
            return

        events_path = _events_path(self.artifact_dir)
        target_seq = max(0, int(self.since_seq))
        if target_seq <= 0 or not os.path.exists(events_path):
            self.initialized = True
            return

        seq = 0
        offset = 0
        with open(events_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line == "":
                    break

                if line.strip():
                    seq += 1
                offset = f.tell()

                if seq >= target_seq:
                    break

        self.offset = offset
        self.next_seq = seq + 1
        self.initialized = True

    def read_new_events(self) -> list[dict[str, Any]]:
        self._initialize_offset()
        events_path = _events_path(self.artifact_dir)
        if not os.path.exists(events_path):
            return []

        with open(events_path, "r", encoding="utf-8") as f:
            f.seek(self.offset)
            lines = f.readlines()
            self.offset = f.tell()

        events: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            seq = self.next_seq
            self.next_seq += 1
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    event = dict(parsed)
                    event["seq"] = seq
                    events.append(event)
            except Exception:
                logger.warning("Failed to parse realtime event line: %s", line[:200])
        return events
