# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from fastapi import Request

from sglang.multimodal_gen.runtime.realtime import (
    RealtimeEventTailReader,
    append_realtime_event,
    read_realtime_state,
    update_realtime_control,
    update_realtime_state,
)

TERMINAL_EVENT_TYPES = frozenset({"completed", "error", "cancelled"})
TERMINAL_SESSION_STATUSES = frozenset({"completed", "cancelled", "failed"})
DEFAULT_REALTIME_SESSION_TTL_SECONDS = 3600


def to_sse_payload(event: dict[str, Any]) -> str:
    seq = event.get("seq")
    event_line = f"id: {seq}\n" if seq is not None else ""
    return f"{event_line}data: {json.dumps(event, ensure_ascii=False)}\n\n"


def ensure_terminal_event(
    *,
    request_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    status = state.get("status", "completed")
    if status == "failed":
        return {
            "type": "error",
            "id": request_id,
            "error": state.get("error", "unknown error"),
        }
    if status == "cancelled":
        return {"type": "cancelled", "id": request_id}
    return {"type": "completed", "id": request_id}


def resolve_realtime_session_ttl_seconds(ttl_seconds: int | None) -> int:
    resolved = int(ttl_seconds or DEFAULT_REALTIME_SESSION_TTL_SECONDS)
    if resolved <= 0:
        raise ValueError("realtime_session_ttl_seconds must be a positive integer.")
    return resolved


def emit_realtime_session_accepted(
    *,
    artifact_dir: str,
    request_id: str,
    kind: str,
    cancel_on_disconnect: bool | None = None,
    expires_at: int | None = None,
) -> None:
    payload: dict[str, Any] = {"kind": kind}
    state_updates: dict[str, Any] = {"status": "queued", "kind": kind}
    if cancel_on_disconnect is not None:
        payload["cancel_on_disconnect"] = bool(cancel_on_disconnect)
        state_updates["cancel_on_disconnect"] = bool(cancel_on_disconnect)
    if expires_at is not None:
        payload["expires_at"] = int(expires_at)
        state_updates["expires_at"] = int(expires_at)

    append_realtime_event(
        artifact_dir=artifact_dir,
        event_type="accepted",
        request_id=request_id,
        payload=payload,
    )
    update_realtime_state(
        artifact_dir=artifact_dir,
        updates=state_updates,
    )


def emit_realtime_terminal_result(
    *,
    artifact_dir: str,
    request_id: str,
    status: str,
    result_payload: dict[str, Any],
) -> None:
    append_realtime_event(
        artifact_dir=artifact_dir,
        event_type=status,
        request_id=request_id,
        payload={"result": result_payload},
    )
    update_realtime_state(
        artifact_dir=artifact_dir,
        updates={"status": status, "response": result_payload},
    )


def emit_realtime_terminal_error(
    *,
    artifact_dir: str,
    request_id: str,
    error_message: str,
) -> None:
    append_realtime_event(
        artifact_dir=artifact_dir,
        event_type="error",
        request_id=request_id,
        payload={"error": error_message},
    )
    update_realtime_state(
        artifact_dir=artifact_dir,
        updates={"status": "failed", "error": error_message},
    )


async def stream_realtime_events(
    *,
    request: Request,
    request_id: str,
    artifact_dir: str,
    producer_task: asyncio.Task | None = None,
    on_disconnect: Callable[[], Awaitable[None]] | None = None,
    since_seq: int = 0,
    stop_on_terminal: bool = True,
    poll_interval_s: float = 0.05,
) -> AsyncGenerator[str, None]:
    reader = RealtimeEventTailReader(
        artifact_dir=artifact_dir,
        since_seq=since_seq,
    )
    emitted_terminal = False

    while True:
        if await request.is_disconnected():
            if on_disconnect is not None:
                await on_disconnect()
            break

        has_new_events = False
        for event in reader.read_new_events():
            has_new_events = True
            if event.get("type") in TERMINAL_EVENT_TYPES:
                emitted_terminal = True
            yield to_sse_payload(event)

        if stop_on_terminal and emitted_terminal:
            break

        if producer_task is not None and producer_task.done():
            for event in reader.read_new_events():
                if event.get("type") in TERMINAL_EVENT_TYPES:
                    emitted_terminal = True
                yield to_sse_payload(event)
            if not emitted_terminal:
                state = read_realtime_state(artifact_dir)
                terminal_event = ensure_terminal_event(
                    request_id=request_id, state=state
                )
                terminal_event["seq"] = reader.next_seq
                reader.next_seq += 1
                yield to_sse_payload(terminal_event)
            break

        if producer_task is None and stop_on_terminal and not has_new_events:
            state = read_realtime_state(artifact_dir)
            if state.get("status") in TERMINAL_SESSION_STATUSES:
                if not emitted_terminal:
                    terminal_event = ensure_terminal_event(
                        request_id=request_id, state=state
                    )
                    terminal_event["seq"] = reader.next_seq
                    reader.next_seq += 1
                    yield to_sse_payload(terminal_event)
                break

        await asyncio.sleep(poll_interval_s)

    yield "data: [DONE]\n\n"


def apply_realtime_control(
    *,
    request_id: str,
    artifact_dir: str,
    control_updates: dict[str, Any],
) -> dict[str, Any]:
    applied = {k: v for k, v in control_updates.items() if v is not None}
    if not applied:
        return {}

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)

    update_realtime_control(artifact_dir=artifact_dir, updates=applied)
    if applied.get("cancel") is True:
        update_realtime_state(
            artifact_dir=artifact_dir,
            updates={"status": "cancelling"},
        )
    return applied
