import asyncio
import contextlib
import os
import time
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeControlRequest,
    RealtimeControlResponse,
    RealtimeEventsResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime_utils import (
    TERMINAL_EVENT_TYPES,
    TERMINAL_SESSION_STATUSES,
    apply_realtime_control,
    ensure_terminal_event,
    stream_realtime_events,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import (
    IMAGE_STORE,
    REALTIME_STORE,
    VIDEO_STORE,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime import (
    RealtimeEventTailReader,
    get_realtime_artifact_dir,
    read_realtime_events_page,
    read_realtime_state,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


async def _get_realtime_artifact_dir(
    request_id: str,
    *,
    require_exists: bool = False,
) -> str:
    # realtime streaming jobs
    for store in (REALTIME_STORE, VIDEO_STORE, IMAGE_STORE):
        item = await store.get(request_id)
        if item is None:
            continue
        artifact_dir = item.get("artifact_dir")
        if artifact_dir:
            if require_exists and not os.path.exists(artifact_dir):
                raise HTTPException(
                    status_code=404,
                    detail=f"Realtime session {request_id!r} was not found.",
                )
            return artifact_dir

    server_args = get_global_server_args()
    if server_args is None:
        raise HTTPException(status_code=500, detail="Server args not initialized")
    artifact_dir = get_realtime_artifact_dir(
        request_id=request_id, output_path=server_args.output_path
    )
    if require_exists and not os.path.exists(artifact_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Realtime session {request_id!r} was not found.",
        )
    return artifact_dir


def _read_expires_at(state: dict[str, Any]) -> float | None:
    expires_at = state.get("expires_at")
    if expires_at is None:
        return None
    try:
        return float(expires_at)
    except (TypeError, ValueError):
        return None


def _raise_if_session_expired(request_id: str, state: dict[str, Any]) -> None:
    expires_at = _read_expires_at(state)
    if expires_at is None:
        return
    if time.time() > expires_at:
        raise HTTPException(
            status_code=410,
            detail=f"Realtime session {request_id!r} has expired.",
        )


def _is_session_expired(state: dict[str, Any]) -> bool:
    expires_at = _read_expires_at(state)
    if expires_at is None:
        return False
    return time.time() > expires_at


async def _mark_request_cancelling(request_id: str) -> None:
    await REALTIME_STORE.update_fields(request_id, {"status": "cancelling"})
    await VIDEO_STORE.update_fields(request_id, {"status": "cancelling"})
    await IMAGE_STORE.update_fields(request_id, {"status": "cancelling"})


async def _apply_realtime_control_updates(
    *,
    request_id: str,
    artifact_dir: str,
    control_updates: dict[str, Any],
) -> dict[str, Any]:
    applied = apply_realtime_control(
        request_id=request_id,
        artifact_dir=artifact_dir,
        control_updates=control_updates,
    )

    if applied.get("cancel") is True:
        await _mark_request_cancelling(request_id)

    return applied


def _normalize_realtime_ws_control_updates(
    message: dict[str, Any],
) -> RealtimeControlRequest:
    control_payload = message.get("control")
    if control_payload is None:
        control_payload = message

    if not isinstance(control_payload, dict):
        raise ValueError(
            "Invalid control payload. Expected object with control fields."
        )

    control_fields = ("cancel", "guidance_scale", "guidance_scale_2", "true_cfg_scale")
    updates = {
        k: control_payload.get(k) for k in control_fields if k in control_payload
    }
    return RealtimeControlRequest(**updates)


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None


class DiffusionModelCard(ModelCard):
    """Extended ModelCard with diffusion-specific fields."""

    num_gpus: Optional[int] = None
    task_type: Optional[str] = None
    dit_precision: Optional[str] = None
    vae_precision: Optional[str] = None
    pipeline_name: Optional[str] = None
    pipeline_class: Optional[str] = None


async def _handle_lora_request(req: Any, success_msg: str, failure_msg: str):
    try:
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return {"status": "ok", "message": success_msg}
        else:
            error_msg = output.error
            raise HTTPException(status_code=500, detail=f"{failure_msg}: {error_msg}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during '{failure_msg}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_lora")
async def set_lora(
    lora_nickname: Union[str, List[str]] = Body(..., embed=True),
    lora_path: Optional[Union[str, List[Optional[str]]]] = Body(None, embed=True),
    target: Union[str, List[str]] = Body("all", embed=True),
    strength: Union[float, List[float]] = Body(1.0, embed=True),
):
    """
    Set LoRA adapter(s) for the specified transformer(s).
    Supports both single LoRA (backward compatible) and multiple LoRA adapters.

    Args:
        lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
        lora_path: Path(s) to the LoRA adapter(s) (local path or HF repo id).
            Can be a string, None, or a list of strings/None. Must match the length of lora_nickname.
        target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
            If a list, must match the length of lora_nickname. Valid values:
            - "all": Apply to all transformers (default)
            - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
            - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
            - "critic": Apply only to the critic model
        strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
            If a list, must match the length of lora_nickname. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = SetLoraReq(
        lora_nickname=lora_nickname,
        lora_path=lora_path,
        target=target,
        strength=strength,
    )
    nickname_str, target_str, strength_str = format_lora_message(
        lora_nickname, target, strength
    )

    return await _handle_lora_request(
        req,
        f"Successfully set LoRA adapter(s): {nickname_str} (target: {target_str}, strength: {strength_str})",
        "Failed to set LoRA adapter",
    )


@router.post("/merge_lora_weights")
async def merge_lora_weights(
    target: str = Body("all", embed=True),
    strength: float = Body(1.0, embed=True),
):
    """
    Merge LoRA weights into the base model.

    Args:
        target: Which transformer(s) to merge. One of "all", "transformer",
                "transformer_2", "critic".
        strength: LoRA strength for merge, default 1.0. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = MergeLoraWeightsReq(target=target, strength=strength)
    return await _handle_lora_request(
        req,
        f"Successfully merged LoRA weights (target: {target}, strength: {strength})",
        "Failed to merge LoRA weights",
    )


@router.post("/unmerge_lora_weights")
async def unmerge_lora_weights(
    target: str = Body("all", embed=True),
):
    """
    Unmerge LoRA weights from the base model.

    Args:
        target: Which transformer(s) to unmerge. One of "all", "transformer",
                "transformer_2", "critic".
    """
    req = UnmergeLoraWeightsReq(target=target)
    return await _handle_lora_request(
        req,
        f"Successfully unmerged LoRA weights (target: {target})",
        "Failed to unmerge LoRA weights",
    )


@router.get("/model_info")
async def model_info():
    """Get the model information."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    result = {
        "model_path": server_args.model_path,
    }
    return result


@router.get("/list_loras")
async def list_loras():
    """List loaded LoRA adapters and current application status per module."""
    try:
        req = ListLorasReq()
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return output.output or {}
        else:
            raise HTTPException(status_code=500, detail=output.error)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during 'list_loras': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime/{request_id}/control", response_model=RealtimeControlResponse)
async def realtime_control(request_id: str, request: RealtimeControlRequest):
    artifact_dir = await _get_realtime_artifact_dir(request_id, require_exists=True)
    _raise_if_session_expired(request_id, read_realtime_state(artifact_dir))
    applied = await _apply_realtime_control_updates(
        request_id=request_id,
        artifact_dir=artifact_dir,
        control_updates=request.model_dump(exclude_none=True),
    )

    if not applied:
        return RealtimeControlResponse(
            id=request_id,
            status="noop",
            applied={},
        )

    return RealtimeControlResponse(
        id=request_id,
        status="ok",
        applied=applied,
    )


@router.get("/realtime/{request_id}/events", response_model=RealtimeEventsResponse)
async def realtime_events(
    request_id: str,
    raw_request: Request,
    since_seq: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
    stream: bool = Query(False),
):
    artifact_dir = await _get_realtime_artifact_dir(request_id, require_exists=True)
    state = read_realtime_state(artifact_dir)
    _raise_if_session_expired(request_id, state)

    if stream:
        return StreamingResponse(
            stream_realtime_events(
                request=raw_request,
                request_id=request_id,
                artifact_dir=artifact_dir,
                since_seq=since_seq,
                stop_on_terminal=True,
            ),
            media_type="text/event-stream",
        )

    events, next_seq, has_more = read_realtime_events_page(
        artifact_dir=artifact_dir,
        since_seq=since_seq,
        limit=limit,
    )
    expires_at = _read_expires_at(state)
    return RealtimeEventsResponse(
        id=request_id,
        status=str(state.get("status", "unknown")),
        since_seq=since_seq,
        next_seq=next_seq,
        has_more=has_more,
        events=events,
        latest_preview=state.get("latest_preview"),
        result=state.get("response"),
        expires_at=int(expires_at) if expires_at is not None else None,
    )


@router.websocket("/realtime/{request_id}/ws")
async def realtime_ws(
    websocket: WebSocket,
    request_id: str,
):
    await websocket.accept()

    try:
        artifact_dir = await _get_realtime_artifact_dir(request_id, require_exists=True)
    except HTTPException as exc:
        await websocket.send_json(
            {"type": "error", "id": request_id, "error": str(exc.detail)}
        )
        await websocket.close(code=1008)
        return

    try:
        since_seq = max(0, int(websocket.query_params.get("since_seq", "0")))
    except ValueError:
        since_seq = 0

    stop_on_terminal = websocket.query_params.get(
        "stop_on_terminal", "true"
    ).lower() not in {"0", "false", "no"}

    try:
        poll_interval_s = float(websocket.query_params.get("poll_interval_s", "0.05"))
    except ValueError:
        poll_interval_s = 0.05
    poll_interval_s = max(0.01, poll_interval_s)

    reader = RealtimeEventTailReader(
        artifact_dir=artifact_dir,
        since_seq=since_seq,
    )
    emitted_terminal = False

    await websocket.send_json(
        {
            "type": "ready",
            "id": request_id,
            "since_seq": since_seq,
            "stop_on_terminal": stop_on_terminal,
        }
    )

    while True:
        try:
            state = read_realtime_state(artifact_dir)
            if _is_session_expired(state):
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": request_id,
                        "error": f"Realtime session {request_id!r} has expired.",
                    }
                )
                await websocket.close(code=1008)
                return

            has_new_events = False
            for event in reader.read_new_events():
                has_new_events = True
                if event.get("type") in TERMINAL_EVENT_TYPES:
                    emitted_terminal = True
                await websocket.send_json(
                    {"type": "event", "id": request_id, "event": event}
                )

            if stop_on_terminal and emitted_terminal:
                await websocket.send_json({"type": "done", "id": request_id})
                await websocket.close(code=1000)
                return

            if stop_on_terminal and not has_new_events:
                state = read_realtime_state(artifact_dir)
                if state.get("status") in TERMINAL_SESSION_STATUSES:
                    if not emitted_terminal:
                        terminal_event = ensure_terminal_event(
                            request_id=request_id,
                            state=state,
                        )
                        terminal_event["seq"] = reader.next_seq
                        reader.next_seq += 1
                        await websocket.send_json(
                            {
                                "type": "event",
                                "id": request_id,
                                "event": terminal_event,
                            }
                        )
                    await websocket.send_json({"type": "done", "id": request_id})
                    await websocket.close(code=1000)
                    return

            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=poll_interval_s
                )
            except asyncio.TimeoutError:
                continue

            if not isinstance(message, dict):
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": request_id,
                        "error": "Invalid websocket message. Expected JSON object.",
                    }
                )
                continue

            message_type = str(message.get("type", "control")).lower()
            if message_type == "ping":
                await websocket.send_json({"type": "pong", "id": request_id})
                continue
            if message_type in {"close", "disconnect"}:
                await websocket.close(code=1000)
                return
            if message_type != "control":
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": request_id,
                        "error": "Unknown message type. Supported: control, ping, close.",
                    }
                )
                continue

            try:
                control_request = _normalize_realtime_ws_control_updates(message)
            except Exception as exc:
                await websocket.send_json(
                    {
                        "type": "control_ack",
                        "id": request_id,
                        "status": "error",
                        "applied": {},
                        "error": str(exc),
                    }
                )
                continue

            control_updates = control_request.model_dump(exclude_none=True)
            applied = await _apply_realtime_control_updates(
                request_id=request_id,
                artifact_dir=artifact_dir,
                control_updates=control_updates,
            )

            await websocket.send_json(
                {
                    "type": "control_ack",
                    "id": request_id,
                    "status": "ok" if applied else "noop",
                    "applied": applied,
                }
            )

        except WebSocketDisconnect:
            return
        except Exception as exc:
            logger.error("Realtime websocket error for %s: %s", request_id, exc)
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {
                        "type": "error",
                        "id": request_id,
                        "error": str(exc),
                    }
                )
                await websocket.close(code=1011)
            return


@router.get("/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint with extended diffusion info."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    model_info = get_model_info(server_args.model_path, backend=server_args.backend)

    card_kwargs = {
        "id": server_args.model_path,
        "root": server_args.model_path,
        # Extended diffusion-specific fields
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
    }

    if model_info:
        card_kwargs["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        card_kwargs["pipeline_class"] = model_info.pipeline_cls.__name__

    model_card = DiffusionModelCard(**card_kwargs)

    # Return dict directly to preserve extended fields (ModelList strips them)
    return {"object": "list", "data": [model_card.model_dump()]}


@router.get("/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieve a model instance. OpenAI-compatible endpoint with extended diffusion info."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    if model != server_args.model_path:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    model_info = get_model_info(server_args.model_path, backend=server_args.backend)

    card_kwargs = {
        "id": model,
        "root": model,
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
    }

    if model_info:
        card_kwargs["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        card_kwargs["pipeline_class"] = model_info.pipeline_cls.__name__

    # Return dict to preserve extended fields
    return DiffusionModelCard(**card_kwargs).model_dump()
