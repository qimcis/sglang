# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime_utils import (
    apply_realtime_control,
    emit_realtime_session_accepted,
    emit_realtime_terminal_error,
    emit_realtime_terminal_result,
    resolve_realtime_session_ttl_seconds,
    stream_realtime_events,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import (
    REALTIME_STORE,
    VIDEO_STORE,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
    add_common_data_to_response,
    build_sampling_params,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/videos", tags=["videos"])


def _build_video_sampling_params(
    request_id: str, request: VideoGenerationsRequest, *, realtime_enabled: bool
):
    """Resolve video-specific defaults (fps, seconds → num_frames) then
    delegate to the shared build_sampling_params."""
    seconds = request.seconds if request.seconds is not None else DEFAULT_VIDEO_SECONDS
    fps = request.fps if request.fps is not None else DEFAULT_FPS
    num_frames = request.num_frames if request.num_frames is not None else fps * seconds

    return build_sampling_params(
        request_id,
        prompt=request.prompt,
        size=request.size,
        num_frames=num_frames,
        fps=fps,
        image_path=request.input_reference,
        output_file_name=request_id,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        guidance_scale_2=request.guidance_scale_2,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        output_path=request.output_path,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        realtime_enabled=realtime_enabled,
        realtime_stream_every_n_steps=request.realtime_stream_every_n_steps,
        realtime_decode_preview=request.realtime_decode_preview,
    )


# extract metadata which http_server needs to know
def _video_job_from_sampling(
    request_id: str, req: VideoGenerationsRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    size_str = f"{sampling.width}x{sampling.height}"
    seconds = int(round((sampling.num_frames or 0) / float(sampling.fps or 24)))
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "sora-2",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": size_str,
        "seconds": str(seconds),
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
    }


async def _save_first_input_image(image_sources, request_id: str) -> str | None:
    """Save the first input image from a list of sources and return its path."""
    image_list = merge_image_input_list(image_sources)
    if not image_list:
        return None
    image = image_list[0]

    uploads_dir = os.path.join("inputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    filename = image.filename if hasattr(image, "filename") else "url_image"
    target_path = os.path.join(uploads_dir, f"{request_id}_{filename}")
    return await save_image_to_path(image, target_path)


async def _dispatch_job_async(job_id: str, batch: Req) -> None:
    artifact_dir = batch.extra.get("realtime_artifact_dir")
    try:
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        save_file_path = save_file_path_list[0]

        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)
        terminal_status = "cancelled" if result.cancelled else "completed"

        update_fields = {
            "status": terminal_status,
            "progress": 100,
            "completed_at": int(time.time()),
            "url": cloud_url,
            "file_path": save_file_path if not cloud_url else None,
        }
        update_fields = add_common_data_to_response(
            update_fields, request_id=job_id, result=result
        )
        await VIDEO_STORE.update_fields(job_id, update_fields)

        if artifact_dir:
            emit_realtime_terminal_result(
                artifact_dir=artifact_dir,
                request_id=job_id,
                status=terminal_status,
                result_payload=update_fields,
            )
            await REALTIME_STORE.update_fields(
                job_id, {"status": terminal_status, "result": update_fields}
            )
    except Exception as e:
        logger.error(f"{e}")
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
        if artifact_dir:
            emit_realtime_terminal_error(
                artifact_dir=artifact_dir,
                request_id=job_id,
                error_message=str(e),
            )
            await REALTIME_STORE.update_fields(
                job_id,
                {"status": "failed", "error": str(e)},
            )


# TODO: support image to video generation
# TODO: this is currently not used
@router.post("", response_model=VideoResponse)
async def create_video(
    request: Request,
    # multipart/form-data fields (optional; used only when content-type is multipart)
    prompt: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    reference_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    num_frames: Optional[int] = Form(None),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    output_quality: Optional[str] = Form("default"),
    output_compression: Optional[int] = Form(None),
    extra_body: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "").lower()
    request_id = generate_request_id()

    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    if "multipart/form-data" in content_type:
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        # Validate image input based on model task type
        image_sources = merge_image_input_list(input_reference, reference_url)
        if task_type.requires_image_input() and not image_sources:
            raise HTTPException(
                status_code=400,
                detail="input_reference or reference_url is required for image-to-video generation",
            )
        try:
            input_path = await _save_first_input_image(image_sources, request_id)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to process image source: {str(e)}"
            )

        # Parse extra_body JSON (if provided in multipart form) to get fps/num_frames overrides
        extra_from_form: Dict[str, Any] = {}
        if extra_body:
            try:
                extra_from_form = json.loads(extra_body)
            except Exception:
                extra_from_form = {}

        fps_val = fps if fps is not None else extra_from_form.get("fps")
        num_frames_val = (
            num_frames if num_frames is not None else extra_from_form.get("num_frames")
        )

        req = VideoGenerationsRequest(
            prompt=prompt,
            input_reference=input_path,
            model=model,
            seconds=seconds if seconds is not None else 4,
            size=size,
            fps=fps_val,
            num_frames=num_frames_val,
            seed=seed,
            generator_device=generator_device,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
            output_compression=output_compression,
            output_quality=output_quality,
            **(
                {"guidance_scale": guidance_scale} if guidance_scale is not None else {}
            ),
        )
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            # If client uses extra_body, merge it into the top-level payload
            payload: Dict[str, Any] = dict(body or {})
            extra = payload.pop("extra_body", None)
            if isinstance(extra, dict):
                # Shallow-merge: only keys like fps/num_frames are expected
                payload.update(extra)
            # openai may turn extra_body to extra_json
            extra_json = payload.pop("extra_json", None)
            if isinstance(extra_json, dict):
                payload.update(extra_json)
            # Validate image input based on model task type
            has_image_input = payload.get("reference_url") or payload.get(
                "input_reference"
            )
            if task_type.requires_image_input() and not has_image_input:
                raise HTTPException(
                    status_code=400,
                    detail="input_reference or reference_url is required for image-to-video generation",
                )
            # for non-multipart/form-data type
            if payload.get("reference_url"):
                try:
                    input_path = await _save_first_input_image(
                        payload.get("reference_url"), request_id
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image source: {str(e)}",
                    )
                payload["input_reference"] = input_path
            req = VideoGenerationsRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    logger.debug(f"Server received from create_video endpoint: req={req}")
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /v1/videos/stream for realtime streaming responses.",
        )

    sampling_params = _build_video_sampling_params(
        request_id, req, realtime_enabled=False
    )
    job = _video_job_from_sampling(request_id, req, sampling_params)

    # Build Req for scheduler
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )
    artifact_dir = batch.extra.get("realtime_artifact_dir")
    if artifact_dir:
        job["artifact_dir"] = artifact_dir
        await REALTIME_STORE.upsert(
            request_id,
            {
                "id": request_id,
                "kind": "video",
                "status": "queued",
                "created_at": int(time.time()),
                "artifact_dir": artifact_dir,
            },
        )
        emit_realtime_session_accepted(
            artifact_dir=artifact_dir,
            request_id=request_id,
            kind="video",
        )
    await VIDEO_STORE.upsert(request_id, job)
    # Add diffusers_kwargs if provided
    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs
    # Enqueue the job asynchronously and return immediately
    asyncio.create_task(_dispatch_job_async(request_id, batch))
    return VideoResponse(**job)


@router.post("/stream")
async def stream_video(
    raw_request: Request,
    request: VideoGenerationsRequest,
):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    if task_type.requires_image_input() and not (
        request.input_reference or request.reference_url
    ):
        raise HTTPException(
            status_code=400,
            detail="input_reference or reference_url is required for image-to-video generation",
        )

    req = request
    if request.reference_url and not request.input_reference:
        try:
            input_path = await _save_first_input_image(
                request.reference_url, request_id
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image source: {str(e)}",
            )
        req = request.model_copy(update={"input_reference": input_path})

    sampling_params = _build_video_sampling_params(
        request_id, req, realtime_enabled=True
    )
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling_params,
    )
    if req.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = req.diffusers_kwargs

    artifact_dir = batch.extra.get("realtime_artifact_dir")
    if artifact_dir is None:
        raise HTTPException(
            status_code=500,
            detail="Realtime artifact directory is not available for this request.",
        )

    try:
        ttl_seconds = resolve_realtime_session_ttl_seconds(
            request.realtime_session_ttl_seconds
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    expires_at = int(time.time()) + ttl_seconds
    cancel_on_disconnect = bool(request.cancel_on_disconnect)

    job = _video_job_from_sampling(request_id, req, sampling_params)
    job["artifact_dir"] = artifact_dir
    await VIDEO_STORE.upsert(request_id, job)
    await REALTIME_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "kind": "video",
            "status": "queued",
            "created_at": int(time.time()),
            "artifact_dir": artifact_dir,
            "cancel_on_disconnect": cancel_on_disconnect,
            "expires_at": expires_at,
        },
    )

    emit_realtime_session_accepted(
        artifact_dir=artifact_dir,
        request_id=request_id,
        kind="video",
        cancel_on_disconnect=cancel_on_disconnect,
        expires_at=expires_at,
    )

    async def _cancel_for_disconnect():
        apply_realtime_control(
            request_id=request_id,
            artifact_dir=artifact_dir,
            control_updates={"cancel": True},
        )
        await REALTIME_STORE.update_fields(request_id, {"status": "cancelling"})
        await VIDEO_STORE.update_fields(request_id, {"status": "cancelling"})

    async def _run_job():
        try:
            save_file_path_list, result = await process_generation_batch(
                async_scheduler_client, batch
            )
            save_file_path = save_file_path_list[0]
            cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

            terminal_status = "cancelled" if result.cancelled else "completed"
            update_fields = {
                "status": terminal_status,
                "progress": 100,
                "completed_at": int(time.time()),
                "url": cloud_url,
                "file_path": save_file_path if not cloud_url else None,
            }
            update_fields = add_common_data_to_response(
                update_fields, request_id=request_id, result=result
            )

            await VIDEO_STORE.update_fields(request_id, update_fields)
            await REALTIME_STORE.update_fields(
                request_id, {"status": terminal_status, "result": update_fields}
            )
            emit_realtime_terminal_result(
                artifact_dir=artifact_dir,
                request_id=request_id,
                status=terminal_status,
                result_payload=update_fields,
            )
        except Exception as e:
            error_message = str(e)
            await VIDEO_STORE.update_fields(
                request_id, {"status": "failed", "error": {"message": error_message}}
            )
            await REALTIME_STORE.update_fields(
                request_id, {"status": "failed", "error": error_message}
            )
            emit_realtime_terminal_error(
                artifact_dir=artifact_dir,
                request_id=request_id,
                error_message=error_message,
            )

    producer_task = asyncio.create_task(_run_job())
    on_disconnect = _cancel_for_disconnect if cancel_on_disconnect else None
    return StreamingResponse(
        stream_realtime_events(
            request=raw_request,
            request_id=request_id,
            artifact_dir=artifact_dir,
            producer_task=producer_task,
            on_disconnect=on_disconnect,
        ),
        media_type="text/event-stream",
    )


@router.get("", response_model=VideoListResponse)
async def list_videos(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    # Normalize order
    order = (order or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"
    jobs = await VIDEO_STORE.list_values()

    reverse = order != "asc"
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=reverse)

    if after is not None:
        try:
            idx = next(i for i, j in enumerate(jobs) if j["id"] == after)
            jobs = jobs[idx + 1 :]
        except StopIteration:
            jobs = []

    if limit is not None:
        jobs = jobs[:limit]
    items = [VideoResponse(**j) for j in jobs]
    return VideoListResponse(data=items)


@router.get("/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoResponse(**job)


@router.delete("/{video_id}", response_model=VideoResponse)
async def delete_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    terminal_statuses = {"completed", "failed", "deleted", "cancelled"}
    if job.get("status") in terminal_statuses:
        removed = await VIDEO_STORE.pop(video_id)
        assert removed is not None
        removed["status"] = "deleted"
        return VideoResponse(**removed)

    artifact_dir = job.get("artifact_dir")
    if artifact_dir:
        apply_realtime_control(
            request_id=video_id,
            artifact_dir=artifact_dir,
            control_updates={"cancel": True},
        )

    await VIDEO_STORE.update_fields(video_id, {"status": "cancelling"})
    await REALTIME_STORE.update_fields(video_id, {"status": "cancelling"})
    updated = await VIDEO_STORE.get(video_id)
    assert updated is not None
    return VideoResponse(**updated)


@router.get("/{video_id}/content")
async def download_video_content(
    video_id: str = Path(...), variant: Optional[str] = Query(None)
):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Video has been uploaded to cloud storage. Please use the cloud URL: {job.get('url')}",
        )

    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    media_type = "video/mp4"  # default variant
    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
