# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import os
import time
from typing import Any, List, Optional

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

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
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
    IMAGE_STORE,
    REALTIME_STORE,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    build_sampling_params,
    choose_output_image_ext,
    merge_image_input_list,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1/images", tags=["images"])
logger = init_logger(__name__)


def _read_b64_for_paths(paths: list[str]) -> list[str]:
    """Read and base64-encode each file. Must be called before cloud upload deletes them."""
    result = []
    for path in paths:
        with open(path, "rb") as f:
            result.append(base64.b64encode(f.read()).decode("utf-8"))
    return result


def _build_image_response_kwargs(
    save_file_path_list: list[str],
    resp_format: str,
    prompt: str,
    request_id: str,
    result: OutputBatch,
    *,
    b64_list: list[str] | None = None,
    cloud_url: str | None = None,
    fallback_url: str | None = None,
) -> dict:
    """Build ImageResponse data list.

    For b64_json: uses pre-read b64_list (call _read_b64_for_paths first).
    For url: uses cloud_url or fallback_url.
    """
    ret = None
    if resp_format == "b64_json":
        if not b64_list:
            raise ValueError("b64_list required for b64_json response_format")
        data = [
            ImageResponseData(
                b64_json=b64,
                revised_prompt=prompt,
                file_path=os.path.abspath(path),
            )
            for b64, path in zip(b64_list, save_file_path_list)
        ]
        ret = {"data": data}
    elif resp_format == "url":
        url = cloud_url or fallback_url
        if not url:
            raise HTTPException(
                status_code=400,
                detail="response_format='url' requires cloud storage to be configured.",
            )
        ret = {
            "data": [
                ImageResponseData(
                    url=url,
                    revised_prompt=prompt,
                    file_path=os.path.abspath(save_file_path_list[0]),
                )
            ],
        }
    else:
        raise HTTPException(
            status_code=400, detail=f"response_format={resp_format} is not supported"
        )

    ret = add_common_data_to_response(ret, request_id=request_id, result=result)

    return ret


def _build_image_batch(
    request_id: str,
    request: ImageGenerationsRequest,
    *,
    realtime_enabled: bool,
):
    ext = choose_output_image_ext(request.output_format, request.background)
    sampling = build_sampling_params(
        request_id,
        prompt=request.prompt,
        size=request.size,
        num_outputs_per_prompt=max(1, min(int(request.n or 1), 10)),
        output_file_name=f"{request_id}.{ext}",
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        negative_prompt=request.negative_prompt,
        enable_teacache=request.enable_teacache,
        output_compression=request.output_compression,
        output_quality=request.output_quality,
        realtime_enabled=realtime_enabled,
        realtime_stream_every_n_steps=request.realtime_stream_every_n_steps,
        realtime_decode_preview=request.realtime_decode_preview,
    )
    batch = prepare_request(
        server_args=get_global_server_args(),
        sampling_params=sampling,
    )
    if request.diffusers_kwargs:
        batch.extra["diffusers_kwargs"] = request.diffusers_kwargs
    return batch


async def _execute_image_batch(
    request: ImageGenerationsRequest,
    request_id: str,
    batch,
) -> tuple[dict[str, Any], OutputBatch]:
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    save_file_path = save_file_path_list[0]
    resp_format = (request.response_format or "b64_json").lower()

    b64_list = (
        _read_b64_for_paths(save_file_path_list) if resp_format == "b64_json" else None
    )
    cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": None if cloud_url else save_file_path,
            "url": cloud_url,
            "artifact_dir": batch.extra.get("realtime_artifact_dir"),
        },
    )

    response_kwargs = _build_image_response_kwargs(
        save_file_path_list,
        resp_format,
        request.prompt,
        request_id,
        result,
        b64_list=b64_list,
        cloud_url=cloud_url,
    )
    return response_kwargs, result


@router.post("/generations", response_model=ImageResponse)
async def generations(
    request: ImageGenerationsRequest,
):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /v1/images/generations/stream for realtime streaming responses.",
        )
    request_id = generate_request_id()
    batch = _build_image_batch(request_id, request, realtime_enabled=False)
    response_kwargs, _ = await _execute_image_batch(request, request_id, batch)
    return ImageResponse(**response_kwargs)


@router.post("/generations/stream")
async def generations_stream(
    raw_request: Request,
    request: ImageGenerationsRequest,
):
    request_id = generate_request_id()
    batch = _build_image_batch(request_id, request, realtime_enabled=True)
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

    await REALTIME_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "kind": "image",
            "status": "queued",
            "artifact_dir": artifact_dir,
            "cancel_on_disconnect": cancel_on_disconnect,
            "expires_at": expires_at,
        },
    )

    emit_realtime_session_accepted(
        artifact_dir=artifact_dir,
        request_id=request_id,
        kind="image",
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

    async def _run_job():
        try:
            response_kwargs, result = await _execute_image_batch(
                request, request_id, batch
            )
            response_payload = ImageResponse(**response_kwargs).model_dump(mode="json")
            terminal_type = "cancelled" if result.cancelled else "completed"
            emit_realtime_terminal_result(
                artifact_dir=artifact_dir,
                request_id=request_id,
                status=terminal_type,
                result_payload=response_payload,
            )
            await REALTIME_STORE.update_fields(
                request_id, {"status": terminal_type, "result": response_payload}
            )
        except Exception as e:
            error_message = str(e)
            emit_realtime_terminal_error(
                artifact_dir=artifact_dir,
                request_id=request_id,
                error_message=error_message,
            )
            await REALTIME_STORE.update_fields(
                request_id,
                {"status": "failed", "error": error_message},
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


@router.post("/edits", response_model=ImageResponse)
async def edits(
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    url: Optional[List[str]] = Form(None),
    url_array: Optional[List[str]] = Form(None, alias="url[]"),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    background: Optional[str] = Form("auto"),
    seed: Optional[int] = Form(1024),
    generator_device: Optional[str] = Form("cuda"),
    user: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    output_quality: Optional[str] = Form("default"),
    output_compression: Optional[int] = Form(None),
    enable_teacache: Optional[bool] = Form(False),
    num_frames: int = Form(1),
):
    request_id = generate_request_id()
    # Resolve images from either `image` or `image[]` (OpenAI SDK sends `image[]` when list is provided)
    images = image or image_array
    urls = url or url_array

    if (not images or len(images) == 0) and (not urls or len(urls) == 0):
        raise HTTPException(
            status_code=422, detail="Field 'image' or 'url' is required"
        )

    # Save all input images; additional images beyond the first are saved for potential future use
    uploads_dir = os.path.join("inputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    image_list = merge_image_input_list(images, urls)

    input_paths = []
    try:
        for idx, img in enumerate(image_list):
            filename = img.filename if hasattr(img, "filename") else f"image_{idx}"
            input_path = await save_image_to_path(
                img, os.path.join(uploads_dir, f"{request_id}_{idx}_{filename}")
            )
            input_paths.append(input_path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process image source: {str(e)}"
        )

    ext = choose_output_image_ext(output_format, background)
    sampling = build_sampling_params(
        request_id,
        prompt=prompt,
        size=size,
        num_outputs_per_prompt=max(1, min(int(n or 1), 10)),
        output_file_name=f"{request_id}.{ext}",
        image_path=input_paths,
        seed=seed,
        generator_device=generator_device,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        enable_teacache=enable_teacache,
        num_frames=num_frames,
        output_compression=output_compression,
        output_quality=output_quality,
    )
    batch = prepare_request(
        server_args=get_global_server_args(),
        sampling_params=sampling,
    )
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    save_file_path = save_file_path_list[0]
    resp_format = (response_format or "b64_json").lower()

    # read b64 before cloud upload may delete the local file
    b64_list = (
        _read_b64_for_paths(save_file_path_list) if resp_format == "b64_json" else None
    )

    cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)

    await IMAGE_STORE.upsert(
        request_id,
        {
            "id": request_id,
            "created_at": int(time.time()),
            "file_path": None if cloud_url else save_file_path,
            "url": cloud_url,
            "input_image_paths": input_paths,
            "num_input_images": len(input_paths),
        },
    )

    response_kwargs = _build_image_response_kwargs(
        save_file_path_list,
        resp_format,
        prompt,
        request_id,
        result,
        b64_list=b64_list,
        cloud_url=cloud_url,
        fallback_url=f"/v1/images/{request_id}/content",
    )

    return ImageResponse(**response_kwargs)


@router.delete("/{image_id}")
async def delete_image(image_id: str = Path(...)):
    realtime_job = await REALTIME_STORE.get(image_id)
    if realtime_job:
        terminal_statuses = {"completed", "failed", "deleted", "cancelled"}
        if realtime_job.get("status") in terminal_statuses:
            await REALTIME_STORE.pop(image_id)
            removed = await IMAGE_STORE.pop(image_id)
            if removed:
                return {"id": image_id, "status": "deleted"}
            return {"id": image_id, "status": "deleted"}

        artifact_dir = realtime_job.get("artifact_dir")
        if artifact_dir:
            apply_realtime_control(
                request_id=image_id,
                artifact_dir=artifact_dir,
                control_updates={"cancel": True},
            )
        await REALTIME_STORE.update_fields(image_id, {"status": "cancelling"})
        return {"id": image_id, "status": "cancelling"}

    item = await IMAGE_STORE.pop(image_id)
    if item:
        return {"id": image_id, "status": "deleted"}

    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/{image_id}/content")
async def download_image_content(
    image_id: str = Path(...), variant: Optional[str] = Query(None)
):
    item = await IMAGE_STORE.get(image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")

    if item.get("url"):
        raise HTTPException(
            status_code=400,
            detail=f"Image has been uploaded to cloud storage. Please use the cloud URL: {item.get('url')}",
        )

    file_path = item.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image is still being generated")

    ext = os.path.splitext(file_path)[1].lower()
    media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"

    return FileResponse(
        path=file_path, media_type=media_type, filename=os.path.basename(file_path)
    )
