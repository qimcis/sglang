"""Calibrate memory-aware dynamic batching profiles.

Example:

    python -m sglang.multimodal_gen.tools.calibrate_batch_memory \
        --calibration-shapes 1024x1024x1 \
        --calibration-batch-sizes 1,2,4,8 \
        --model-path black-forest-labs/FLUX.2-klein-4B \
        --num-inference-steps 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zmq

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import SchedulerClient
from sglang.multimodal_gen.runtime.server_args import ServerArgs, prepare_server_args


@dataclass(frozen=True)
class CalibrationShape:
    width: int
    height: int
    num_frames: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}x{self.num_frames}"


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, ServerArgs]:
    parser = argparse.ArgumentParser(
        description=(
            "Run real dynamic batches and persist memory-aware admission profiles. "
            "Pass normal SGLang diffusion server args after the calibration args."
        )
    )
    parser.add_argument(
        "--calibration-shapes",
        "--shapes",
        dest="calibration_shapes",
        required=True,
        help="Comma-separated WIDTHxHEIGHT or WIDTHxHEIGHTxFRAMES entries.",
    )
    parser.add_argument(
        "--calibration-batch-sizes",
        "--batch-sizes",
        dest="calibration_batch_sizes",
        default="1,2,4,8",
        help="Comma-separated concurrent request counts to run for each shape.",
    )
    parser.add_argument(
        "--calibration-output-cache",
        "--output-cache",
        dest="calibration_output_cache",
        default="auto",
        help=(
            "Memory profile cache directory, or 'auto' for the normal SGLang "
            "diffusion cache location."
        ),
    )
    parser.add_argument(
        "--calibration-summary-json",
        default=None,
        help="Optional path for the calibration summary JSON.",
    )
    parser.add_argument(
        "--calibration-output-dir",
        default=None,
        help=(
            "Directory for generated calibration outputs. Defaults to a "
            "calibration_outputs directory under the profile cache root."
        ),
    )
    parser.add_argument(
        "--calibration-prompt",
        "--prompt",
        dest="calibration_prompt",
        default="A detailed photo of a mountain landscape at sunrise",
        help="Prompt used for calibration requests.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=42,
        help="Base seed for calibration requests.",
    )
    parser.add_argument(
        "--calibration-negative-prompt",
        "--negative-prompt",
        dest="calibration_negative_prompt",
        default=None,
        help="Optional negative prompt used for calibration requests.",
    )
    parser.add_argument(
        "--calibration-num-inference-steps",
        "--num-inference-steps",
        dest="calibration_num_inference_steps",
        type=int,
        default=None,
        help="Number of denoising steps used for calibration requests.",
    )
    parser.add_argument(
        "--calibration-guidance-scale",
        "--guidance-scale",
        dest="calibration_guidance_scale",
        type=float,
        default=None,
        help="Guidance scale used for calibration requests.",
    )
    parser.add_argument(
        "--calibration-fps",
        "--fps",
        dest="calibration_fps",
        type=int,
        default=None,
        help="FPS used for calibration video requests.",
    )
    parser.add_argument(
        "--calibration-delay-ms",
        type=float,
        default=10.0,
        help="Batching delay used to coalesce concurrent calibration requests.",
    )
    parser.add_argument(
        "--calibration-timeout-s",
        "--per-shape-timeout-s",
        dest="calibration_timeout_s",
        type=float,
        default=600.0,
        help="Timeout for each calibration batch size run.",
    )
    parser.add_argument(
        "--calibration-max-workers",
        type=int,
        default=None,
        help="Maximum concurrent client threads. Defaults to max calibration batch size.",
    )
    parser.add_argument(
        "--calibration-stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop larger batch sizes for a shape after the first failed run.",
    )

    calibration_args, server_argv = parser.parse_known_args(argv)
    server_args = _prepare_server_args(server_argv)
    return calibration_args, server_args


def parse_shapes(raw: str) -> list[CalibrationShape]:
    shapes: list[CalibrationShape] = []
    for item in _csv(raw):
        parts = item.lower().split("x")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid calibration shape {item!r}. Expected WIDTHxHEIGHT[xFRAMES]."
            )
        width = int(parts[0])
        height = int(parts[1])
        frames = int(parts[2]) if len(parts) == 3 else 1
        if width <= 0 or height <= 0 or frames <= 0:
            raise ValueError(f"Calibration shape values must be positive: {item!r}")
        shapes.append(CalibrationShape(width=width, height=height, num_frames=frames))
    if not shapes:
        raise ValueError("At least one calibration shape is required")
    return shapes


def parse_batch_sizes(raw: str) -> list[int]:
    sizes = sorted({int(item) for item in _csv(raw)})
    if not sizes or sizes[0] < 1:
        raise ValueError("Calibration batch sizes must be positive")
    return sizes


def configure_server_args(
    server_args: ServerArgs,
    calibration_args: argparse.Namespace,
    batch_sizes: list[int],
) -> None:
    server_args.batching_mode = "dynamic"
    server_args.batching_max_size = max(server_args.batching_max_size, max(batch_sizes))
    server_args.batching_delay_ms = max(
        float(server_args.batching_delay_ms),
        float(calibration_args.calibration_delay_ms),
    )
    server_args.batching_memory_profile_cache = (
        calibration_args.calibration_output_cache
    )
    if hasattr(server_args, "_validate_batching"):
        server_args._validate_batching()


def run_calibration(
    calibration_args: argparse.Namespace,
    server_args: ServerArgs,
) -> dict[str, Any]:
    shapes = parse_shapes(calibration_args.calibration_shapes)
    requested_batch_sizes = parse_batch_sizes(calibration_args.calibration_batch_sizes)
    batch_sizes = densify_batch_sizes(requested_batch_sizes)
    configure_server_args(server_args, calibration_args, batch_sizes)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_path": server_args.model_path,
        "requested_batch_sizes": requested_batch_sizes,
        "batch_sizes": batch_sizes,
        "cache_root": _cache_root(server_args.batching_memory_profile_cache),
        "results": [],
    }
    user_output_dir = calibration_args.calibration_output_dir
    temp_output_dir = None
    if calibration_args.calibration_output_dir is None:
        temp_output_dir = tempfile.TemporaryDirectory(prefix="sglang-calib-")
        calibration_args.calibration_output_dir = temp_output_dir.name
    summary["output_dir"] = user_output_dir
    summary["outputs_persisted"] = user_output_dir is not None
    max_workers = max(1, calibration_args.calibration_max_workers or max(batch_sizes))
    run_started_at = time.time()

    try:
        with DiffGenerator.from_server_args(server_args, local_mode=True) as _generator:
            for shape in shapes:
                singleton_ok = False
                for batch_size in batch_sizes:
                    if batch_size > 1 and not singleton_ok:
                        summary["results"].append(
                            {
                                "shape": shape.label,
                                "batch_size": batch_size,
                                "status": "skipped",
                                "reason": "singleton_failed",
                            }
                        )
                        continue

                    result = _run_one_batch(
                        server_args=server_args,
                        calibration_args=calibration_args,
                        shape=shape,
                        batch_size=batch_size,
                        max_workers=max_workers,
                    )
                    summary["results"].append(result)

                    if batch_size == 1 and result["status"] == "success":
                        singleton_ok = True
                    if (
                        result["status"] != "success"
                        and calibration_args.calibration_stop_on_failure
                    ):
                        break
    finally:
        if temp_output_dir is not None:
            temp_output_dir.cleanup()

    summary["profile_files"] = _profile_files(summary["cache_root"])
    summary["profile_observations"] = _profile_observations(
        summary["cache_root"], modified_after=run_started_at - 1.0
    )
    return summary


def _run_one_batch(
    *,
    server_args: ServerArgs,
    calibration_args: argparse.Namespace,
    shape: CalibrationShape,
    batch_size: int,
    max_workers: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    timeout_s = max(1.0, float(calibration_args.calibration_timeout_s))
    reqs = [
        _build_req(
            server_args=server_args,
            calibration_args=calibration_args,
            shape=shape,
            index=i,
            batch_size=batch_size,
        )
        for i in range(batch_size)
    ]

    outputs = []
    client_errors = []
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_workers, batch_size)
    )
    try:
        futures = [
            executor.submit(_forward_req, server_args, req, timeout_s) for req in reqs
        ]
        try:
            completed = concurrent.futures.as_completed(
                futures, timeout=timeout_s + 5.0
            )
            for future in completed:
                try:
                    outputs.append(future.result())
                except Exception as e:
                    client_errors.append(str(e))
        except concurrent.futures.TimeoutError:
            for future in futures:
                future.cancel()
            client_errors.append(f"calibration timed out after {timeout_s:.1f}s")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    output_errors = [output.error for output in outputs if output.error]
    errors = output_errors + client_errors
    oom_count = sum(1 for output in outputs if getattr(output, "is_oom", False))
    oom_count += sum(1 for error in errors if _is_oom_error(error))
    peak_reserved = max(
        (output.peak_memory_mb or 0.0 for output in outputs), default=0.0
    )
    peak_allocated = max(
        (output.peak_allocated_memory_mb or 0.0 for output in outputs),
        default=0.0,
    )
    status = "success" if not errors else "oom" if oom_count else "error"
    return {
        "shape": shape.label,
        "batch_size": batch_size,
        "status": status,
        "duration_s": time.perf_counter() - start,
        "peak_memory_mb": peak_reserved,
        "peak_allocated_memory_mb": peak_allocated,
        "oom_count": oom_count,
        "error_count": len(errors),
        "errors": errors[:3],
    }


def _prepare_server_args(server_argv: list[str]) -> ServerArgs:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *server_argv]
        return prepare_server_args(server_argv)
    finally:
        sys.argv = original_argv


def densify_batch_sizes(requested_batch_sizes: list[int]) -> list[int]:
    """Include the geometric ramp needed by memory-aware cold-start admission."""
    max_batch_size = max(requested_batch_sizes)
    sizes = {1, *requested_batch_sizes}
    batch_size = 1
    while batch_size < max_batch_size:
        batch_size *= 2
        sizes.add(min(batch_size, max_batch_size))
    return sorted(sizes)


def _is_oom_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = str(error).lower()
    return "out of memory" in lowered or "cuda oom" in lowered or "hip oom" in lowered


def _build_req(
    *,
    server_args: ServerArgs,
    calibration_args: argparse.Namespace,
    shape: CalibrationShape,
    index: int,
    batch_size: int,
):
    sampling_kwargs = {
        "prompt": calibration_args.calibration_prompt,
        "width": shape.width,
        "height": shape.height,
        "num_frames": shape.num_frames,
        "seed": calibration_args.calibration_seed + index,
        "request_id": f"calib-{shape.label}-b{batch_size}-{index}-{time.time_ns()}",
        "output_path": calibration_args.calibration_output_dir,
        "suppress_logs": True,
    }
    if calibration_args.calibration_negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = (
            calibration_args.calibration_negative_prompt
        )
    if calibration_args.calibration_num_inference_steps is not None:
        sampling_kwargs["num_inference_steps"] = (
            calibration_args.calibration_num_inference_steps
        )
    if calibration_args.calibration_guidance_scale is not None:
        sampling_kwargs["guidance_scale"] = calibration_args.calibration_guidance_scale
    if calibration_args.calibration_fps is not None:
        sampling_kwargs["fps"] = calibration_args.calibration_fps

    sampling_params = SamplingParams.from_user_sampling_params_args(
        server_args.model_path,
        server_args=server_args,
        **sampling_kwargs,
    )
    return prepare_request(server_args=server_args, sampling_params=sampling_params)


def _forward_req(server_args: ServerArgs, req, timeout_s: float):
    client = SchedulerClient()
    client.initialize(server_args)
    timeout_ms = int(max(1.0, timeout_s) * 1000.0)
    client.scheduler_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    client.scheduler_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        return client.forward(req)
    finally:
        client.close()


def write_summary(summary: dict[str, Any], requested_path: str | None) -> str:
    if requested_path is None:
        cache_root = Path(summary["cache_root"]).expanduser()
        requested_path = str(
            cache_root / f"calibration_summary_{int(time.time())}.json"
        )
    path = Path(requested_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    calibration_args, server_args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run_calibration(calibration_args, server_args)
    summary_path = write_summary(summary, calibration_args.calibration_summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Calibration summary written to {summary_path}")
    return 0 if _has_successful_singletons(summary) else 1


def _has_successful_singletons(summary: dict[str, Any]) -> bool:
    by_shape: dict[str, bool] = {}
    for result in summary["results"]:
        by_shape.setdefault(result["shape"], False)
        if result["batch_size"] == 1 and result["status"] == "success":
            by_shape[result["shape"]] = True
    return all(by_shape.values()) if by_shape else False


def _cache_root(cache_setting: str | None) -> str:
    if cache_setting in (None, "", "auto"):
        return str(
            Path(
                os.environ.get(
                    "SGLANG_DIFFUSION_BATCH_MEMORY_CACHE",
                    Path.home() / ".cache" / "sglang" / "diffusion_batch_memory",
                )
            ).expanduser()
        )
    return str(Path(cache_setting).expanduser())


def _profile_files(cache_root: str) -> list[str]:
    path = Path(cache_root).expanduser()
    if not path.is_dir():
        return []
    return sorted(str(item) for item in path.glob("*.json"))


def _profile_observations(
    cache_root: str, *, modified_after: float
) -> list[dict[str, Any]]:
    observations = []
    for path_str in _profile_files(cache_root):
        path = Path(path_str)
        try:
            if path.stat().st_mtime < modified_after:
                continue
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        for item in payload.get("profiles", []):
            successes = item.get("profile", {}).get("successes", [])
            observations.append(
                {
                    "profile_file": path_str,
                    "success_count": len(successes),
                    "batch_sizes": sorted(
                        {int(success.get("batch_size", 1)) for success in successes}
                    ),
                }
            )
    return observations


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
