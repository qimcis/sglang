from __future__ import annotations

import concurrent.futures
import logging
from typing import Iterable, List, Tuple

import torch
import tqdm

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    inverse_transform_scale_ue8m0,
    normalize_e4m3fn_to_e4m3fnuz,
    quant_weight_ue8m0,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.utils import (
    maybe_executor_submit,
    should_async_load,
    should_deepgemm_weight_requant_ue8m0,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import bind_or_assign, get_bool_env_var, log_info_on_rank0

from .utils import (
    NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
    _is_cpu,
    _is_cpu_amx_available,
    _is_cuda,
    _is_fp8_fnuz,
    _is_hip,
    _is_npu,
    _use_aiter,
    awq_dequantize,
    enable_nextn_moe_bf16_cast_to_fp8,
    quark_post_load_weights,
)

logger = logging.getLogger(__name__)


class DeepseekV2WeightLoaderMixin:
    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 256
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def post_load_weights(self, is_nextn=False, weight_names=None):
        # Perform post-processing after loading weights
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.model.start_layer, self.model.end_layer)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if layer_id < self.config.num_hidden_layers:
                            layer_ids.add(layer_id)

        for layer_id in layer_ids:
            self_attn = (
                self.model.layers[layer_id].self_attn
                if not is_nextn
                else self.model.decoder.self_attn
            )
            if hasattr(self_attn.kv_b_proj, "qweight"):
                # AWQ compatible
                if _is_cuda or _is_hip or _is_npu:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
            else:
                w = self_attn.kv_b_proj.weight
            # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
            # This may affect the accuracy of fp8 model.
            # Fix deepseek v3 blockwise bmm by using deep_gemm
            use_deep_gemm_bmm = False

            if w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                # For mixed quantization (experts int4, linear fp8), use linear_fp8_config
                selected_quant_config = getattr(
                    self.quant_config, "linear_fp8_config", None
                )
                if selected_quant_config is None:
                    selected_quant_config = self.quant_config
                weight_block_size = getattr(
                    selected_quant_config, "weight_block_size", None
                )
                if weight_block_size is not None:
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv") or hasattr(
                        self_attn.kv_b_proj, "weight_scale"
                    )
                    weight_scale = (
                        self_attn.kv_b_proj.weight_scale
                        if hasattr(self_attn.kv_b_proj, "weight_scale")
                        else self_attn.kv_b_proj.weight_scale_inv
                    )
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=weight_scale,
                            input_scale=None,
                        )
                    else:
                        weight = w

                    # In multiple weight loading scenarios (e.g. RL), we need to inverse the scale of the weights after the requantization happened at the first loading.
                    if (
                        should_deepgemm_weight_requant_ue8m0(
                            weight_block_size=getattr(
                                self.quant_config, "weight_block_size", None
                            )
                        )
                        and weight_scale.format_ue8m0
                    ):
                        weight_scale = inverse_transform_scale_ue8m0(
                            weight_scale, mn=weight.shape[-2]
                        )

                    if (
                        _is_cuda
                        and weight_block_size[0] == 128
                        and weight_block_size[1] == 128
                    ):
                        if (
                            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                            and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                            and get_bool_env_var("SGL_USE_DEEPGEMM_BMM", "false")
                        ):
                            block_scale = weight_scale
                            use_deep_gemm_bmm = True
                        else:
                            w = block_quant_dequant(
                                weight,
                                weight_scale,
                                weight_block_size,
                                torch.bfloat16,
                            )
                    else:
                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                else:
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale

                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                    self_attn.w_scale = scale

            if w.dtype == torch.int8:
                if hasattr(self.quant_config, "weight_block_size"):
                    # block-wise int8 need it
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                else:
                    # channel-wise int8 need it
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                        torch.bfloat16
                    )

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)

            if (
                _use_aiter
                and self.quant_config is not None
                and self.quant_config.get_name() == "quark"
                and quark_post_load_weights is not None
            ):
                w_kc, self_attn.w_scale_k, w_vc, self_attn.w_scale_v = (
                    quark_post_load_weights(self_attn, w, "mxfp4")
                )

            if not use_deep_gemm_bmm:
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                w_vc = w_vc.contiguous().transpose(1, 2)
                if _is_npu:
                    w_vc = w_vc.contiguous()
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = bind_or_assign(
                        self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                    )
                    if _is_hip:
                        self_attn.w_scale *= 2.0
                # TODO: remove this after adding FP8 support in bmm cpu kernel
                if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
                    self_attn.w_kc = (
                        self_attn.w_kc.to(torch.bfloat16) * self_attn.w_scale
                    )
                    self_attn.w_vc = (
                        self_attn.w_vc.to(torch.bfloat16) * self_attn.w_scale
                    )
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign(
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign(
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
                self_attn.use_deep_gemm_bmm = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        weights = self._maybe_quant_weights_to_fp8_ue8m0(
            weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, is_nextn
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.
        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            weight_names = []
            for name, loaded_weight in weights:
                use_async_loading = should_async_load(loaded_weight)
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    continue
                if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                    name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts}",
                    )

                weight_names.append(name)

                if not is_nextn:
                    if hasattr(self.config, "num_nextn_predict_layers"):
                        num_nextn_layers = self.config.num_nextn_predict_layers
                        if num_nextn_layers > 0 and name.startswith("model.layers"):
                            name_list = name.split(".")
                            if (
                                len(name_list) >= 3
                                and int(name_list[2]) >= self.config.num_hidden_layers
                            ):
                                continue
                else:
                    if not name.startswith(nextn_layer_prefix):
                        continue

                    # Use shared head and embed weights from target model
                    if "shared_head.head" in name or "embed_tokens" in name:
                        continue

                    is_decoder = True
                    # For nextn specific weights
                    for weight_name in nextn_spec_weight_names:
                        if weight_name in name:
                            name = name.replace(nextn_layer_prefix, "model")
                            is_decoder = False
                            break
                    # For decoder layer weights
                    if is_decoder:
                        name = name.replace(nextn_layer_prefix, "model.decoder")

                if "rotary_emb.inv_freq" in name:
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if weight_name not in name:
                        continue
                    if _is_npu:
                        name = name.replace("weight_packed", "weight")
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    maybe_executor_submit(
                        executor=executor,
                        futures=futures,
                        use_async=use_async_loading,
                        func=weight_loader,
                        func_args=(param, loaded_weight, shard_id),
                    )
                    break
                else:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(
                                param,
                                loaded_weight,
                                name,
                            ),
                            func_kwargs={
                                "shard_id": shard_id,
                                "expert_id": expert_id,
                            },
                        )
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        # Skip loading embed_tokens if not first rank in pipeline parallelism
                        if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                            continue
                        # Skip loading norm if not last rank in pipeline parallelism
                        if ".norm." in name and not self.pp_group.is_last_rank:
                            continue
                        if fuse_qkv_a_proj and (
                            "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                        ):
                            cached_a_proj[name] = loaded_weight
                            q_a_proj_name = (
                                name
                                if "q_a_proj" in name
                                else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                            )
                            kv_a_proj_name = (
                                name
                                if "kv_a_proj_with_mqa" in name
                                else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                            )

                            # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                            if (
                                q_a_proj_name in cached_a_proj
                                and kv_a_proj_name in cached_a_proj
                            ):
                                q_a_proj_weight = cached_a_proj[q_a_proj_name]
                                kv_a_proj_weight = cached_a_proj[kv_a_proj_name]

                                if q_a_proj_weight.shape == torch.Size(
                                    []
                                ) and kv_a_proj_weight.shape == torch.Size([]):
                                    fused_weight = q_a_proj_weight
                                else:
                                    cat_dim = 0
                                    if self.quant_config is not None and (
                                        self.quant_config.get_name() == "awq"
                                        or self.quant_config.get_name() == "awq_marlin"
                                        or self.quant_config.get_name() == "moe_wna16"
                                    ):
                                        cat_dim = 1

                                    fused_weight = torch.cat(
                                        [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                                    )

                                param_name = (
                                    name.replace(
                                        "q_a_proj", "fused_qkv_a_proj_with_mqa"
                                    )
                                    if "q_a_proj" in name
                                    else name.replace(
                                        "kv_a_proj_with_mqa",
                                        "fused_qkv_a_proj_with_mqa",
                                    )
                                )
                                param = params_dict[param_name]

                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, fused_weight),
                                )
                                cached_a_proj.pop(q_a_proj_name)
                                cached_a_proj.pop(kv_a_proj_name)
                        else:
                            if (
                                "k_scale" in name or "v_scale" in name
                            ) and name not in params_dict:
                                # modelopt attn kv scale is named differently
                                for scale in ["k_scale", "v_scale"]:
                                    if scale in name:
                                        name = name.replace(
                                            f"{scale[0]}_proj", "attn_mqa"
                                        )
                                        break
                            if name not in params_dict:
                                # modelopt ckpt contains not needed weights for MTP module:
                                # model.decoder.self_attn.attn_mqa.v_scale and
                                # model.decoder.self_attn.attn_mqa.k_scale
                                logger.warning(f"{name} not found in params_dict.")
                                continue
                            param = params_dict[name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(param, loaded_weight),
                            )

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Mark the ue8m0 flag of nextn moe weights as True to avoid requantization
    def _mark_nextn_moe_weights_as_ue8m0(self):
        experts = self.model.decoder.mlp.experts
        w13_scale = (
            experts.w13_weight_scale_inv
            if hasattr(experts, "w13_weight_scale_inv")
            else experts.w13_weight_scale
        )
        w2_scale = (
            experts.w2_weight_scale_inv
            if hasattr(experts, "w2_weight_scale_inv")
            else experts.w2_weight_scale
        )
        w13_scale.format_ue8m0 = True
        w2_scale.format_ue8m0 = True

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, is_nextn=False
    ):
        # Quantize some weights to fp8 ue8m0 for DeepSeek nvfp4 checkpoint
        partial_names: List[str] = []
        nextn_layer_id = (
            0 if self.config.num_hidden_layers == 1 else self.config.num_hidden_layers
        )
        weights_dict = dict(weights)
        weight_block_size = [128, 128]

        if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
            layer_ids = (
                list(range(self.config.num_hidden_layers))
                if not is_nextn
                else [nextn_layer_id]
            )
            for layer_id in layer_ids:
                for stem in attn_quant_modules:
                    partial_names.append(f"model.layers.{layer_id}.self_attn.{stem}")

        if is_nextn and enable_nextn_moe_bf16_cast_to_fp8(self.quant_config):
            for expert_sub_name in [
                "shared_experts",
                *[
                    f"experts.{expert_id}"
                    for expert_id in range(self.config.n_routed_experts)
                ],
            ]:
                for stem in [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]:
                    partial_names.append(
                        f"model.layers.{nextn_layer_id}.mlp.{expert_sub_name}.{stem}"
                    )

        for partial_name in tqdm.tqdm(
            partial_names,
            desc="quant weights to fp8 ue8m0",
        ):
            original_weight = weights_dict[f"{partial_name}.weight"]
            out_w, out_s = quant_weight_ue8m0(
                original_weight, weight_block_size=weight_block_size
            )
            weights_dict[f"{partial_name}.weight"] = out_w
            weights_dict[f"{partial_name}.weight_scale_inv"] = out_s

        if is_nextn and enable_nextn_moe_bf16_cast_to_fp8(self.quant_config):
            self._mark_nextn_moe_weights_as_ue8m0()

        return list(weights_dict.items())
