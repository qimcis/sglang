# Dynamic Batching

Dynamic batching is an opt-in mode for `sglang.multimodal_gen` that merges compatible queued diffusion requests into one native pipeline batch. It is separate from LLM continuous batching and tokenizer batching.

Use it for concurrent T2I or T2V traffic with the same model and sampling shape. Keep singleton serving for latency-sensitive or highly mixed traffic.

## Enable

Dynamic batching is disabled by default with `--batching-max-size 1`.

```bash
sglang serve \
  --model-path black-forest-labs/FLUX.1-dev \
  --port 30010 \
  --batching-mode dynamic \
  --batching-max-size 8 \
  --batching-delay-ms 5 \
  --enable-batching-metrics
```

For request formats, see the [OpenAI-Compatible API](../api/openai_api.md).

Use `--batching-config /path/to/batching_config.json` to load JSON rules when a model or resolution needs a lower cap than `--batching-max-size`:

```json
{
  "schema_version": 1,
  "rules": [
    {
      "model_contains": "Qwen-Image",
      "resolution": "1024x1024",
      "max_batch_size": 1
    }
  ]
}
```

## Compatibility

An initial implementation of dynamic batching for T2I and T2V models can be found in [#18764](https://github.com/sgl-project/sglang/pull/18764). The current compatibility grid is below and will be updated as more coverage is added. See [Supported Models](../compatibility_matrix.md) for full model IDs.

`✅` means supported, `❌` means not currently supported, `?` means untested, and `-` means not applicable.

### Image

| Model | T2I | I2I |
|---|:---:|:---:|
| FLUX.1-dev | ✅ | - |
| FLUX.2-dev | ✅ | ❌ |
| FLUX.2-dev-NVFP4 | ? | ? |
| FLUX.2-Klein-4B | ✅ | ❌ |
| FLUX.2-Klein-9B | ? | ? |
| Z-Image | ? | - |
| Z-Image-Turbo | ❌ | - |
| GLM-Image | ❌ | - |
| Qwen Image | ✅ | - |
| Qwen Image 2512 | ✅ | - |
| Qwen Image Edit | - | ❌ |
| Qwen Image Edit 2509 | - | ? |
| Qwen Image Edit 2511 | - | ? |
| Qwen Image Layered | ? | ? |
| SD3 Medium | ? | - |
| SD3.5 Medium | ? | - |
| SD3.5 Large | ? | - |
| Hunyuan3D-2 | ? | - |
| SANA 1.5 1.6B | ? | - |
| SANA 1.5 4.8B | ? | - |
| SANA 1600M 1024px | ? | - |
| SANA 600M 1024px | ? | - |
| SANA 1600M 512px | ? | - |
| SANA 600M 512px | ? | - |
| FireRed-Image-Edit 1.0 | - | ? |
| FireRed-Image-Edit 1.1 | - | ? |
| ERNIE-Image | ? | - |
| ERNIE-Image-Turbo | ? | - |

### Video

| Model | Support |
|---|:---:|
| FastWan2.1 T2V 1.3B | ✅ |
| FastWan2.2 TI2V 5B Full Attn | ❌ |
| Wan2.2 TI2V 5B | ❌ |
| Wan2.2 T2V A14B | ✅ |
| Wan2.2 I2V A14B | ❌ |
| HunyuanVideo | ❌ |
| FastHunyuan | ❌ |
| Wan2.1 T2V 1.3B | ✅ |
| Wan2.1 T2V 14B | ✅ |
| Wan2.1 I2V 480P | ? |
| Wan2.1 I2V 720P | ? |
| TurboWan2.1 T2V 1.3B | ✅ |
| TurboWan2.1 T2V 14B | ✅ |
| TurboWan2.1 T2V 14B 720P | ✅ |
| TurboWan2.2 I2V A14B | ? |
| Wan2.1 Fun 1.3B InP | ? |
| Helios Base | ? |
| Helios Mid | ? |
| Helios Distilled | ? |
| LTX-2 | ? |
| LTX-2.3 | ? |

## Notes

- Requests batch only when model inputs, sampling parameters, output handling, and any configured rules are compatible.
- There is no startup probing, runtime learning, OOM retry, or automatic fallback to singletons. If a merged batch fails or cannot be split, every request in that batch receives an error.
- Batch shape can change kernels, so singleton and dynamic outputs are not expected to be bit-exact.
- Use `--enable-batching-metrics` to inspect realized batches:

```text
Dynamic batch dispatch: size=2/8, queue_wait=5.12ms, stop_reason=delay
Dynamic batch dispatch: size=1/8, queue_wait=0.04ms, stop_reason=config_cap:1
Dynamic batch stats (last 5 dispatches): avg_size=2.80, merged_rate=60.0%, full_rate=20.0%, utilization=35.0%, wait_avg=3.21ms, wait_p95=5.12ms, top_rejects=none
```
