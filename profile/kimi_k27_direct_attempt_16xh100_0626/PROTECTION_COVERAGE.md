# Kimi KV Protection Coverage

## Kimi K2.7

Kimi K2.7 uses `KimiK25ForConditionalGeneration`, whose language model is
`DeepseekV3ForCausalLM`. SGLang classifies this decoder as dense MLA, so its
absorbed decode path consumes the ordinary paged MLA table through FA3 rather
than a DSA top-k table.

The FA3 page-protection implementation covers this path on Hopper when all of
the following are true:

- The decode attention backend is `fa3`.
- `SGLANG_KV_PAGE_PROTECTION=1` is set on the PD decode worker.
- Speculative decoding, pipeline parallelism, DP attention, and shared radix
  prefix pages are disabled, as required by the current protection checks.
- The sgl-kernel FA3 implementation containing the protection patch is loaded.

The validation runs in the FA3 consumer before it reads the paged MLA cache. It
checks page bounds, request ownership, logical page position, allocation
generation, attention tag, and transfer tag. Invalid requests leave their
output untouched and are aborted by the existing fused-status path; other
requests in the batch continue.

The recorded TP8-per-side attempt is not runtime proof of this path. Protection
was disabled, the normalized server arguments selected `flashinfer` for
language attention, and both workers OOMed during weight allocation before
attention backend initialization. A protected rerun must explicitly select
`--attention-backend fa3`. Based on the failed capacity test, the unquantized
model is expected to require TP16 per side, or 32 H100-class GPUs for one
prefill and one decode worker.

### H200 Kernel Validation

On 2026-07-18, the patched FA3 extension was built and tested on one NVIDIA
H200 (compute capability 9.0, CUDA 13.0). The validation build was deliberately
limited to the model's published BF16 absorbed-MLA shape:

- `qk_rope_head_dim=64`
- `kv_lora_rank=512`, which is the FA3 `qv` and V width

All eight focused FA3 protection tests passed. They cover healthy protected
attention, page sizes 1 and 64, split and non-split fail-closed execution,
CUDA-graph healthy and injected-failure replay, graph padding slot 0, malformed
partial descriptors, and unsupported cache-batch remapping. The two FA version
dispatch tests and 22 protection gating/abort tests also passed.

This validates the Kimi kernel specialization, not a full production wheel or
an end-to-end Kimi server. FP16, FP8, unrelated head dimensions, the complete
wheel build, and TP16-per-side model startup remain outside this focused run.

## Kimi K3 Boundary

Protection should follow cache semantics rather than the model name:

- Dense, gated, or full MLA layers can reuse paged FA3 page protection without
  a Kimi-specific protection kernel.
- KDA layers cannot use KV page tags for their recurrent convolution and
  temporal states. They need request-slot ownership, generation, and state-byte
  integrity checks at the KDA consumer.
- Cached KDA prefill checkpoints need the same lifecycle checks on
  `MambaCheckpointPool` slots and transfer payloads.
- AttnRes-specific state cannot be implemented safely until its serving layout
  and lifetime are available. If it is paged KV, it can use the common page
  descriptor; if it is recurrent or depth state, it needs the corresponding
  state-slot descriptor.

Kimi K3 therefore should not receive one monolithic model-specific protection
implementation. Each backend and state pool should declare its protection
capability, and startup should fail when protection is requested for an
uncovered state type.
