import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    extract_transfer_fields,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.models.dits.glm_image import (
    GlmImageAttention,
    GlmImageLayerKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageBeforeDenoisingStage,
    _PerRowLogitsSampler,
)
from sglang.test.test_utils import CustomTestCase


class _BatchEncoding(dict):
    def to(self, device):
        return self


class _GlyphTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        token_ids = {
            "": [9],
            "A": [1, 2],
            "LONG": [3, 4, 5, 6],
        }
        return SimpleNamespace(input_ids=[token_ids[text] for text in texts])


class _GlyphEncoder:
    dtype = torch.float32

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))


class _SchedulerConfig(dict):
    num_train_timesteps = 1000


class _Scheduler:
    def __init__(self):
        self.config = _SchedulerConfig()
        self.timesteps = None

    def set_timesteps(self, timesteps=None, sigmas=None, device=None, **kwargs):
        self.timesteps = torch.as_tensor(timesteps, device=device)


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.config = SimpleNamespace(in_channels=2, patch_size=2, num_layers=1)


class _TupleIdentity(torch.nn.Module):
    def forward(self, value):
        return value, None


class _CaptureAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask = None
        self.num_replicated_prefix = None

    def forward(self, query, key, value, attn_mask=None, num_replicated_prefix=0):
        self.attn_mask = attn_mask
        self.num_replicated_prefix = num_replicated_prefix
        return query


class TestGlmImageDynamicBatching(CustomTestCase):
    def test_config_and_sampling_defaults_enable_t2i_batching(self):
        self.assertTrue(GlmImagePipelineConfig().supports_dynamic_batching())
        sampling_params = GlmImageSamplingParams()
        self.assertEqual(sampling_params.negative_prompt, "")
        self.assertEqual((sampling_params.height, sampling_params.width), (1024, 1024))

    def test_scheduler_merges_text_requests_but_not_image_requests(self):
        scheduler = Scheduler.__new__(Scheduler)
        first = Req(sampling_params=GlmImageSamplingParams(prompt="first", seed=10))
        second = Req(sampling_params=GlmImageSamplingParams(prompt="second", seed=20))

        merged = scheduler._try_merge_generation_reqs([first, second])

        self.assertEqual(merged.prompt, ["first", "second"])
        self.assertEqual(merged.extra["dynamic_batch_seeds"], [10, 20])
        image_request = Req(
            sampling_params=GlmImageSamplingParams(
                prompt="edit", image_path="image.png"
            )
        )
        self.assertFalse(scheduler._can_dynamic_batch(first, image_request))

    def test_dynamic_batch_preserves_explicit_multi_output_seeds(self):
        batch = Req(
            sampling_params=GlmImageSamplingParams(
                prompt=["first", "second"],
                num_outputs_per_prompt=2,
                generator_device="cpu",
            )
        )
        batch.extra["dynamic_batch_seeds"] = [[10, 11], [20, 21]]
        stage = InputValidationStage.__new__(InputValidationStage)

        stage._generate_seeds(
            batch,
            SimpleNamespace(pipeline_config=SimpleNamespace(generator_device=None)),
        )

        self.assertEqual(batch.seeds, [10, 11, 20, 21])
        self.assertEqual(
            [generator.initial_seed() for generator in batch.generator],
            [10, 11, 20, 21],
        )

    def test_disaggregation_transfers_glm_conditioning_tensors(self):
        batch = Req(sampling_params=GlmImageSamplingParams(prompt="test"))
        batch.prior_token_id = torch.tensor([[1, 2]])
        batch.prior_token_drop_cond = torch.tensor([[False, False]])
        batch.prior_token_drop_uncond = torch.tensor([[True, True]])
        batch.target_size = torch.tensor([[1024, 1024]])
        batch.crop_coords = torch.tensor([[0, 0]])
        batch.kv_caches = object()
        batch.glm_image_kv_cache_keys = [torch.ones((1, 2, 1, 1))]
        batch.glm_image_kv_cache_values = [torch.full((1, 2, 1, 1), 2.0)]

        tensor_fields, scalar_fields = extract_transfer_fields(batch)

        for field_name in (
            "prior_token_id",
            "prior_token_drop_cond",
            "prior_token_drop_uncond",
            "target_size",
            "crop_coords",
            "glm_image_kv_cache_keys",
            "glm_image_kv_cache_values",
        ):
            self.assertIn(field_name, tensor_fields)
        self.assertNotIn("kv_caches", tensor_fields)
        self.assertNotIn("kv_caches", scalar_fields)

    def test_disaggregation_rebuilds_glm_kv_cache(self):
        batch = Req(sampling_params=GlmImageSamplingParams(prompt="test"))
        key = torch.ones((1, 2, 1, 1))
        value = torch.full((1, 2, 1, 1), 2.0)
        batch.glm_image_kv_cache_keys = [key]
        batch.glm_image_kv_cache_values = [value]

        kv_caches = GlmImagePipelineConfig._get_kv_caches(batch)

        self.assertIs(batch.kv_caches, kv_caches)
        self.assertIs(kv_caches[0].k_cache, key)
        self.assertIs(kv_caches[0].v_cache, value)

    def test_ar_logits_sampler_uses_one_generator_per_row(self):
        generators = [
            torch.Generator().manual_seed(10),
            torch.Generator().manual_seed(20),
        ]
        expected_generators = [
            torch.Generator().manual_seed(10),
            torch.Generator().manual_seed(20),
        ]
        scores = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        warped_scores = scores / 0.5
        top_k_threshold = torch.topk(warped_scores, 2, dim=-1).values[:, -1:]
        warped_scores = warped_scores.masked_fill(
            warped_scores < top_k_threshold, float("-inf")
        )
        expected = [
            torch.multinomial(
                torch.softmax(warped_scores[row], dim=-1),
                1,
                generator=expected_generators[row],
            ).item()
            for row in range(2)
        ]

        sampler = _PerRowLogitsSampler(
            generators,
            SimpleNamespace(temperature=0.5, top_k=2, top_p=1.0),
        )
        forced_scores = sampler(torch.zeros((2, 1), dtype=torch.long), scores)

        self.assertEqual(forced_scores.argmax(dim=-1).tolist(), expected)
        self.assertEqual(len(sampler.warpers), 2)

    def test_ar_generates_and_extracts_every_prompt_row(self):
        inputs = _BatchEncoding(
            input_ids=torch.zeros((2, 3), dtype=torch.long),
            image_grid_thw=torch.tensor(
                [
                    [1, 2, 2],
                    [1, 1, 1],
                    [1, 2, 2],
                    [1, 1, 1],
                ]
            ),
            images_per_sample=torch.tensor([2, 2]),
        )
        processor = MagicMock()
        processor.apply_chat_template.return_value = inputs
        generated = torch.tensor(
            [
                [0, 0, 0, 99, 1, 2, 3, 4, 100],
                [0, 0, 0, 99, 11, 12, 13, 14, 100],
            ]
        )
        vision_language_encoder = SimpleNamespace(
            device=torch.device("cpu"),
            generate=MagicMock(return_value=generated),
        )
        stage = GlmImageAR.__new__(GlmImageAR)
        stage.processor = processor
        stage.vision_language_encoder = vision_language_encoder

        prior_token_ids, image_token_ids = stage.generate_prior_tokens(
            ["first", "second"],
            height=64,
            width=64,
            generator=[
                torch.Generator().manual_seed(1),
                torch.Generator().manual_seed(2),
            ],
        )

        self.assertIsNone(image_token_ids)
        self.assertEqual(prior_token_ids.shape, (2, 16))
        self.assertEqual(set(prior_token_ids[0].tolist()), {1, 2, 3, 4})
        self.assertEqual(set(prior_token_ids[1].tolist()), {11, 12, 13, 14})
        call = processor.apply_chat_template.call_args
        self.assertEqual(len(call.args[0]), 2)
        self.assertTrue(call.kwargs["padding"])
        logits_sampler = vision_language_encoder.generate.call_args.kwargs[
            "logits_processor"
        ][0]
        self.assertEqual(
            [generator.initial_seed() for generator in logits_sampler.generators],
            [1, 2],
        )

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_ar_normalizes_dimensions_for_sequence_parallelism(self, _):
        stage = GlmImageAR.__new__(GlmImageAR)
        stage.generate_prior_tokens = MagicMock(
            return_value=(torch.zeros((2, 4), dtype=torch.long), None)
        )
        params = GlmImageSamplingParams(
            prompt=["first", "second"],
            height=1000,
            width=1001,
            num_outputs_per_prompt=1,
        )
        batch = Req(sampling_params=params)
        batch.generator = [
            torch.Generator().manual_seed(1),
            torch.Generator().manual_seed(2),
        ]

        result = stage.forward(batch, SimpleNamespace(sp_degree=4))

        self.assertEqual((result.height, result.width), (960, 992))
        self.assertEqual(stage.generate_prior_tokens.call_args.kwargs["factor"], 64)
        self.assertEqual(
            [
                generator.initial_seed()
                for generator in stage.generate_prior_tokens.call_args.kwargs[
                    "generator"
                ]
            ],
            [1, 2],
        )

    def test_ar_preserves_single_request_processor_shape(self):
        inputs = _BatchEncoding(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            image_grid_thw=torch.tensor([[1, 2, 2], [1, 1, 1]]),
        )
        processor = MagicMock()
        processor.apply_chat_template.return_value = inputs
        generated = torch.tensor([[0, 0, 0, 99, 1, 2, 3, 4, 100]])
        vision_language_encoder = SimpleNamespace(
            device=torch.device("cpu"),
            generate=MagicMock(return_value=generated),
        )
        stage = GlmImageAR.__new__(GlmImageAR)
        stage.processor = processor
        stage.vision_language_encoder = vision_language_encoder

        stage.generate_prior_tokens("single", height=64, width=64)

        call = processor.apply_chat_template.call_args
        self.assertIsInstance(call.args[0], list)
        self.assertIsInstance(call.args[0][0], dict)
        self.assertFalse(call.kwargs["padding"])

    def test_glyph_embeddings_preserve_rows_and_mask_padding(self):
        stage = GlmImageBeforeDenoisingStage.__new__(GlmImageBeforeDenoisingStage)
        stage.tokenizer = _GlyphTokenizer()
        stage.text_encoder = _GlyphEncoder()

        embeds, attention_mask = stage._get_glyph_embeds(
            ["a sign saying 'A'", "a sign saying 'LONG'"],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(embeds.shape, (2, 4, 1))
        self.assertTrue(
            torch.equal(embeds[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 2.0]))
        )
        self.assertTrue(
            torch.equal(embeds[1, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
        )
        self.assertTrue(
            torch.equal(
                attention_mask,
                torch.tensor([[False, False, True, True], [True, True, True, True]]),
            )
        )

    def test_encode_prompt_repeats_in_request_major_order(self):
        stage = GlmImageBeforeDenoisingStage.__new__(GlmImageBeforeDenoisingStage)
        positive = torch.tensor([[[1.0]], [[2.0]]])
        positive_mask = torch.tensor([[False], [True]])
        negative = torch.tensor([[[3.0]], [[4.0]]])
        stage._get_glyph_embeds = MagicMock(
            side_effect=[
                (positive, positive_mask),
                (negative, None),
            ]
        )

        pos, neg, pos_mask, neg_mask = stage.encode_prompt(
            ["first", "second"],
            num_images_per_prompt=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(pos[:, 0, 0].tolist(), [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(neg[:, 0, 0].tolist(), [3.0, 3.0, 4.0, 4.0])
        self.assertEqual(pos_mask[:, 0].tolist(), [False, False, True, True])
        self.assertIsNone(neg_mask)

    @patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image.get_local_torch_device",
        return_value=torch.device("cpu"),
    )
    def test_before_denoising_uses_effective_batch_size(self, _):
        stage = GlmImageBeforeDenoisingStage.__new__(GlmImageBeforeDenoisingStage)
        stage.transformer = _Transformer()
        stage.scheduler = _Scheduler()
        stage.vae_scale_factor = 8
        stage.vae = SimpleNamespace()
        prompt_embeds = torch.zeros((4, 2, 3))
        negative_prompt_embeds = torch.zeros((4, 1, 3))
        prompt_mask = torch.tensor(
            [[False, True], [False, True], [True, True], [True, True]]
        )
        stage.encode_prompt = MagicMock(
            return_value=(
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                None,
            )
        )
        stage.prepare_latents = MagicMock(return_value=torch.zeros((4, 2, 4, 4)))

        params = GlmImageSamplingParams(
            prompt=["first", "second"],
            height=32,
            width=32,
            num_outputs_per_prompt=2,
            generator_device="cpu",
        )
        batch = Req(sampling_params=params)
        batch.generator = [torch.Generator().manual_seed(seed) for seed in range(4)]
        batch.prior_token_id = torch.tensor([[1, 2], [3, 4]])
        batch.prior_token_image_ids = None

        result = stage.forward(batch, SimpleNamespace())

        self.assertEqual(stage.prepare_latents.call_args.kwargs["batch_size"], 4)
        self.assertIs(
            stage.prepare_latents.call_args.kwargs["generator"], batch.generator
        )
        self.assertEqual(
            result.prior_token_id.tolist(),
            [[1, 2], [1, 2], [3, 4], [3, 4]],
        )
        self.assertEqual(result.target_size.shape, (4, 2))
        self.assertEqual(result.crop_coords.shape, (4, 2))
        self.assertTrue(torch.equal(result.prompt_attention_mask[0], prompt_mask))
        self.assertEqual(result.latents.shape[0], 4)

    def test_glm_attention_extends_text_mask_over_image_tokens(self):
        stage = GlmImageAttention.__new__(GlmImageAttention)
        torch.nn.Module.__init__(stage)
        stage.num_local_heads = 1
        stage.num_local_kv_heads = 1
        stage.to_q = _TupleIdentity()
        stage.to_k = _TupleIdentity()
        stage.to_v = _TupleIdentity()
        stage.norm_q = None
        stage.norm_k = None
        stage.attn = _CaptureAttention()
        stage.to_out = torch.nn.ModuleList([_TupleIdentity()])

        hidden_states = torch.zeros((2, 3, 2))
        encoder_hidden_states = torch.zeros((2, 4, 2))
        text_mask = torch.tensor([[False, False, True, True], [True, True, True, True]])

        stage(
            hidden_states,
            encoder_hidden_states,
            attention_mask=text_mask,
        )

        expected = torch.cat([text_mask, torch.ones((2, 3), dtype=torch.bool)], dim=1)
        self.assertTrue(torch.equal(stage.attn.attn_mask, expected))
        self.assertEqual(stage.attn.num_replicated_prefix, 4)

    def test_glm_attention_mask_includes_cached_image_tokens(self):
        stage = GlmImageAttention.__new__(GlmImageAttention)
        torch.nn.Module.__init__(stage)
        stage.num_local_heads = 1
        stage.num_local_kv_heads = 1
        stage.to_q = _TupleIdentity()
        stage.to_k = _TupleIdentity()
        stage.to_v = _TupleIdentity()
        stage.norm_q = None
        stage.norm_k = None
        stage.attn = _CaptureAttention()
        stage.to_out = torch.nn.ModuleList([_TupleIdentity()])
        cache = GlmImageLayerKVCache()
        cache.store(torch.zeros((1, 2, 1, 2)), torch.zeros((1, 2, 1, 2)))
        cache.mode = "read"
        text_mask = torch.tensor([[False, False, True, True], [True, True, True, True]])

        stage(
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 4, 2)),
            attention_mask=text_mask,
            kv_cache=cache,
        )

        expected = torch.cat(
            [
                torch.ones((2, 2), dtype=torch.bool),
                text_mask,
                torch.ones((2, 3), dtype=torch.bool),
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(stage.attn.attn_mask, expected))

    def test_image_kv_cache_broadcasts_across_multiple_outputs(self):
        cache = GlmImageLayerKVCache()
        cache.store(torch.ones((1, 2, 1, 1)), torch.full((1, 2, 1, 1), 2.0))
        key = torch.zeros((3, 4, 1, 1))
        value = torch.zeros((3, 4, 1, 1))

        combined_key, combined_value = cache.get(key, value)

        self.assertEqual(combined_key.shape, (3, 6, 1, 1))
        self.assertEqual(combined_value.shape, (3, 6, 1, 1))
        self.assertTrue(torch.equal(combined_key[:, :2], torch.ones((3, 2, 1, 1))))
        self.assertTrue(
            torch.equal(combined_value[:, :2], torch.full((3, 2, 1, 1), 2.0))
        )

    def test_empty_image_kv_cache_returns_current_key_value(self):
        cache = GlmImageLayerKVCache()
        key = torch.zeros((2, 4, 1, 1))
        value = torch.ones((2, 4, 1, 1))

        combined_key, combined_value = cache.get(key, value)

        self.assertIs(combined_key, key)
        self.assertIs(combined_value, value)


if __name__ == "__main__":
    unittest.main()
