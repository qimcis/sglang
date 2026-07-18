"""Unit tests for FlashAttention version dispatch."""

import unittest
from unittest.mock import patch

from sglang.jit_kernel import flash_attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashAttentionDispatch(CustomTestCase):
    @patch.object(flash_attention, "fa3_flash_attn_with_kvcache")
    def test_fa3_forwards_kv_page_protection(self, mock_fa3):
        protection = {"request_indices": object()}
        expected = object()
        mock_fa3.return_value = expected

        result = flash_attention.flash_attn_with_kvcache(
            None,
            None,
            None,
            ver=3,
            kv_page_protection=protection,
        )

        self.assertIs(result, expected)
        self.assertIs(mock_fa3.call_args.kwargs["kv_page_protection"], protection)

    def test_fa4_rejects_kv_page_protection(self):
        with self.assertRaisesRegex(RuntimeError, "supported only by FA3"):
            flash_attention.flash_attn_with_kvcache(
                None,
                None,
                None,
                ver=4,
                kv_page_protection={"request_indices": object()},
            )


if __name__ == "__main__":
    unittest.main()
