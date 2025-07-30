import multiprocessing
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_and_check_memory_leak,
)


class TestVisionMemoryLeak(CustomTestCase):
    def vlm_workload_func(self, base_url, model):
        def process_func():
            def run_one(_):
                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"},
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly.",
                                },
                            ],
                        }],
                        "model": "default",
                        "temperature": 0,
                        "max_tokens": 100,
                    },
                )

            with ThreadPoolExecutor(16) as executor:
                list(executor.map(run_one, list(range(16))))

        p = multiprocessing.Process(target=process_func)
        p.start()
        time.sleep(0.5)
        p.terminate()
        time.sleep(10)

    def test_memory_leak(self):
        run_and_check_memory_leak(
            self.vlm_workload_func,
            disable_radix_cache=False,
            enable_mixed_chunk=False,
            disable_overlap=False,
            chunked_prefill_size=8192,
            assert_has_abort=True,
            other_server_args=[
                "--enable-multimodal",
            ],
        )


if __name__ == "__main__":
    unittest.main()