import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestQwen2VLServerTP2(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tp-size", "2",
                "--mem-fraction-static", "0.35",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        self._test_video_chat_completion()


class TestQwen2_5_VLServerTP2(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tp-size", "2",
                "--mem-fraction-static", "0.35",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        self._test_video_chat_completion()


class TestMinicpmvServerTP2(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-V-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tp-size", "2",
                "--trust-remote-code",
                "--mem-fraction-static", "0.35",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
    unittest.main()