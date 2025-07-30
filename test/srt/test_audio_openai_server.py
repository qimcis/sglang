import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestMiniCPMOAudioServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.65",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()


class TestPhi4MMAudioServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import constants, snapshot_download

        snapshot_download(
            "microsoft/Phi-4-multimodal-instruct",
            allow_patterns=["**/adapter_config.json"],
        )

        cls.model = "microsoft/Phi-4-multimodal-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.70",
                "--disable-radix-cache",
                "--max-loras-per-batch", "2",
                "--revision", revision,
                "--lora-paths",
                f"vision={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/vision-lora",
                f"speech={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/speech-lora",
            ],
        )
        cls.base_url += "/v1"

    def get_audio_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "speech",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()


if __name__ == "__main__":
    unittest.main()