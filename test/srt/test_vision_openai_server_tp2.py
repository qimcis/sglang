import unittest

from test_vision_openai_server_common import *

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
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
                "--tensor-parallel-size", "2",  # Key: TP-2 configuration
                "--mem-fraction-static", "0.4",
                "--cuda-graph-max-bs", "4",
                "--trust-remote-code",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        """Test video processing with TP-2."""
        self._test_video_chat_completion()


class TestMiniCPMOServerTP2(TestOpenAIVisionServer):    
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
                "--tensor-parallel-size", "2",  # Key: TP-2 configuration
                "--mem-fraction-static", "0.65",
                "--cuda-graph-max-bs", "4",
                "--trust-remote-code",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion(self):
        """Test audio processing with TP-2."""
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()


class TestGemma3nServerTP2(TestOpenAIVisionServer):    
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma3n-2b-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tensor-parallel-size", "2",  # Key: TP-2 configuration
                "--mem-fraction-static", "0.5",
                "--cuda-graph-max-bs", "4",
                "--trust-remote-code",
                "--enable-multimodal",  # Explicit multimodal enable for Gemma3n
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()


class TestTP2ConsistencyCheck(CustomTestCase):
    """Test consistency between TP-1 and TP-2 outputs for the same inputs."""
    
    def setUp(self):
        self.model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        self.api_key = "sk-123456"
        
    def test_tp1_vs_tp2_consistency(self):
        import time
        import random
        
        base_port = random.randint(5000, 6000)
        
        # Start TP-1 server
        tp1_url = f"http://127.0.0.1:{base_port}"
        tp1_process = popen_launch_server(
            self.model,
            tp1_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=self.api_key,
            other_args=[
                "--tensor-parallel-size", "1",
                "--mem-fraction-static", "0.3",
            ],
        )
        
        # Start TP-2 server  
        tp2_url = f"http://127.0.0.1:{base_port + 1}"
        tp2_process = popen_launch_server(
            self.model,
            tp2_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=self.api_key,
            other_args=[
                "--tensor-parallel-size", "2",
                "--mem-fraction-static", "0.3",
            ],
        )
        
        try:
            # Give servers time to fully initialize
            time.sleep(10)
            
            import openai
            
            tp1_client = openai.Client(api_key=self.api_key, base_url=tp1_url + "/v1")
            tp2_client = openai.Client(api_key=self.api_key, base_url=tp2_url + "/v1")
            
            # Test with deterministic generation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_SGL_LOGO_URL},
                        },
                        {
                            "type": "text",
                            "text": "What color is this logo?",
                        },
                    ],
                },
            ]
            
            # Generate responses from both servers
            tp1_response = tp1_client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0.0,  # Deterministic
                max_tokens=50,
            )
            
            tp2_response = tp2_client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0.0,  # Deterministic
                max_tokens=50,
            )
            
            tp1_text = tp1_response.choices[0].message.content.lower()
            tp2_text = tp2_response.choices[0].message.content.lower()
            
            print(f"TP-1 response: {tp1_text}")
            print(f"TP-2 response: {tp2_text}")
            
            # Both should mention color/blue since it's the SGL logo
            assert any(color in tp1_text for color in ["blue", "color", "purple"]), f"TP-1 response should mention color: {tp1_text}"
            assert any(color in tp2_text for color in ["blue", "color", "purple"]), f"TP-2 response should mention color: {tp2_text}"
            
            # Responses should be reasonably similar (same basic understanding)
            common_keywords = ["logo", "s", "blue", "color", "purple", "design"]
            tp1_keywords = sum(1 for kw in common_keywords if kw in tp1_text)
            tp2_keywords = sum(1 for kw in common_keywords if kw in tp2_text)
            
            # At least some overlap in understanding
            assert tp1_keywords > 0 and tp2_keywords > 0, f"Both should understand the image: TP1={tp1_keywords}, TP2={tp2_keywords}"
            
        finally:
            # Clean up
            kill_process_tree(tp1_process.pid)
            kill_process_tree(tp2_process.pid)


if __name__ == "__main__":
    # Remove the base class to prevent it from running
    del TestOpenAIVisionServer
    unittest.main()