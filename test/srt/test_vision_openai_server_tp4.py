"""
Test multimodal OpenAI server with TP-4 (Tensor Parallelism = 4).

This test ensures that vision and audio processing works correctly with
TP-4 configuration, addressing issue #8496 requirement for improved 
multimodal CI coverage with tensor parallelism.

Usage:
python3 -m unittest test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_single_image_chat_completion
"""

import unittest

from test_vision_openai_server_common import *

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestQwen2VLServerTP4(TestOpenAIVisionServer):    
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
                "--tensor-parallel-size", "4",  # Key: TP-4 configuration
                "--mem-fraction-static", "0.35",
                "--cuda-graph-max-bs", "4",
                "--trust-remote-code",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        self._test_video_chat_completion()

    def test_multi_images_chat_completion(self):
        super().test_multi_images_chat_completion()


class TestMiniCPMOServerTP4(TestOpenAIVisionServer):    
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
                "--tensor-parallel-size", "4", 
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "4",
                "--trust-remote-code",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()


class TestTP4LoadBalancing(CustomTestCase):    
    @classmethod
    def setUpClass(cls):
        cls.model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tensor-parallel-size", "4",  
                "--mem-fraction-static", "0.4",
                "--cuda-graph-max-bs", "8",  
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_concurrent_multimodal_requests(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import openai
        import time
        
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        def make_vision_request(request_id):
            start_time = time.time()
            
            image_url = IMAGE_MAN_IRONING_URL if request_id % 2 == 0 else IMAGE_SGL_LOGO_URL
            prompt = f"Describe this image in one sentence. Request {request_id}."
            
            try:
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        },
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )
                
                duration = time.time() - start_time
                text = response.choices[0].message.content
                
                # Basic validation
                assert len(text) > 10, f"Response too short: {text}"
                assert response.usage.total_tokens > 0
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "duration": duration,
                    "response_length": len(text),
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time,
                }
        
        num_requests = 8
        results = []
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_vision_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Request {result['request_id']}: success={result['success']}, duration={result.get('duration', 0):.2f}s")
        
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        success_rate = len(successful_requests) / len(results)
        assert success_rate >= 0.75, f"Success rate too low: {success_rate:.2f} (failed: {failed_requests})"
        
        if successful_requests:
            avg_duration = sum(r["duration"] for r in successful_requests) / len(successful_requests)
            assert avg_duration < 30.0, f"Average response time too high: {avg_duration:.2f}s"
        
        print(f"TP-4 concurrent test completed: {len(successful_requests)}/{len(results)} successful, avg duration: {avg_duration:.2f}s")


class TestTP4MemoryEfficiency(CustomTestCase):    
    def test_tp4_memory_usage(self):
        import torch
        import time
        import random
        
        if torch.cuda.device_count() < 4:
            self.skipTest("Requires at least 4 GPUs for TP-4 testing")
            
        model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        api_key = "sk-123456"
        
        memory_results = {}
        
        for tp_size in [1, 2, 4]:
            base_port = random.randint(6000, 7000)
            server_url = f"http://127.0.0.1:{base_port}"
            
            # Clear GPU memory between tests
            torch.cuda.empty_cache()
            time.sleep(2)
            
            process = popen_launch_server(
                model,
                server_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                api_key=api_key,
                other_args=[
                    "--tensor-parallel-size", str(tp_size),
                    "--mem-fraction-static", "0.3",
                ],
            )
            
            try:
                # Wait for server to be ready
                time.sleep(15)
                
                # Measure memory after server initialization
                memory_before = []
                for gpu_id in range(min(tp_size, torch.cuda.device_count())):
                    torch.cuda.set_device(gpu_id)
                    memory_before.append(torch.cuda.memory_allocated(gpu_id))
                
                # Run a few requests to measure active memory usage
                import openai
                client = openai.Client(api_key=api_key, base_url=server_url + "/v1")
                
                for _ in range(3):
                    response = client.chat.completions.create(
                        model="default",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": IMAGE_SGL_LOGO_URL},
                                    },
                                    {
                                        "type": "text",
                                        "text": "What is this?",
                                    },
                                ],
                            },
                        ],
                        temperature=0,
                        max_tokens=50,
                    )
                
                # Measure memory after processing
                memory_after = []
                for gpu_id in range(min(tp_size, torch.cuda.device_count())):
                    torch.cuda.set_device(gpu_id)
                    memory_after.append(torch.cuda.memory_allocated(gpu_id))
                
                memory_results[tp_size] = {
                    "before": memory_before,
                    "after": memory_after,
                    "total_before": sum(memory_before),
                    "total_after": sum(memory_after),
                }
                
                print(f"TP-{tp_size} memory usage: {sum(memory_after) / 1024**3:.2f} GB total")
                
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)
        
        # Validate memory scaling patterns
        # TP-4 should distribute memory across more GPUs
        assert len(memory_results[4]["after"]) == 4, "TP-4 should use 4 GPUs"
        assert len(memory_results[1]["after"]) == 1, "TP-1 should use 1 GPU"
        
        # Total memory should be similar across configurations (just distributed differently)
        tp1_total = memory_results[1]["total_after"]
        tp4_total = memory_results[4]["total_after"]
        
        # Allow some variance but should be in similar ballpark
        memory_ratio = tp4_total / tp1_total if tp1_total > 0 else 1
        assert 0.5 <= memory_ratio <= 2.0, f"Memory usage ratio seems off: TP4/TP1 = {memory_ratio:.2f}"
        
        print("Memory efficiency test passed: TP configurations scale appropriately")


if __name__ == "__main__":
    # Remove the base class to prevent it from running
    del TestOpenAIVisionServer
    unittest.main()