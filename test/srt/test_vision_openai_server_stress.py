import gc
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import psutil
import torch

from test_vision_openai_server_common import *

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class MemoryMonitor:    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        self.monitoring = True
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        return {
            "cpu_memory": {
                "samples": len(self.memory_samples),
                "max_mb": max(self.memory_samples) if self.memory_samples else 0,
                "min_mb": min(self.memory_samples) if self.memory_samples else 0,
                "avg_mb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
                "final_mb": self.memory_samples[-1] if self.memory_samples else 0,
            },
            "gpu_memory": {
                "samples": len(self.gpu_memory_samples),
                "max_mb": max(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
                "min_mb": min(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
                "avg_mb": sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
                "final_mb": self.gpu_memory_samples[-1] if self.gpu_memory_samples else 0,
            }
        }
        
    def _monitor_loop(self):
        while self.monitoring:
            # CPU memory
            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(cpu_memory_mb)
            
            # GPU memory (if available)
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.gpu_memory_samples.append(gpu_memory_mb)
                
            time.sleep(self.monitor_interval)


class TestMultimodalStressTests(CustomTestCase):    
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
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "8",
                "--log-level", "debug",  # Enable debug logging for leak detection
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_sustained_vision_load(self):
        monitor = MemoryMonitor(monitor_interval=2.0)
        monitor.start_monitoring()
        
        try:
            import openai
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Run sustained load for 5 minutes
            duration_seconds = 300  # 5 minutes
            start_time = time.time()
            request_count = 0
            errors = []
            
            print(f"Starting sustained vision load test for {duration_seconds} seconds...")
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Vary the images to test different processing paths
                    image_url = IMAGE_MAN_IRONING_URL if request_count % 2 == 0 else IMAGE_SGL_LOGO_URL
                    
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
                                        "text": f"Describe this image briefly. Request {request_count}.",
                                    },
                                ],
                            },
                        ],
                        temperature=0.5,
                        max_tokens=100,
                    )
                    
                    # Basic validation
                    assert len(response.choices[0].message.content) > 10
                    request_count += 1
                    
                    # Log progress every 50 requests
                    if request_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = request_count / elapsed
                        print(f"Processed {request_count} requests in {elapsed:.1f}s (rate: {rate:.2f} req/s)")
                    
                    # Brief pause to avoid overwhelming the server
                    time.sleep(0.5)
                    
                except Exception as e:
                    errors.append(f"Request {request_count}: {str(e)}")
                    if len(errors) > 10:  # Stop if too many errors
                        break
                        
        finally:
            memory_stats = monitor.stop_monitoring()
            
        print(f"Sustained load test completed: {request_count} requests, {len(errors)} errors")
        
        # Analyze memory usage for leaks
        cpu_memory = memory_stats["cpu_memory"]
        gpu_memory = memory_stats["gpu_memory"]
        
        print(f"CPU Memory - Max: {cpu_memory['max_mb']:.1f}MB, Final: {cpu_memory['final_mb']:.1f}MB")
        print(f"GPU Memory - Max: {gpu_memory['max_mb']:.1f}MB, Final: {gpu_memory['final_mb']:.1f}MB")
        
        # Memory leak detection heuristics
        # 1. Final memory should not be significantly higher than average
        if cpu_memory['avg_mb'] > 0:
            cpu_growth_ratio = cpu_memory['final_mb'] / cpu_memory['avg_mb']
            assert cpu_growth_ratio < 2.0, f"Potential CPU memory leak detected: final/avg = {cpu_growth_ratio:.2f}"
            
        if gpu_memory['avg_mb'] > 0:
            gpu_growth_ratio = gpu_memory['final_mb'] / gpu_memory['avg_mb']
            assert gpu_growth_ratio < 1.5, f"Potential GPU memory leak detected: final/avg = {gpu_growth_ratio:.2f}"
        
        # 2. Error rate should be low
        error_rate = len(errors) / max(request_count, 1)
        assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%} ({len(errors)}/{request_count})"
        
        # 3. Should process reasonable number of requests
        assert request_count > 100, f"Too few requests processed: {request_count}"

    def test_concurrent_multimodal_stress(self):
        monitor = MemoryMonitor(monitor_interval=1.0)
        monitor.start_monitoring()
        
        try:
            import openai
            
            def make_vision_request(request_id):
                """Make a vision request."""
                client = openai.Client(api_key=self.api_key, base_url=self.base_url)
                
                image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL]
                image_url = image_urls[request_id % len(image_urls)]
                
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
                                    "text": f"Quick description. Request {request_id}.",
                                },
                            ],
                        },
                    ],
                    temperature=0.7,
                    max_tokens=50,
                )
                
                return {
                    "request_id": request_id,
                    "type": "vision",
                    "success": True,
                    "response_length": len(response.choices[0].message.content),
                }
            
            def make_multi_image_request(request_id):
                """Make a multi-image request to stress multimodal processing."""
                client = openai.Client(api_key=self.api_key, base_url=self.base_url)
                
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                                    "modalities": "multi-images",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_SGL_LOGO_URL},
                                    "modalities": "multi-images",
                                },
                                {
                                    "type": "text",
                                    "text": f"Compare these images. Request {request_id}.",
                                },
                            ],
                        },
                    ],
                    temperature=0.5,
                    max_tokens=100,
                )
                
                return {
                    "request_id": request_id,
                    "type": "multi-image",
                    "success": True,
                    "response_length": len(response.choices[0].message.content),
                }
            
            # Run multiple waves of concurrent requests
            all_results = []
            num_waves = 5
            requests_per_wave = 20
            
            print(f"Starting concurrent stress test: {num_waves} waves of {requests_per_wave} requests each")
            
            for wave in range(num_waves):
                print(f"Starting wave {wave + 1}/{num_waves}")
                
                with ThreadPoolExecutor(max_workers=requests_per_wave) as executor:
                    futures = []
                    
                    # Mix of single and multi-image requests
                    for i in range(requests_per_wave):
                        request_id = wave * requests_per_wave + i
                        if i % 3 == 0:  # Every 3rd request is multi-image
                            future = executor.submit(make_multi_image_request, request_id)
                        else:
                            future = executor.submit(make_vision_request, request_id)
                        futures.append(future)
                    
                    # Collect results
                    wave_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60)  # 60s timeout per request
                            wave_results.append(result)
                        except Exception as e:
                            wave_results.append({
                                "success": False,
                                "error": str(e),
                                "type": "unknown",
                            })
                    
                    all_results.extend(wave_results)
                    
                    # Brief pause between waves
                    time.sleep(5)
                    
                    # Force garbage collection between waves
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
        finally:
            memory_stats = monitor.stop_monitoring()
        
        # Analyze results
        successful_results = [r for r in all_results if r.get("success", False)]
        failed_results = [r for r in all_results if not r.get("success", False)]
        
        total_requests = len(all_results)
        success_rate = len(successful_results) / total_requests if total_requests > 0 else 0
        
        print(f"Concurrent stress test completed:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {len(failed_results)}")
        print(f"  Success rate: {success_rate:.2%}")
        
        # Memory analysis
        cpu_memory = memory_stats["cpu_memory"]
        gpu_memory = memory_stats["gpu_memory"]
        
        print(f"Memory usage during concurrent stress:")
        print(f"  CPU Memory - Max: {cpu_memory['max_mb']:.1f}MB, Final: {cpu_memory['final_mb']:.1f}MB")
        print(f"  GPU Memory - Max: {gpu_memory['max_mb']:.1f}MB, Final: {gpu_memory['final_mb']:.1f}MB")
        
        # Assertions
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2%}"
        assert len(successful_results) >= 80, f"Too few successful requests: {len(successful_results)}"
        
        # Memory leak checks 
        if cpu_memory['avg_mb'] > 0:
            cpu_growth_ratio = cpu_memory['final_mb'] / cpu_memory['avg_mb']
            assert cpu_growth_ratio < 2.5, f"Potential CPU memory leak in concurrent test: {cpu_growth_ratio:.2f}"
            
        if gpu_memory['avg_mb'] > 0:
            gpu_growth_ratio = gpu_memory['final_mb'] / gpu_memory['avg_mb']
            assert gpu_growth_ratio < 2.0, f"Potential GPU memory leak in concurrent test: {gpu_growth_ratio:.2f}"

    def test_repeated_operations_memory_stability(self):
        monitor = MemoryMonitor(monitor_interval=1.0)
        monitor.start_monitoring()
        
        try:
            import openai
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Test different operation types repeatedly
            operations = [
                {
                    "name": "single_image",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": IMAGE_SGL_LOGO_URL}},
                                {"type": "text", "text": "What is this?"},
                            ],
                        }
                    ],
                },
                {
                    "name": "multi_image",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                                    "modalities": "multi-images",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_SGL_LOGO_URL},
                                    "modalities": "multi-images",
                                },
                                {"type": "text", "text": "Compare these images."},
                            ],
                        }
                    ],
                },
                {
                    "name": "text_only",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello, how are you?"},
                            ],
                        }
                    ],
                },
            ]
            
            # Repeat each operation multiple times
            repetitions_per_operation = 50
            results = {op["name"]: [] for op in operations}
            
            print(f"Testing memory stability with {repetitions_per_operation} repetitions per operation type")
            
            for operation in operations:
                print(f"Testing operation: {operation['name']}")
                
                for i in range(repetitions_per_operation):
                    try:
                        response = client.chat.completions.create(
                            model="default",
                            messages=operation["messages"],
                            temperature=0.3,
                            max_tokens=100,
                        )
                        
                        results[operation["name"]].append({
                            "success": True,
                            "response_length": len(response.choices[0].message.content),
                            "tokens": response.usage.total_tokens,
                        })
                        
                        # Periodic garbage collection
                        if i % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        results[operation["name"]].append({
                            "success": False,
                            "error": str(e),
                        })
                        
                # Brief pause between operation types
                time.sleep(2)
                
        finally:
            memory_stats = monitor.stop_monitoring()
        
        # Analyze results and memory stability
        print("Memory stability test results:")
        
        for op_name, op_results in results.items():
            successful = len([r for r in op_results if r.get("success", False)])
            total = len(op_results)
            success_rate = successful / total if total > 0 else 0
            
            print(f"  {op_name}: {successful}/{total} successful ({success_rate:.1%})")
            assert success_rate >= 0.95, f"Operation {op_name} success rate too low: {success_rate:.1%}"
        
        # Memory stability analysis
        cpu_memory = memory_stats["cpu_memory"]
        gpu_memory = memory_stats["gpu_memory"]
        
        print(f"Memory stability analysis:")
        print(f"  CPU Memory - Max: {cpu_memory['max_mb']:.1f}MB, Min: {cpu_memory['min_mb']:.1f}MB, Final: {cpu_memory['final_mb']:.1f}MB")
        print(f"  GPU Memory - Max: {gpu_memory['max_mb']:.1f}MB, Min: {gpu_memory['min_mb']:.1f}MB, Final: {gpu_memory['final_mb']:.1f}MB")
        
        # Check memory stability (should not grow significantly over time)
        if cpu_memory['max_mb'] > 0 and cpu_memory['min_mb'] > 0:
            cpu_variance = (cpu_memory['max_mb'] - cpu_memory['min_mb']) / cpu_memory['max_mb']
            assert cpu_variance < 0.5, f"CPU memory variance too high: {cpu_variance:.2%}"
            
        if gpu_memory['max_mb'] > 0 and gpu_memory['min_mb'] > 0:
            gpu_variance = (gpu_memory['max_mb'] - gpu_memory['min_mb']) / gpu_memory['max_mb']
            assert gpu_variance < 0.3, f"GPU memory variance too high: {gpu_variance:.2%}"


if __name__ == "__main__":
    unittest.main()