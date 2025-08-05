
import gc
import os
import threading
import time
from typing import Dict, List, Optional, Any
import requests

import psutil
import torch


class MemoryMonitor:    
    def __init__(self, monitor_interval: float = 1.0, detailed_gpu_tracking: bool = True):
        self.monitor_interval = monitor_interval
        self.detailed_gpu_tracking = detailed_gpu_tracking
        self.monitoring = False
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        self.monitoring = True
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, Any]:
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        cpu_stats = self._analyze_memory_samples(self.memory_samples)
        gpu_stats = self._analyze_memory_samples(self.gpu_memory_samples)
        
        return {
            "duration_seconds": duration,
            "cpu_memory": cpu_stats,
            "gpu_memory": gpu_stats,
            "leak_detected": self._detect_memory_leak(),
            "peak_usage": {
                "cpu_mb": max(self.memory_samples) if self.memory_samples else 0,
                "gpu_mb": max(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            }
        }
        
    def _monitor_loop(self):
        while self.monitoring:
            try:
                # CPU memory monitoring
                process = psutil.Process()
                cpu_memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(cpu_memory_mb)
                
                # GPU memory monitoring
                if torch.cuda.is_available() and self.detailed_gpu_tracking:
                    total_gpu_memory = 0
                    for gpu_id in range(torch.cuda.device_count()):
                        try:
                            torch.cuda.set_device(gpu_id)
                            gpu_memory = torch.cuda.memory_allocated(gpu_id) / 1024 / 1024
                            total_gpu_memory += gpu_memory
                        except Exception:
                            pass  # Skip if GPU not accessible
                    
                    self.gpu_memory_samples.append(total_gpu_memory)
                    
            except Exception as e:
                # Continue monitoring even if individual sample fails
                pass
                
            time.sleep(self.monitor_interval)
    
    def _analyze_memory_samples(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {
                "samples": 0, "max_mb": 0, "min_mb": 0, "avg_mb": 0, 
                "final_mb": 0, "initial_mb": 0, "growth_mb": 0, "growth_rate": 0
            }
            
        return {
            "samples": len(samples),
            "max_mb": max(samples),
            "min_mb": min(samples),
            "avg_mb": sum(samples) / len(samples),
            "final_mb": samples[-1],
            "initial_mb": samples[0],
            "growth_mb": samples[-1] - samples[0],
            "growth_rate": (samples[-1] - samples[0]) / samples[0] if samples[0] > 0 else 0,
        }
    
    def _detect_memory_leak(self) -> Dict[str, bool]:
        cpu_leak = False
        gpu_leak = False
        
        if len(self.memory_samples) > 10:
            # Check if memory consistently grows over time
            initial_samples = self.memory_samples[:5]
            final_samples = self.memory_samples[-5:]
            
            avg_initial = sum(initial_samples) / len(initial_samples)
            avg_final = sum(final_samples) / len(final_samples)
            
            # Consider it a leak if final is >20% higher than initial
            if avg_final > avg_initial * 1.2:
                cpu_leak = True
        
        if len(self.gpu_memory_samples) > 10:
            # Similar check for GPU memory
            initial_samples = self.gpu_memory_samples[:5]
            final_samples = self.gpu_memory_samples[-5:]
            
            avg_initial = sum(initial_samples) / len(initial_samples)
            avg_final = sum(final_samples) / len(final_samples)
            
            if avg_final > avg_initial * 1.15:  # Stricter threshold for GPU
                gpu_leak = True
        
        return {"cpu": cpu_leak, "gpu": gpu_leak}


class MultimodalTestConfig:    
    # Default test URLs
    DEFAULT_IMAGES = {
        "man_ironing": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png",
        "sgl_logo": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png",
    }
    
    DEFAULT_VIDEOS = {
        "jobs_ipod": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/videos/jobs_presenting_ipod.mp4",
    }
    
    DEFAULT_AUDIOS = {
        "trump_speech": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3",
        "bird_song": "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3",
    }
    
    # Model configurations for different test scenarios
    MODEL_CONFIGS = {
        "vision_only": {
            "models": [
                "lmms-lab/llava-onevision-qwen2-0.5b-ov",
                "Qwen/Qwen2-VL-7B-Instruct",
                "openbmb/MiniCPM-V-2_6",
            ],
            "modalities": ["image", "video"],
        },
        "audio_capable": {
            "models": [
                "openbmb/MiniCPM-o-2_6",
                "Qwen/Qwen2-Audio-7B-Instruct", 
                "google/gemma3n-2b-it",
            ],
            "modalities": ["image", "audio", "multimodal"],
        },
        "lightweight": {
            "models": [
                "lmms-lab/llava-onevision-qwen2-0.5b-ov",
            ],
            "modalities": ["image"],
        },
    }
    
    # Tensor parallelism configurations
    TP_CONFIGS = {
        "tp1": {"tp_size": 1, "mem_fraction": 0.5},
        "tp2": {"tp_size": 2, "mem_fraction": 0.4},
        "tp4": {"tp_size": 4, "mem_fraction": 0.35},
    }
    
    @classmethod
    def get_server_args_for_tp(cls, tp_size: int, model_specific_args: Optional[List[str]] = None) -> List[str]:
        """Get server arguments for tensor parallelism configuration."""
        config = cls.TP_CONFIGS.get(f"tp{tp_size}", cls.TP_CONFIGS["tp1"])
        
        args = [
            "--tensor-parallel-size", str(config["tp_size"]),
            "--mem-fraction-static", str(config["mem_fraction"]),
            "--cuda-graph-max-bs", "4",
        ]
        
        if model_specific_args:
            args.extend(model_specific_args)
            
        return args
    
    @classmethod
    def get_models_for_test_type(cls, test_type: str) -> List[str]:
        """Get suitable models for a specific test type."""
        return cls.MODEL_CONFIGS.get(test_type, {}).get("models", [])


class FileDownloadCache:    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/sglang_test_assets")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._download_lock = threading.Lock()
    
    def get_file(self, url: str, force_redownload: bool = False) -> str:
        """Download and cache a file, returning local path."""
        file_name = url.split("/")[-1]
        file_path = os.path.join(self.cache_dir, file_name)
        
        if force_redownload or not os.path.exists(file_path):
            with self._download_lock:
                # Double-check after acquiring lock
                if force_redownload or not os.path.exists(file_path):
                    self._download_file(url, file_path)
        
        return file_path
    
    def _download_file(self, url: str, file_path: str):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff


class PerformanceProfiler:    
    def __init__(self):
        self.measurements = {}
        self.current_operation = None
        self.start_time = None
    
    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        if self.current_operation:
            self.end_operation()  # End previous operation if forgotten
            
        self.current_operation = operation_name
        self.start_time = time.time()
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
    
    def end_operation(self) -> float:
        if not self.current_operation or not self.start_time:
            return 0.0
            
        duration = time.time() - self.start_time
        self.measurements[self.current_operation].append(duration)
        
        self.current_operation = None
        self.start_time = None
        
        return duration
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        
        for operation, durations in self.measurements.items():
            if durations:
                stats[operation] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                    "p95_time": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
                }
            else:
                stats[operation] = {
                    "count": 0, "total_time": 0, "avg_time": 0,
                    "min_time": 0, "max_time": 0, "p95_time": 0,
                }
        
        return stats


class MultimodalTestAssertions:    
    @staticmethod
    def assert_response_quality(response_text: str, min_length: int = 10, 
                              expected_keywords: Optional[List[str]] = None):
        assert isinstance(response_text, str), f"Response should be string, got {type(response_text)}"
        assert len(response_text) >= min_length, f"Response too short: {len(response_text)} < {min_length}"
        
        if expected_keywords:
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response_text.lower()]
            assert len(found_keywords) > 0, f"No expected keywords found in response. Expected: {expected_keywords}, Response: {response_text}"
    
    @staticmethod
    def assert_memory_stability(memory_stats: Dict[str, Any], max_growth_rate: float = 0.2):
        cpu_memory = memory_stats.get("cpu_memory", {})
        gpu_memory = memory_stats.get("gpu_memory", {})
        
        # Check CPU memory growth
        cpu_growth_rate = cpu_memory.get("growth_rate", 0)
        assert abs(cpu_growth_rate) <= max_growth_rate, \
            f"CPU memory growth rate too high: {cpu_growth_rate:.2%} > {max_growth_rate:.2%}"
        
        # Check GPU memory growth
        gpu_growth_rate = gpu_memory.get("growth_rate", 0)
        assert abs(gpu_growth_rate) <= max_growth_rate, \
            f"GPU memory growth rate too high: {gpu_growth_rate:.2%} > {max_growth_rate:.2%}"
        
        # Check leak detection
        leak_detected = memory_stats.get("leak_detected", {})
        assert not leak_detected.get("cpu", False), "CPU memory leak detected"
        assert not leak_detected.get("gpu", False), "GPU memory leak detected"
    
    @staticmethod
    def assert_performance_benchmarks(perf_stats: Dict[str, Dict[str, float]], 
                                    operation_limits: Dict[str, float]):
        for operation, limits in operation_limits.items():
            if operation in perf_stats:
                avg_time = perf_stats[operation].get("avg_time", 0)
                assert avg_time <= limits, \
                    f"Operation '{operation}' too slow: {avg_time:.2f}s > {limits:.2f}s"
    
    @staticmethod  
    def assert_tp_consistency(tp1_response: str, tp2_response: str, 
                            similarity_threshold: float = 0.3):        # Simple keyword-based similarity check
        tp1_words = set(tp1_response.lower().split())
        tp2_words = set(tp2_response.lower().split())
        
        if len(tp1_words) == 0 and len(tp2_words) == 0:
            return  # Both empty is fine
            
        overlap = len(tp1_words.intersection(tp2_words))
        total_unique = len(tp1_words.union(tp2_words))
        
        similarity = overlap / total_unique if total_unique > 0 else 0
        
        assert similarity >= similarity_threshold, \
            f"TP responses too different: similarity={similarity:.2%} < {similarity_threshold:.2%}\n" \
            f"TP1: {tp1_response}\nTP2: {tp2_response}"


# Convenience instances for immediate use
default_file_cache = FileDownloadCache()
default_test_config = MultimodalTestConfig()
default_assertions = MultimodalTestAssertions()