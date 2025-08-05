import unittest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from test_vision_openai_server_common import *

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestQwen2AudioServer(TestOpenAIVisionServer):    
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-Audio-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "4",
            ],
        )
        cls.base_url += "/v1"

    def test_speech_transcription(self):
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "Listen to this audio carefully and provide an accurate transcription.",
            category="speech_transcription"
        )
        
        # Check for key phrases in Trump's WEF speech
        expected_phrases = [
            "thank you",
            "privilege", 
            "leader",
            "science",
            "art"
        ]
        
        for phrase in expected_phrases:
            assert phrase in audio_response, f"Missing expected phrase '{phrase}' in transcription: {audio_response}"

    def test_audio_classification(self):        # Test with bird song
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "What type of audio is this? Describe what you hear.",
            category="audio_classification"
        )
        
        # Should identify it as bird-related audio
        bird_keywords = ["bird", "song", "chirp", "tweet", "nature", "sound"]
        assert any(keyword in audio_response for keyword in bird_keywords), \
            f"Failed to identify bird audio: {audio_response}"

    def test_audio_question_answering(self):
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "Who is speaking in this audio and what is the main topic?",
            category="audio_qa"
        )
        
        # Should identify speaker characteristics and topic
        content_indicators = ["speak", "leader", "davos", "economic", "forum", "business"]
        assert any(indicator in audio_response for indicator in content_indicators), \
            f"Failed to answer question about audio content: {audio_response}"

    def test_audio_sentiment_analysis(self):
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "What is the tone and sentiment of this speech?",
            category="audio_sentiment"
        )
        
        # Should identify formal/professional tone
        tone_indicators = ["formal", "professional", "confident", "positive", "diplomatic"]
        assert any(indicator in audio_response for indicator in tone_indicators), \
            f"Failed to analyze audio sentiment: {audio_response}"


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
                "--cuda-graph-max-bs", "4",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_transcription_accuracy(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()

    def test_multimodal_audio_vision_combination(self):
        import openai
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Download audio file first
        audio_file_path = self.get_or_download_file(AUDIO_TRUMP_SPEECH_URL)
        
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
                            "type": "audio_url", 
                            "audio_url": {"url": audio_file_path},
                        },
                        {
                            "type": "text",
                            "text": "I have provided both an image and an audio clip. Please describe what you see in the image and what you hear in the audio separately.",
                        },
                    ],
                },
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        multimodal_response = response.choices[0].message.content.lower()
        
        print(f"Multimodal audio+vision response: {multimodal_response}")
        
        # Should mention both visual and audio elements
        visual_indicators = ["logo", "blue", "s", "design", "image", "see"]
        audio_indicators = ["speech", "voice", "speak", "audio", "hear", "sound"]
        
        has_visual = any(indicator in multimodal_response for indicator in visual_indicators)
        has_audio = any(indicator in multimodal_response for indicator in audio_indicators)
        
        assert has_visual, f"Failed to process visual content in multimodal request: {multimodal_response}"
        assert has_audio, f"Failed to process audio content in multimodal request: {multimodal_response}"


class TestGemma3nAudioServer(TestOpenAIVisionServer):    
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
                "--trust-remote-code",
                "--mem-fraction-static", "0.5",
                "--cuda-graph-max-bs", "4",
                "--enable-multimodal",  # Explicit enable for Gemma3n
            ],
        )
        cls.base_url += "/v1"

    def test_audio_understanding(self):
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "Describe what you hear in this audio clip.",
            category="gemma3n_audio"
        )
        
        # Should recognize it as natural/bird sounds
        nature_keywords = ["bird", "nature", "sound", "sing", "chirp"]
        assert any(keyword in audio_response for keyword in nature_keywords), \
            f"Gemma3n failed to understand audio: {audio_response}"


class TestAudioPerformanceAndRobustness(CustomTestCase):    
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"  # Use MiniCPM-O as it has good audio support
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "6",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def get_or_download_file(self, url: str) -> str:
        import os
        import requests
        
        cache_dir = os.path.expanduser("~/.cache")
        file_name = url.split("/")[-1]
        file_path = os.path.join(cache_dir, file_name)
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(file_path):
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
        return file_path

    def test_concurrent_audio_processing(self):
        import openai
        
        def process_audio_request(request_id):
            try:
                client = openai.Client(api_key=self.api_key, base_url=self.base_url)
                
                # Alternate between different audio files
                audio_url = AUDIO_TRUMP_SPEECH_URL if request_id % 2 == 0 else AUDIO_BIRD_SONG_URL
                audio_file_path = self.get_or_download_file(audio_url)
                
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": audio_file_path},
                                },
                                {
                                    "type": "text",
                                    "text": f"Briefly describe this audio. Request {request_id}.",
                                },
                            ],
                        }
                    ],
                    temperature=0.5,
                    max_tokens=100,
                )
                
                duration = time.time() - start_time
                audio_response = response.choices[0].message.content
                
                return {
                    "request_id": request_id,
                    "success": True,
                    "duration": duration,
                    "response_length": len(audio_response),
                    "response": audio_response,
                }
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time if 'start_time' in locals() else 0,
                }

        # Test with concurrent requests
        num_concurrent_requests = 8
        results = []
        
        print(f"Testing concurrent audio processing with {num_concurrent_requests} requests")
        
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(process_audio_request, i) for i in range(num_concurrent_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                status = "SUCCESS" if result["success"] else "FAILED"
                duration = result.get("duration", 0)
                print(f"Request {result['request_id']}: {status} in {duration:.2f}s")

        # Analyze results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        success_rate = len(successful_results) / len(results)
        avg_duration = sum(r["duration"] for r in successful_results) / len(successful_results) if successful_results else 0
        
        print(f"Concurrent audio test results:")
        print(f"  Success rate: {success_rate:.1%} ({len(successful_results)}/{len(results)})")
        print(f"  Average duration: {avg_duration:.2f}s")
        
        # Assertions
        assert success_rate >= 0.75, f"Audio processing success rate too low: {success_rate:.1%}"
        assert avg_duration < 30.0, f"Audio processing too slow: {avg_duration:.2f}s average"
        
        # Validate response quality for successful requests
        for result in successful_results[:3]:  # Check first few responses
            response = result["response"].lower()
            assert len(response) > 10, f"Response too short: {response}"
            # Should contain some relevant audio-related terms
            audio_terms = ["audio", "sound", "voice", "speech", "music", "noise", "hear"]
            assert any(term in response for term in audio_terms), f"Response doesn't seem audio-related: {response}"

    def test_audio_processing_latency(self):
        import openai
        
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Test different audio types for performance comparison
        test_cases = [
            {
                "name": "speech_audio",
                "url": AUDIO_TRUMP_SPEECH_URL,
                "prompt": "Transcribe this speech.",
            },
            {
                "name": "ambient_audio", 
                "url": AUDIO_BIRD_SONG_URL,
                "prompt": "What sounds do you hear?",
            },
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"Testing audio latency for: {test_case['name']}")
            
            audio_file_path = self.get_or_download_file(test_case["url"])
            latencies = []
            
            # Run multiple iterations to get average latency
            for i in range(5):
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": audio_file_path},
                                },
                                {
                                    "type": "text",
                                    "text": test_case["prompt"],
                                },
                            ],
                        }
                    ],
                    temperature=0.0,  # Deterministic for consistency
                    max_tokens=150,
                )
                
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Validate response
                response_text = response.choices[0].message.content
                assert len(response_text) > 5, f"Response too short for {test_case['name']}: {response_text}"
                
                # Brief pause between requests
                time.sleep(1)
            
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            results[test_case["name"]] = {
                "avg_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "latencies": latencies,
            }
            
            print(f"  {test_case['name']} - Avg: {avg_latency:.2f}s, Min: {min_latency:.2f}s, Max: {max_latency:.2f}s")
        
        # Performance assertions
        for test_name, metrics in results.items():
            assert metrics["avg_latency"] < 25.0, f"{test_name} average latency too high: {metrics['avg_latency']:.2f}s"
            assert metrics["max_latency"] < 40.0, f"{test_name} max latency too high: {metrics['max_latency']:.2f}s"
            
            # Check for reasonable consistency (max shouldn't be more than 3x avg)
            consistency_ratio = metrics["max_latency"] / metrics["avg_latency"]
            assert consistency_ratio < 3.0, f"{test_name} latency too inconsistent: {consistency_ratio:.2f}x"

    def test_mixed_modality_stress(self):
        import openai
        
        def mixed_modality_request(request_id):
            try:
                client = openai.Client(api_key=self.api_key, base_url=self.base_url)
                
                # Prepare content based on request type
                content = []
                
                if request_id % 3 == 0:
                    # Audio + Text
                    audio_file_path = self.get_or_download_file(AUDIO_BIRD_SONG_URL)
                    content = [
                        {"type": "audio_url", "audio_url": {"url": audio_file_path}},
                        {"type": "text", "text": f"Describe this audio. Request {request_id}."},
                    ]
                elif request_id % 3 == 1:
                    # Image + Text
                    content = [
                        {"type": "image_url", "image_url": {"url": IMAGE_SGL_LOGO_URL}},
                        {"type": "text", "text": f"Describe this image. Request {request_id}."},
                    ]
                else:
                    # Audio + Image + Text (if supported)
                    audio_file_path = self.get_or_download_file(AUDIO_TRUMP_SPEECH_URL)
                    content = [
                        {"type": "image_url", "image_url": {"url": IMAGE_MAN_IRONING_URL}},
                        {"type": "audio_url", "audio_url": {"url": audio_file_path}},
                        {"type": "text", "text": f"Describe both the image and audio. Request {request_id}."},
                    ]
                
                response = client.chat.completions.create(
                    model="default",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.5,
                    max_tokens=150,
                )
                
                return {
                    "request_id": request_id,
                    "modality_type": request_id % 3,
                    "success": True,
                    "response": response.choices[0].message.content,
                }
                
            except Exception as e:
                return {
                    "request_id": request_id,
                    "modality_type": request_id % 3,
                    "success": False,
                    "error": str(e),
                }
        
        # Run mixed modality stress test
        num_requests = 15  # 5 of each type
        
        print(f"Running mixed modality stress test with {num_requests} requests")
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(mixed_modality_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze by modality type
        modality_results = {0: [], 1: [], 2: []}  # audio, image, audio+image
        
        for result in results:
            modality_type = result["modality_type"]
            modality_results[modality_type].append(result)
        
        # Validate each modality type
        modality_names = {0: "audio", 1: "image", 2: "audio+image"}
        
        for modality_type, modality_results_list in modality_results.items():
            successful = len([r for r in modality_results_list if r["success"]])
            total = len(modality_results_list)
            success_rate = successful / total if total > 0 else 0
            
            print(f"  {modality_names[modality_type]}: {successful}/{total} successful ({success_rate:.1%})")
            
            # Each modality type should have reasonable success rate
            assert success_rate >= 0.6, f"{modality_names[modality_type]} success rate too low: {success_rate:.1%}"


if __name__ == "__main__":
    # Remove the base class to prevent it from running
    del TestOpenAIVisionServer
    unittest.main()