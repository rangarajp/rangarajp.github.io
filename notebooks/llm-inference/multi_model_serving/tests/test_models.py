import unittest
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Run from multi_model_serving/ so relative imports work
os.chdir(Path(__file__).resolve().parents[1])

from app.server import app
from app.store import REPO_ROOT


class TestMultiModelLLMServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.model_ids = {
            "qwen": "qwen2.5-0.5b",
            "tinyllama": "tinyllama-1.1b",
        }
        cls.qwen_path = REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
        cls.tiny_path = REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"

    def test_list_models(self):
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("available_models", data)
        self.assertIn("loaded_models", data)
        available = data["available_models"]
        self.assertIn(self.model_ids["qwen"], available)
        self.assertIn(self.model_ids["tinyllama"], available)

    def test_invalid_model_id(self):
        response = self.client.post(
            "/generate",
            json={"model_id": "invalid-id", "prompt": "hello", "max_new_tokens": 8},
        )
        self.assertEqual(response.status_code, 404)

    @unittest.skipUnless(
        (REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct" / "config.json").exists(),
        "Qwen2.5 checkpoint not found under models/",
    )
    def test_generate_qwen(self):
        response = self.client.post(
            "/generate",
            json={
                "model_id": self.model_ids["qwen"],
                "prompt": "Say hello in one short sentence.",
                "max_new_tokens": 24,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("generated_text", data)
        self.assertEqual(data["model_id"], self.model_ids["qwen"])
        self.assertIsInstance(data["generated_text"], str)

    @unittest.skipUnless(
        (REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0" / "config.json").exists(),
        "TinyLlama checkpoint not found under models/",
    )
    def test_generate_tinyllama_via_predict(self):
        response = self.client.post(
            "/predict",
            json={
                "model_id": self.model_ids["tinyllama"],
                "input_data": {
                    "prompt": "What is 2+2? Answer briefly.",
                    "max_new_tokens": 24,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("generated_text", data)
        self.assertEqual(data["model_id"], self.model_ids["tinyllama"])

    @unittest.skipUnless(
        (REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct" / "config.json").exists()
        and (REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0" / "config.json").exists(),
        "Both local LLM checkpoints required for cache test",
    )
    def test_model_cache_holds_both(self):
        # Load both models; max_models=2 so both should stay loaded
        for mid in (self.model_ids["qwen"], self.model_ids["tinyllama"]):
            response = self.client.post(
                "/generate",
                json={"model_id": mid, "prompt": "Hi", "max_new_tokens": 8},
            )
            self.assertEqual(response.status_code, 200)

        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        loaded = response.json()["loaded_models"]
        self.assertLessEqual(len(loaded), 2)
        self.assertIn(self.model_ids["qwen"], loaded)
        self.assertIn(self.model_ids["tinyllama"], loaded)


if __name__ == "__main__":
    unittest.main()
