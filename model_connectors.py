# model_connectors.py
"""
Unified connector layer for calling any teacher/student model backend
from a single interface: connector.chat(system_prompt, user_prompt) -> str
"""

import requests
from abc import ABC, abstractmethod


class BaseConnector(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        ...


# ── 1. Groq (default teacher) ────────────────────────────────────────────
class GroqConnector(BaseConnector):
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def chat(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content


# ── 2. OpenAI-compatible local servers: Ollama / LM Studio ──────────────
class OpenAICompatibleConnector(BaseConnector):
    """
    Works for Ollama (default http://localhost:11434/v1),
    LM Studio (default http://localhost:1234/v1),
    or any OpenAI-compatible custom endpoint.
    """
    def __init__(self, base_url: str, model_name: str, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key

    def chat(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── 3. Hugging Face local model (transformers pipeline) ─────────────────
class HFLocalConnector(BaseConnector):
    def __init__(self, model_name: str, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device
        )

    def chat(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
        import torch
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01)
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text


# ── 4. Custom API (user-provided arbitrary endpoint) ─────────────────────
class CustomAPIConnector(BaseConnector):
    """
    For any user-supplied endpoint that doesn't follow OpenAI schema exactly.
    User provides the URL and simple keys for payload/response.
    """
    def __init__(self, url: str, api_key: str, request_key: str = "prompt", response_key: str = "text", headers: dict = None):
        self.url = url
        self.api_key = api_key
        self.request_key = request_key
        self.response_key = response_key
        self.headers = headers or {}

        def chat(self, system_prompt, user_prompt, max_tokens=800, temperature=0.0):
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if not r.ok:
                raise Exception(f"{r.status_code} error from {url}: {r.text[:500]}")
            return r.json()["choices"][0]["message"]["content"]


# ── Factory function used by app.py ──────────────────────────────────────
def get_connector(source: str, **kwargs) -> BaseConnector:
    """
    source: "groq" | "ollama" | "lmstudio" | "huggingface" | "custom"
    kwargs: connector-specific params (api_key, model_name, base_url, url, etc.)
    """
    source = source.lower()
    if source == "groq":
        return GroqConnector(api_key=kwargs["api_key"], model_name=kwargs.get("model_name", "llama-3.3-70b-versatile"))
    elif source == "ollama":
        return OpenAICompatibleConnector(base_url=kwargs.get("base_url", "http://localhost:11434/v1"), model_name=kwargs["model_name"])
    elif source == "lmstudio":
        return OpenAICompatibleConnector(base_url=kwargs.get("base_url", "http://localhost:1234/v1"), model_name=kwargs["model_name"])
    elif source == "huggingface":
        return HFLocalConnector(model_name=kwargs["model_name"], device=kwargs.get("device", "auto"))
    elif source == "custom":
        return CustomAPIConnector(
            url=kwargs["url"],
            api_key=kwargs.get("api_key", ""),
            request_key=kwargs.get("request_key", "prompt"),
            response_key=kwargs.get("response_key", "text"),
            headers=kwargs.get("headers", {})
        )
    else:
        raise ValueError(f"Unknown model source: {source}")