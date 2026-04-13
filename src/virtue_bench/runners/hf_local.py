"""
Local HuggingFace model runner for VirtueBench V2.

Loads a model (with optional LoRA adapter) and runs inference locally.

Usage via CLI:
    virtue-bench run --model meta-llama/Llama-3.1-8B-Instruct --runner hf-local
    virtue-bench run --model meta-llama/Llama-3.1-8B-Instruct --runner hf-local --hf-adapter /path/to/adapter
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelRunner


class HFLocalRunner(ModelRunner):
    """Run inference on a local HuggingFace model."""

    def __init__(self, model_name: str, adapter_path: str | None = None):
        self._model_name = model_name
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        if self._adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.eval()

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        **kwargs,
    ) -> dict:
        self._ensure_loaded()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
            ).to(self._model.device)

            gen_kwargs = dict(max_new_tokens=max_tokens)
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output = self._model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            response = self._tokenizer.decode(
                output[0][input_len:], skip_special_tokens=True
            )

            return {"response": response, "infra_error": None}

        except Exception as e:
            return {"response": "", "infra_error": str(e)}

    def model_id(self) -> str:
        name = self._model_name.split("/")[-1]
        if self._adapter_path:
            adapter_name = self._adapter_path.rstrip("/").split("/")[-2]
            return f"{name}+{adapter_name}"
        return name
