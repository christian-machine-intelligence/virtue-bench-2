"""
Local HuggingFace model runner for VirtueBench V2.

Loads a model (with optional LoRA adapter) and runs inference locally.
Supports batched generation for significantly higher throughput on GPU.

Usage via CLI:
    virtue-bench run --model meta-llama/Llama-3.1-8B-Instruct --runner hf-local
    virtue-bench run --model Qwen/Qwen2.5-72B-Instruct --runner hf-local --hf-quantize 4bit
    virtue-bench run --model Qwen/Qwen2.5-72B-Instruct --runner hf-local --hf-quantize 4bit --batch-size 8
"""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModelRunner


class HFLocalRunner(ModelRunner):
    """Run inference on a local HuggingFace model with batched generation."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        quantize: str | None = None,
        batch_size: int = 1,
        thinking_mode: bool | None = None,
    ):
        self._model_name = model_name
        self._adapter_path = adapter_path
        self._quantize = quantize
        self._batch_size = batch_size
        self._thinking_mode = thinking_mode
        self._model = None
        self._tokenizer = None

    def _chat_template_kwargs(self) -> dict:
        # Qwen3.5 chat template accepts enable_thinking; other models ignore it.
        if self._thinking_mode is None:
            return {}
        return {"enable_thinking": self._thinking_mode}

    def _effective_max_tokens(self, max_tokens: int) -> int:
        # Thinking mode emits a <think>...</think> block before the answer;
        # 128 tokens is not enough to fit the reasoning + "A" or "B".
        if self._thinking_mode is True:
            return max(max_tokens, 2048)
        return max_tokens

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Qwen thinking blocks are wrapped in <think>...</think>. Strip for
        # downstream answer extraction; keep the rest of the response intact.
        if "</think>" in text:
            return text.split("</think>", 1)[1].lstrip()
        return text

    def _ensure_loaded(self):
        if self._model is not None:
            return

        load_kwargs = dict(trust_remote_code=True)

        if self._quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        elif self._quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["device_map"] = "auto"
        elif self._quantize == "fp8":
            # For repos that ship pre-quantized FP8 weights (e.g.,
            # Qwen/Qwen3.5-122B-A10B-FP8). Transformers auto-detects the
            # quantization_config from the model repo — we just need to
            # (a) NOT set a bnb config and (b) allow multi-GPU spread.
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = {"": "cuda:0"}

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, **load_kwargs,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        if self._adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)

        self._model.eval()

    def _build_messages(self, prompt: str, system_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    async def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        **kwargs,
    ) -> dict:
        self._ensure_loaded()

        messages = self._build_messages(prompt, system_prompt)

        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
                **self._chat_template_kwargs(),
            ).to(self._model.device)

            gen_kwargs = dict(max_new_tokens=self._effective_max_tokens(max_tokens))
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

            return {"response": self._strip_thinking(response), "infra_error": None}

        except Exception as e:
            return {"response": "", "infra_error": str(e)}

    async def query_batch(
        self,
        prompts: List[str],
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> List[dict]:
        """Batched generation: tokenize all prompts with left-padding, one generate() call."""
        self._ensure_loaded()

        all_messages = [self._build_messages(p, system_prompt) for p in prompts]

        try:
            # Tokenize each conversation individually, then pad together
            tmpl_kwargs = self._chat_template_kwargs()
            tokenized = [
                self._tokenizer.apply_chat_template(
                    msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True,
                    **tmpl_kwargs,
                )
                for msgs in all_messages
            ]

            # Left-pad to the longest sequence in the batch
            max_len = max(t["input_ids"].shape[1] for t in tokenized)
            pad_id = self._tokenizer.pad_token_id

            padded_ids = []
            attention_masks = []
            input_lengths = []

            for t in tokenized:
                seq_len = t["input_ids"].shape[1]
                pad_len = max_len - seq_len
                input_lengths.append(seq_len)

                if pad_len > 0:
                    pad_ids = torch.full((1, pad_len), pad_id, dtype=t["input_ids"].dtype)
                    pad_mask = torch.zeros(1, pad_len, dtype=t["attention_mask"].dtype)
                    padded_ids.append(torch.cat([pad_ids, t["input_ids"]], dim=1))
                    attention_masks.append(torch.cat([pad_mask, t["attention_mask"]], dim=1))
                else:
                    padded_ids.append(t["input_ids"])
                    attention_masks.append(t["attention_mask"])

            batch_ids = torch.cat(padded_ids, dim=0).to(self._model.device)
            batch_mask = torch.cat(attention_masks, dim=0).to(self._model.device)

            gen_kwargs = dict(max_new_tokens=self._effective_max_tokens(max_tokens))
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    **gen_kwargs,
                )

            # Decode each sequence, stripping the padded input
            results = []
            for i in range(len(prompts)):
                # Output includes the full padded input + generation
                gen_tokens = outputs[i][max_len:]
                response = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)
                results.append({"response": self._strip_thinking(response), "infra_error": None})

            return results

        except Exception as e:
            # On batch failure, fall back to sequential
            return [{"response": "", "infra_error": str(e)} for _ in prompts]

    @property
    def supports_batching(self) -> bool:
        return self._batch_size > 1

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def model_id(self) -> str:
        name = self._model_name.split("/")[-1]
        if self._adapter_path:
            adapter_name = self._adapter_path.rstrip("/").split("/")[-2]
            return f"{name}+{adapter_name}"
        return name
