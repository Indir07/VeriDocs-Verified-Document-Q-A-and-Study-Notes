from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class HFGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        output = self.model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
