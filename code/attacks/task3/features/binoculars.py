from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BinocularsConfig:
    observer_model: str = "gpt2"
    performer_model: str = "gpt2-medium"
    max_length: int = 512
    device: str = "cpu"


class BinocularsExtractor:
    def __init__(self, cfg: BinocularsConfig):
        self.cfg = cfg
        self.obs_tok = AutoTokenizer.from_pretrained(cfg.observer_model)
        self.obs_mod = AutoModelForCausalLM.from_pretrained(cfg.observer_model).to(cfg.device).eval()
        self.per_tok = AutoTokenizer.from_pretrained(cfg.performer_model)
        self.per_mod = AutoModelForCausalLM.from_pretrained(cfg.performer_model).to(cfg.device).eval()

    @torch.no_grad()
    def _avg_neg_logprob(self, text: str, tok, model) -> float:
        encoded = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.cfg.device)
        if input_ids.shape[1] < 2:
            return 0.0
        logits = model(input_ids).logits[0, :-1]
        targets = input_ids[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return float(-tok_lp.mean().item())

    def featurize(self, text: str) -> dict[str, float]:
        obs_nll = self._avg_neg_logprob(text, self.obs_tok, self.obs_mod)
        per_nll = self._avg_neg_logprob(text, self.per_tok, self.per_mod)
        obs_ppl = math.exp(obs_nll)
        per_ppl = math.exp(per_nll)
        score = math.log(obs_ppl + 1e-9) / (math.log(per_ppl + 1e-9) + 1e-9)
        return {
            "binoculars_score": float(score),
            "binoculars_obs_nll": float(obs_nll),
            "binoculars_per_nll": float(per_nll),
            "binoculars_delta_nll": float(obs_nll - per_nll),
        }
