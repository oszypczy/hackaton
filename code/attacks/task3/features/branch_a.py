from __future__ import annotations

import gzip
import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BranchAConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    device: str = "cpu"


class BranchAExtractor:
    def __init__(self, cfg: BranchAConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        use_fp16 = cfg.device.startswith("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )
        self.model.to(cfg.device)
        self.model.eval()

    @torch.no_grad()
    def _token_logprobs_and_ranks(self, text: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.cfg.device)
        if input_ids.shape[1] < 2:
            return np.zeros(1), np.zeros(1), input_ids[0].tolist()
        logits = self.model(input_ids).logits[0, :-1]
        targets = input_ids[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
        tgt_logits = logits.gather(1, targets.unsqueeze(1))
        ranks = (logits > tgt_logits).sum(dim=-1).cpu().numpy()
        return tok_lp, ranks, input_ids[0].tolist()

    @staticmethod
    def _safe_percentile(arr: np.ndarray, q: float) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, q))

    @staticmethod
    def _log_diversity(token_ids: list[int], n: int) -> float:
        if len(token_ids) < n:
            return 0.0
        grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
        if not grams:
            return 0.0
        return float(-math.log(1 - len(set(grams)) / len(grams) + 1e-9))

    @staticmethod
    def _burstiness(text: str) -> float:
        sents = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        lens = [len(s.split()) for s in sents]
        if len(lens) < 2:
            return 0.0
        return float(np.std(lens) / (np.mean(lens) + 1e-9))

    @staticmethod
    def _gzip_ratio(text: str) -> float:
        raw = text.encode("utf-8")
        if not raw:
            return 1.0
        return float(len(gzip.compress(raw)) / len(raw))

    def featurize(self, text: str) -> dict[str, float]:
        tok_lp, ranks, token_ids = self._token_logprobs_and_ranks(text)
        n = max(1, len(ranks))
        words = text.split()
        ttr = len(set(words)) / max(1, len(words))
        return {
            "a_lp_mean": float(tok_lp.mean()) if tok_lp.size else 0.0,
            "a_lp_std": float(tok_lp.std()) if tok_lp.size else 0.0,
            "a_lp_p10": self._safe_percentile(tok_lp, 10),
            "a_lp_p90": self._safe_percentile(tok_lp, 90),
            "a_gltr_top10": float((ranks < 10).sum() / n),
            "a_gltr_top100": float((ranks < 100).sum() / n),
            "a_gltr_top1000": float((ranks < 1000).sum() / n),
            "a_gltr_rest": float((ranks >= 1000).sum() / n),
            "a_logdiv2": self._log_diversity(token_ids, 2),
            "a_logdiv3": self._log_diversity(token_ids, 3),
            "a_burst": self._burstiness(text),
            "a_gzip_ratio": self._gzip_ratio(text),
            "a_ttr": float(ttr),
            "a_n_tokens": float(len(token_ids)),
        }
