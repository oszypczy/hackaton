"""Stylometric features — pure Python, fast (CPU only).

Features: sentence/word length stats, punctuation, function words, syntactic markers,
repetition rates, structural patterns. Watermarki MOGĄ subtle wpływać na te metryki
(zielone tokeny mają tendency to być common words → mniej rzadkich słów).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

# Common English function words (top 100)
FUNCTION_WORDS = set("""
the of and to a in is it you that he was for on are with as i his they be at one have this from
or had by hot but some what there we can out other were all your when up use word how said an
each she which do their time if will way about many then them write would like so these her long
make thing see him two has look more day could go come did number sound no most people my over
know water than call first who may down side been now find any new work part take get place made
live where after back little only round man year came show every good me give our under name
""".split())

LIST_PATTERNS = [
    re.compile(r"^\d+\."),  # 1.
    re.compile(r"^\*"),  # bullet
    re.compile(r"^-"),
    re.compile(r"^•"),
]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _split_words(text: str) -> list[str]:
    # Simple word tokenization
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def extract(text: str) -> dict[str, float]:
    feats: dict[str, float] = {}

    # ── Sentence statistics
    sents = _split_sentences(text)
    sent_lens = [len(s.split()) for s in sents]
    feats["sty_n_sents"] = float(len(sents))
    feats["sty_sent_len_mean"] = float(np.mean(sent_lens)) if sent_lens else 0.0
    feats["sty_sent_len_std"] = float(np.std(sent_lens)) if len(sent_lens) >= 2 else 0.0
    feats["sty_sent_len_max"] = float(max(sent_lens)) if sent_lens else 0.0
    feats["sty_sent_len_min"] = float(min(sent_lens)) if sent_lens else 0.0

    # ── Word statistics
    words = _split_words(text)
    feats["sty_n_words"] = float(len(words))
    word_lens = [len(w) for w in words]
    feats["sty_word_len_mean"] = float(np.mean(word_lens)) if word_lens else 0.0
    feats["sty_word_len_std"] = float(np.std(word_lens)) if len(word_lens) >= 2 else 0.0

    # Hapax legomena ratio (words appearing once)
    word_counts = Counter(words)
    feats["sty_hapax_ratio"] = (
        sum(1 for c in word_counts.values() if c == 1) / max(len(words), 1)
    )

    # Type-token ratio
    feats["sty_ttr"] = len(set(words)) / max(len(words), 1)

    # Function word ratio
    n_func = sum(1 for w in words if w in FUNCTION_WORDS)
    feats["sty_func_ratio"] = n_func / max(len(words), 1)

    # ── Punctuation
    n_chars = len(text)
    feats["sty_punct_ratio"] = sum(1 for c in text if c in ".,;:!?") / max(n_chars, 1)
    feats["sty_comma_ratio"] = text.count(",") / max(n_chars, 1)
    feats["sty_period_ratio"] = text.count(".") / max(n_chars, 1)

    # Uppercase ratio
    feats["sty_upper_ratio"] = sum(1 for c in text if c.isupper()) / max(n_chars, 1)
    feats["sty_digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(n_chars, 1)

    # ── Structural markers
    list_lines = 0
    for line in text.split("\n"):
        line = line.strip()
        for pat in LIST_PATTERNS:
            if pat.match(line):
                list_lines += 1
                break
    feats["sty_list_lines"] = float(list_lines)
    feats["sty_newlines"] = float(text.count("\n"))
    feats["sty_doublenewlines"] = float(text.count("\n\n"))

    # ── Repetition
    if len(words) >= 4:
        # Bigram repetition
        bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        feats["sty_bigram_max_rep"] = float(max(bigram_counts.values()))
        feats["sty_bigram_unique_ratio"] = len(bigram_counts) / max(len(bigrams), 1)
    else:
        feats["sty_bigram_max_rep"] = 0.0
        feats["sty_bigram_unique_ratio"] = 1.0

    # Char-level entropy (Shannon)
    if n_chars > 0:
        char_counts = Counter(text)
        char_probs = np.array(list(char_counts.values())) / n_chars
        feats["sty_char_entropy"] = float(-(char_probs * np.log2(char_probs + 1e-12)).sum())
    else:
        feats["sty_char_entropy"] = 0.0

    return feats
