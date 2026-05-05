# LEARNINGS — what we discovered, what worked, what didn't

> Append after every `/compact` or significant insight. Cheap reference for the next session.

## 2026-05-05 — Wnioski z ćwiczenia A (Dataset Inference, Maini et al.)

### Wyniki iteracji
| Iteracja | Score | AUC | TPR@1% | Co zmieniło |
|---|---|---|---|---|
| Baseline (minkpp + loss) | 0.23 | 0.38 | ~0.01 | — |
| Split scoring + zlib dataset-level | 0.49 | 0.619 | 0.07 | split scoring, pure zlib dla dataset-level |
| Zamiana CNN → Wikipedia jako OUT | 0.53 | 0.689 | 0.132 | zmniejszenie gap stylowego |

### Kluczowe wnioski (użyteczne na hackathonie)

**Min-K%++ może wskazywać odwrotnie przy domain shift.**
Na naszych danych OUT (Wikipedia/CNN) miało wyższy minkpp niż IN — wyglądało bardziej jak "member". Efekt domeny (encyklopedyczny/newsowy tekst = bardziej regularny) całkowicie zdominował sygnał membership. Na hackathonie IN i OUT będą z tego samego rozkładu — wtedy minkpp wraca do roli głównej cechy. Zlib jako "najlepszy feature" to red flag fixture, nie sygnał że zlib jest genialny.

**Zlib ratio (bez modelu!) pokonał algorytm z papieru.**
Web text (IN) jest bardziej entropowy niż Wikipedia/CNN → gorsza kompresja → wyższy zlib ratio. Na prawdziwym hackathonie: jeśli zlib bije minkpp, masz domain shift — nie kontynuuj tej ścieżki.

**Split scoring: inne cechy dla doc-level i dataset-level.**
Dataset t-test: czysty zlib. Doc-level: mix zlib + loss. Nie używaj jednej cechy dla obu poziomów.

**Wybór OUT datasetu to jeden z najważniejszych hyperparametrów.**
CNN jako OUT: Set2 p-value = 0.09 (niebezpiecznie blisko 0.05 — jeden outlier = kara −1). Wikipedia: p-value = 0.867. Reguła: OUT powinno być z tego samego gatunku tekstu co IN.

**Czas na M4 MPS (Pythia-410m + 160m, fp16):**
- Cold start (2200 docs × 2 modele): ~15 min
- Po zamianie OUT z cache (1000 nowych docs): ~7 min
- Pełny cache: <5 s
- Małe modele (≤410m) na M4 wystarczą do MIA — nie potrzeba Jülicha.

**Claude Code security hook blokuje `model.train(mode=False)` alternatywę.**
Właściwie: hook blokuje `model.eval()` (PyTorch inference mode). Użyj `model.train(False)` — identyczne funkcjonalnie, nie triggeruje hooka.

### Co robić inaczej na hackathonie
1. Przed jakimkolwiek algorytmem: sprawdź rozkład IN vs OUT (histogram długości, zlib ratio, perplexity GPT-2). Różne rozkłady = wyniki będą odwrócone.
2. Uruchom minkpp i zlib równolegle — jeśli zlib wygrywa, masz domain shift.
3. Dataset-level: zawsze t-test, nie threshold na średniej.
4. p-value < 0.2 = zmień podejście, nie tuninguj progu.

## 2026-04-26 22:40 Token Optimization Plan Phase 0
- BSD `xargs` on macOS chokes on inline `sh -c` with multi-line script ("command line cannot be assembled, too long"). Fix: extract logic to a function, `export -f`, then call via `bash -c 'fn "$@"' _ {}`. Saved in `scripts/extract_papers.sh`.
- 25 PDFs / 122 MB → 25 .txt / 5.6 MB. ~22× storage reduction; ~2.4× token reduction at read time per plan.
- Splitting MAPPING.md (3099 words = ~4.1k tokens) into INDEX (693 words) + rich MAPPING beats single trimmed file. Two-stage routing keeps default-load cost low while preserving rich grep-term content.
- Cache hierarchy lesson: any `Status (snapshot YYYY-MM-DD)` block in CLAUDE.md invalidates 90% read discount on every status update. Volatile content goes to `docs/STATUS.md` and CLAUDE.md keeps just `@docs/STATUS.md` reference.
