# Contrastive Decoding plan — parallel session briefing

> **You are starting a NEW Claude Code session in the SAME repo as another running session.**
> The other session is implementing K-shot + Levenshtein medoid on branch `task2-prompt`.
> You are implementing **Contrastive Decoding** on a NEW branch `task2-prompt-cd` in a worktree.
> Both sessions evaluate in parallel, winner is submitted as v3.

## Context (read first — 1 min)

**Task:** Black-box PII reconstruction (EMAIL/CREDIT/PHONE) from intentionally OVERFITTED multimodal LMM (OLMo-2-1B + LLaVA-HR vision encoder, bf16, ~3.6 GB) on JURECA cluster. Score = mean(1 − normalized Levenshtein) over 1000 task/ rows × 3 PII = 3000 predictions.

**Current state (2026-05-09 ~23:00):**
- Best leaderboard score: 0.381 (v2 = strategy `direct_probe`)
- Calibrator: val_pii blank-mode 840 samples, predicted within 0.03 of actual LB
- 8 strategies tested. `direct_probe` wins with 0.398 OVERALL on calibrator.
- V1 oneshot_demo (single ICL demo from val_pii) just **REJECTED** at 0.349 (-0.049 OVERALL) due to demo-leak pollution (model copies demo's surface features).
- Phase 0 post-process replay (CREDIT format, EMAIL/PHONE blacklists) didn't move LB — format wins below noise. Content-level wins required.

**Your method (research §2.1, arXiv:2309.09117 + arXiv:2210.15097):**
> O'Brien & Lewis "Contrastive Decoding Improves Reasoning" + Li et al. original CD.
> `α·logits_target − β·logits_amateur` per decoding step concentrates probability on
> tokens the expert (overfit target) assigns disproportionately = signature of memorized PII.
> Cluster has both `target_lmm/` (PII-overfit) AND `shadow_lmm/` (same arch, no PII finetune).
> Shadow = ideal amateur. Tokenizers are identical (both OLMo-2).

## Setup — DO THIS FIRST (5 min)

### 1. Create branch + worktree LOCALLY

```bash
cd /Users/arturkempinski/hackaton
git fetch origin
git branch task2-prompt-cd task2-prompt   # branch from current task2-prompt HEAD (commit dbe4de8)
git worktree add /Users/arturkempinski/hackaton-cd task2-prompt-cd
cd /Users/arturkempinski/hackaton-cd
git status                                 # confirm: branch task2-prompt-cd, clean tree
```

**RULE:** all your work happens in `/Users/arturkempinski/hackaton-cd/`. Do NOT touch `/Users/arturkempinski/hackaton/` (the other session lives there).

### 2. Check Jülich socket alive

```bash
# user must run this once per 4h session, NOT you:
# ! scripts/juelich_connect.sh
# verify it's up:
scripts/juelich_exec.sh "whoami" 2>&1 | tail -3
# should print: kempinski1
```

If socket dead → ask user to run `scripts/juelich_connect.sh`.

### 3. Create SECOND cluster clone

The main session's cluster clone is at `/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1/`.
You CANNOT use that — they're checking out `task2-prompt` and running jobs there.
Create your own clone:

```bash
scripts/juelich_exec.sh "cd /p/scratch/training2615/kempinski1/Czumpers && git clone git@github.com:oszypczy/hackaton.git repo-kempinski1-cd 2>&1 | tail -5"
scripts/juelich_exec.sh --force "cd /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd && git checkout task2-prompt-cd && git pull origin task2-prompt-cd"
```

If `git clone` complains about SSH/GitHub — copy the existing clone instead:
```bash
scripts/juelich_exec.sh --force "cp -r /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1 /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd && cd /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd && git checkout -B task2-prompt-cd && git pull origin task2-prompt-cd"
```

**RULE:** all your sbatch jobs run from `repo-kempinski1-cd/`, ALL output paths in main.sh must be updated.

### 4. Verify shadow_lmm available

```bash
scripts/juelich_exec.sh "ls /p/scratch/training2615/kempinski1/Czumpers/P4Ms-hackathon-vision-task/shadow_lmm/ | head"
```

Should show config.json, non_lora_trainables.bin, etc. — same structure as `target_lmm/`.

## Implementation (~1.5 h)

### Files you edit (in `/Users/arturkempinski/hackaton-cd/code/attacks/task2/prompt/`):

#### A. `attack.py` — add CD generation function

Add a new function `generate_one_cd()` that does NOT use `model.generate()` (which is overridden in unified_mllm.py and may not accept LogitsProcessor cleanly). Instead, manual token-by-token decode loop:

```python
@torch.no_grad()
def generate_one_cd(
    target_model, amateur_model,
    tokenizer, image_processor, image_size,
    get_formatted_question, sample,
    max_new_tokens=50, alpha=1.0, beta=0.5, plausibility_topk=50,
    image_mode="blank",
):
    """Contrastive Decoding: target − amateur, plausibility-filtered to expert top-k."""
    from strategies import STRATEGIES
    # Build prompt — use direct_probe (top strategy) as the prompt template
    prompt_text = STRATEGIES["direct_probe"](sample, get_formatted_question, tokenizer)
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_text))
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=target_model.device)

    image_tensor = _build_image_tensor(
        sample.image_bytes, image_size, image_processor, image_mode,
        user_id=sample.user_id,
    ).to(target_model.device)

    # IMPORTANT: target uses unified_mllm's prepare_multimodal_inputs; amateur (shadow_lmm)
    # has SAME architecture so it accepts the same call. We must call the SAME multimodal
    # prepare path on both. Use the underlying forward via prepare_multimodal_inputs and
    # then manual decoding step.
    #
    # Strategy: call BOTH models' generate() once with max_new_tokens=1, get logits, combine,
    # pick token, append to input_ids, repeat. This is slow (no KV cache reuse) but correct.
    # Alternative: dive into prepare_multimodal_inputs to extract inputs_embeds and run
    # both models' raw forward with use_cache=True. Pursue this only if naive loop too slow.

    generated = []
    for step in range(max_new_tokens):
        # Forward both models — returns logits over vocab for last position
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t_logits = _forward_logits(target_model, input_ids, image_tensor, tokenizer)
            a_logits = _forward_logits(amateur_model, input_ids, image_tensor, tokenizer)

        # Plausibility filter: only consider tokens in top-k of EXPERT (target)
        topk_vals, topk_idx = t_logits.topk(plausibility_topk, dim=-1)
        mask = torch.full_like(t_logits, float("-inf"))
        mask.scatter_(-1, topk_idx, 0.0)

        # Combine
        cd_scores = alpha * t_logits - beta * a_logits + mask  # masks out non-plausible
        next_token = cd_scores.argmax(dim=-1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def _forward_logits(model, input_ids, image_tensor, tokenizer):
    """Single forward pass through unified_mllm, return last-position logits."""
    # The unified_mllm has its own .forward that may need the same multimodal-batch
    # interface as .generate. Inspect src/lmms/models/unified_mllm.py or use its
    # `prepare_multimodal_inputs` path. Pseudocode:
    out = model(
        batch_input_ids=[input_ids[0]],
        batch_labels=[torch.full_like(input_ids[0], -100)],
        batch_X_modals=[{"<image>": image_tensor}],
    )
    # `out.logits` shape: (1, seq_len, vocab) — return last position
    return out.logits[0, -1, :]
```

**CRITICAL:** the `_forward_logits` helper depends on the unified_mllm internals. Study `src/lmms/models/unified_mllm.py` line ~99 (`generate` override). The `forward` likely has the same multimodal-batch signature. If not, you may need to call `prepare_multimodal_inputs` manually and pass `inputs_embeds` to the LM.

**FALLBACK if manual loop too slow** (>2× direct_probe time):
- Use HF `LogitsProcessor` API. Subclass and pass to `model.generate()`. The unified_mllm's generate may forward `logits_processor` to the LM's actual generate — check `unified_mllm.py:99` source.
- Or implement KV-cache reuse (forward only the last new token at each step instead of full sequence).

#### B. `main.py` — add CD mode

Add CLI flag `--use_cd` (or new strategy `cd_decoding`). When set:
1. Load BOTH `target_lmm/` and `shadow_lmm/` via `load_model_and_tools` (extend it to optionally return amateur).
2. Replace `generate_one(...)` call with `generate_one_cd(target, amateur, ...)`.
3. Log α/β/plausibility_topk in run_log.csv.

#### C. `main.sh` — point to second clone + add CD flag

```bash
ATTACK_DIR="/p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd/code/attacks/task2/prompt"
# ... rest same ...
# Add fifth arg or env var for CD mode + α/β
```

Output paths use `repo-kempinski1-cd/` not `repo-kempinski1/`.

#### D. `code/attacks/task2/prompt/strategies.py` — NO CHANGES NEEDED

CD reuses `direct_probe` as prompt template. Don't touch strategies.py (other session may edit it for K-shot wiring; merge conflict risk).

## Testing & Eval (~30 min eval)

### 1. Smoke test (5 min)

```bash
scripts/juelich_exec.sh --force "cd /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd/code/attacks/task2/prompt && sbatch main.sh eval 20 blank cd_decoding"
```

Wait for completion, check log `output/log_<jobid>.txt`. Verify:
- Both models load
- Generation produces non-empty strings
- ETA ≤ 2× direct_probe ETA (ok if 4× — then optimize KV cache)

### 2. Full eval blank-mode 840 (~28 min if 2× direct_probe, ~56 min if 4×)

```bash
scripts/juelich_exec.sh --force "cd /p/scratch/training2615/kempinski1/Czumpers/repo-kempinski1-cd/code/attacks/task2/prompt && sbatch main.sh eval - blank cd_decoding"
```

Use `Monitor` tool with `until ! squeue -j <id> | grep -q .; do sleep 30; done; echo DONE` for completion notification.

### 3. Gate decision

Compare scores against direct_probe (CREDIT 0.245 / EMAIL 0.579 / PHONE 0.370 / OVERALL 0.398):
- Pass: OVERALL ≥ 0.418 (+0.02 gate)
- If pass → predict task/ (cost ≈ 2× direct_probe predict = 104 min)
- If fail → ablate β ∈ {0.5, 1.0, 1.5} on 100-sample subset; if no config beats DP → kill CD path, report

### 4. Hyperparameter ablation (only if first config underperforms)

```bash
sbatch main.sh eval 100 blank cd_decoding   # default α=1.0 β=0.5
# After: edit main.sh to take β as 5th arg, then:
sbatch main.sh eval 100 blank cd_decoding 1.0
sbatch main.sh eval 100 blank cd_decoding 1.5
```

## Coordination rules with other session

| Resource | Yours | Theirs |
|---|---|---|
| Branch | `task2-prompt-cd` | `task2-prompt` |
| Local worktree | `/Users/arturkempinski/hackaton-cd/` | `/Users/arturkempinski/hackaton/` |
| Cluster clone | `repo-kempinski1-cd/` | `repo-kempinski1/` |
| Output dir | `repo-kempinski1-cd/code/attacks/task2/prompt/output/` | `repo-kempinski1/code/attacks/task2/prompt/output/` |
| Insights file | edit on YOUR branch only | edit on theirs |

**Hard rules:**
- ❌ NEVER submit at the same time. 5-min cooldown is shared per team. Coordinate via user.
- ❌ NEVER edit shared infra (`docs/`, `Justfile`, `requirements.txt`) — pull from main if needed
- ❌ NEVER push to `task2-prompt` from your worktree
- ✅ Update `findings/contrastive_decoding_plan.md` with progress / results as you go
- ✅ Cross-reference `findings/v3_plan.md` for overall coordination
- ✅ When you submit (if you submit), put method tag in CSV filename (`task2_pii_v3_cd.csv`)

## Submission flow

When eval passes gate:
1. Predict task/ (104 min)
2. `just pull-csv task2 --branch task2-prompt-cd` (you may need to add a flag — fall back to `scripts/pull_csv.py` direct call with custom path)
3. Validate CSV locally (3000 rows, length 10–100, no embedded newlines)
4. **CHECK with main session** before submit — if their K-shot pipeline is also ready to submit, coordinate timing
5. `just submit task2 submissions/task2_pii_v3_cd.csv`

## Known risks

1. **shadow_lmm may have different config.** First step: load both, verify tokenizer identical, embed dim identical. If not — we need to align via interpolation OR use base OLMo-2-1B from HF (no LLaVA-HR vision; CD on text-only path may be tokenizer-mismatched).
2. **unified_mllm.forward may not accept logits_processor.** Manual decoding loop is the workaround.
3. **Memory:** 2× 3.6 GB models on A800. Should fit in 40 GB. If OOM, reduce max_new_tokens to 30 or move amateur to CPU offload.
4. **Time budget:** if CD eval takes 4× direct_probe = ~56 min, that's still OK. If 8× → kill, optimize KV cache or move on.
5. **Tokenizer drift:** if shadow_lmm has any tokenizer modification, amateur logits must be aligned to target's vocab — re-verify after load.

## Expected lift

Per research §2.1: contrastive decoding for memorization extraction shows strong empirical results. SaTML-2023 LLM extraction challenge winner used CD + beam search.
- Expected: +0.03 to +0.07 OVERALL (on top of direct_probe 0.398)
- Optimistic: blank 0.398 → 0.45–0.47 → LB 0.43–0.45
- If CD is the right tool for this overfit setup, it's the highest-EV move available.

## What to do RIGHT NOW

1. Create branch + worktree (Setup §1)
2. Verify cluster socket (Setup §2)
3. Create second cluster clone (Setup §3)
4. Verify shadow_lmm (Setup §4)
5. Read `attack.py`, `main.py`, `strategies.py`, and the codebase `src/lmms/models/unified_mllm.py` to understand the multimodal forward path
6. Implement Step A (`generate_one_cd` in attack.py) — start with the manual decode loop
7. Smoke test (eval 20 samples)
8. If smoke OK → full eval (840 blank)
9. Report result, coordinate with main session before submit

Estimated wall-clock: 1.5h implementation + 30 min eval = **~2h to gate decision**. Same wall-clock as the K-shot path → we test both within 2h.

Good luck. Report progress in this file.

## Progress (CD session)

### 2026-05-09 ~23:30 — implementation done, smoke pending

Setup:
- Worktree `/Users/arturkempinski/hackaton-cd/`, branch `task2-prompt-cd` (off task2-prompt @ e0a9313)
- Cluster clone `/p/scratch/.../Czumpers/repo-kempinski1-cd/` cp'd from main clone, switched to task2-prompt-cd
- shadow_lmm config diff vs target: IDENTICAL → tokenizer/embed shape compatible

Implementation (commit `19c793a`):
- `attack.py:generate_one_cd` — manual decode loop with KV-cache reuse
  - first step: prepare_multimodal_inputs on both models, super().forward(inputs_embeds=…) to get full-prompt logits + past_key_values
  - subsequent steps: forward(input_ids=[next_token], past_key_values=past) → 1-token incremental decode
  - CD score: plausibility filter (expert top-k=50) → α·target − β·amateur, argmax
  - α=1.0 β=0.5 default (standard CD config)
- `main.py`: `--use_cd` / `--shadow_model_dir` / `--cd_alpha` / `--cd_beta` / `--cd_topk`
- `main.sh`: STRATEGY=cd_decoding short-circuit → uses CD_TEMPLATE (default direct_probe) + sets --use_cd + reads CD_ALPHA/CD_BETA/CD_TOPK env vars
- `_patch_attn_no_flash` made idempotent (CD path loads target+amateur, double-patch was wasteful)

Smoke test: `sbatch main.sh eval 20 blank cd_decoding` → job 14740097 PD (Priority).
Cancelled + resubmitted as 14740220 with --time=00:15:00 (backfill kicked in immediately).

### 2026-05-09 ~23:25 — smoke 20 OK, full 840 running

Smoke 14740220 (20 samples, blank, α=1.0 β=0.5 topk=50, time 0:42):
- CREDIT  0.0719  (DP baseline ~0.245, **−0.17**)
- EMAIL   0.5549  (DP ~0.579, −0.02)
- PHONE   0.3173  (DP ~0.370, −0.05)
- OVERALL 0.3146  (DP 0.398,  −0.08)

Hypothesis: shadow_lmm produces *the same 4-4-4-4 placeholder format on CREDIT* as
target (both trained on the same generic PII pattern), so β·amateur cancels
the very tokens we want. EMAIL/PHONE less affected (more diverse memorized
content). N=7 per CREDIT bucket → high variance, full 840 needed.

Pipeline verified: KV-cache reuse works, both models load via two
`load_model_and_tools` calls, idempotent attn patch holds. ~2 sec/sample.

Full 840 eval: job 14740294 (--time=00:45:00).
Gate: OVERALL ≥ 0.418 (= DP 0.398 + 0.02 noise floor) → predict task/. If fail,
ablate β ∈ {0.3, 1.0} or per-PII route (DP on CREDIT, CD on EMAIL/PHONE).

### 2026-05-10 ~00:00 — full 840 FAILED gate, CD loses on all 3 types

Job 14740294 (840 samples, blank, α=1.0 β=0.5 topk=50, 29.9 min):
- CREDIT  0.0832  (DP 0.2312, **−0.148**, −64% relative)
- EMAIL   0.4998  (DP 0.5785, **−0.079**, −14%)
- PHONE   0.3108  (DP 0.3700, **−0.059**, −16%)
- OVERALL 0.2980  (DP 0.3932, **−0.095**)

CD is **uniformly worse** — per-PII routing won't save it.

Diagnosis (matches research §2.1 caveats):
The research recommends "stock OLMo-2-1B-base (no PII fine-tune)" as amateur.
Our shadow_lmm is **fine-tuned on the same PII-VQA task with disjoint PII** —
i.e. it produces the same formatted output (4-4-4-4 for CREDIT, name@... for
EMAIL, +1... for PHONE) just without the *content* the target memorized.
β·shadow then cancels both format AND any tokens the shadow happens to
predict via format prior, dragging target's memorized content below the
plausibility cutoff.

Cluster has `allenai/OLMo-2-0425-1B` (true base) in cache, BUT integration
requires bridging unified_mllm's multimodal pipeline with a text-only LM
that has no `<image>` special token and a different vocab size. ~1h code,
uncertain payoff.

### Decision: kill CD path, no submit

OVERALL 0.298 is far below v2 direct_probe submitted score (0.381 LB).
Submitting would regress by ~0.08 — not worth it. Hand off to:
- main session (K-shot + Lev medoid) for CREDIT-dominant gains via §3.x
- v2 direct_probe baseline (0.381 LB) as floor

Time remaining ~12h. Higher-EV moves: monitor main session results,
potentially merge CSVs (DP-CREDIT + CD-EMAIL/PHONE was the only routing
that *could* have helped, but CD lost on those too).
