# Skip the vector DB: a 24-hour ML security hackathon RAG verdict

**For a 3-person team, MacBook M4, 19 papers + 200 MB code, and likely fewer than 100 queries, the right answer is not RAG — it's a curated `MAPPING.md` router plus Claude Code's native Read tool with explicit 1-hour prompt caching.** That stack gets you running in under two hours, costs roughly $2 for 50 questions on Sonnet 4.6, and avoids the 6–13 hours of pipeline maintenance a real RAG system burns on a 24-hour clock. If you must have semantic search (e.g., for the 200 MB PyTorch code), the only configuration worth shipping in a hackathon is **Qdrant in Docker plus the official `mcp-server-qdrant`, indexed with `voyage-code-3`** — both are free at this corpus size and have the cleanest snapshot/recovery story. Anthropic itself recommends the no-RAG path for any corpus under 200k tokens; your 19-paper text-only corpus is roughly 380k tokens — borderline but tractable with a section-level router. The deeper reason: prompt caching at 1-hour TTL makes a cached re-read of a full paper cost about **$0.018 per 60k-token PDF on Sonnet 4.6**, which is cheaper than maintaining a vector index unless you intend to ask hundreds of questions.

## The April 2026 baseline that changes the math

Anthropic's pricing and Claude Code's native capabilities have shifted the RAG-vs-long-context calculus significantly since 2024. **Sonnet 4.6 lists at $3 input / $15 output per million tokens, with cache reads at $0.30/M (a 90% discount) and 1-hour cache writes at $6/M**; Haiku 4.5 lands at $1/$5 with $0.10 cache reads. Opus 4.6 dropped 3× in February 2026 to $5/$25, putting 1M-token context inside reach at standard rates on Sonnet 4.6 and Opus 4.6+. Critically, **prompt caching is automatic in Claude Code sessions** — once a file is read, every subsequent turn re-uses it at the cached-read rate without explicit `cache_control` blocks. There is one important regression: around early March 2026 the Claude Code default cache TTL silently moved from 1 hour to 5 minutes (issue #46829). For API scripts, you must explicitly pass `"cache_control": {"type": "ephemeral", "ttl": "1h"}` or pay the cache-write cost on every long iteration cycle.

Claude Code's Read tool now natively ingests PDFs through the vision pipeline, but with hard limits that matter for academic papers. **A typical 30-page text-heavy ML paper consumes 45,000–60,000 tokens** when read via the vision path (Simon Willison measured a real 15 MB paper at 60,934 tokens on Opus 4.7 in April 2026); the same content as `.txt` from `pdftotext` is roughly 20,000–35,000 tokens. Read enforces a **20-page-per-call ceiling and a 25,000-token hard cap per file**, so any paper over ~17 pages requires the `pages: "1-15"` parameter or pre-conversion to text. Grep and Glob are ripgrep-backed and trivially handle the 200 MB code corpus in sub-second times — but **Grep cannot search inside PDFs**; you must pre-extract text first.

Concrete cost for a representative read pattern of one 60k-token PDF and five questions on Sonnet 4.6: **roughly $0.34 with caching versus ~$1.00 without**. Re-read the same paper 10 times in a session and you pay $0.34 versus $1.80 — about 81% savings, automatic.

## Where RAG actually wins (and where it's a trap)

The token math superficially favors RAG: a top-5 chunk retrieval at 512 tokens each is ~3k input tokens versus ~20k for a full paper read — an 85% nominal saving. **That advantage collapses once prompt caching enters the picture**: a cached full-paper read costs ~$0.006 on Sonnet 4.6, which is *cheaper* than a typical RAG retrieval+synthesis cycle once you account for query-embedding cost and reranker fees. Anthropic's own Contextual Retrieval blog post explicitly states: *"if your knowledge base is smaller than 200,000 tokens (about 500 pages), you can just include the entire knowledge base in the prompt… no need for RAG."* Your text-only corpus is roughly 380k tokens spread across 19 papers — too big for one shot but cleanly partitioned: a router + per-paper read pattern fits the regime perfectly.

The trap with RAG on technical ML papers is well-documented and shows up exactly where you'd query in a security hackathon. **Naive top-20 retrieval has a 5.7% failure rate even on technical corpora** in Anthropic's eval; contextual embeddings cut that to 3.7%, hybrid BM25 to 2.9%, and a reranker to 1.9% — a 67% total reduction, but each layer adds engineering. Failure modes that hit ML security work specifically: querying for an algorithm and surfacing only its related-work mention, equation-heavy spans broken across chunk boundaries, and cross-paper synthesis questions ("compare threat models in CDI versus LiRA") that structurally defeat top-k retrieval because they need *recall on every relevant paper simultaneously*. Chroma's 2025 *Context Rot* research further showed that even *correct* retrieval at long context can produce 20+ percentage-point accuracy drops on Claude — but bad retrieval is worse, because the model treats wrong chunks as ground truth and confidently hallucinates.

The break-even is unforgiving. **RAG fixed cost is roughly 8 hours of engineering plus ~$1 of embedding versus 1 hour for a MAPPING.md approach.** On pure token economics for a 20-page paper, RAG saves about $0.05 per uncached read or $0.001 per cached read; recouping a $1 embedding bill takes about 20 uncached queries or *1,000 cached queries*. With Anthropic prompt caching active, **break-even moves into the high hundreds of queries** — well past anything a 24-hour hackathon will produce. Hamel Husain, Jason Liu, and Eugene Yan all independently arrived at the same 2025 consensus: naive single-vector RAG is dead, but smart retrieval (agentic search, curated indexes, hybrid methods) is more important than ever — and "even manually copy-pasting text into a prompt is, technically, a form of RAG."

## Token economics for "How does CDI's statistical test work?"

Three paths, priced on Sonnet 4.6 ($3/M input, $15/M output, $0.30/M cache reads):

| Path | Input tokens | Output | First-call cost | Cached-call cost |
|---|---|---|---|---|
| Full PDF read (vision, 60k-token paper) | 60,000 | 800 | $0.192 | $0.030 |
| Full PDF as text (`pdftotext` first, 25k tokens) | 25,000 | 800 | $0.087 | $0.0125 |
| Vector retrieval, top-5 chunks @ 512 tok | ~3,000 | 800 | $0.021 | n/a |
| MAPPING.md router only (~1,000 tokens) | 1,000 | 200 | $0.006 | $0.0003 |
| Router → load 1 specific paper section (~6k tokens) | 7,000 | 800 | $0.033 | $0.005 |

The router-then-section pattern hits a sweet spot the literature rarely names explicitly: **roughly $0.005 per warm-cache query, no embedding pipeline, and full citation provenance** because you actually loaded the canonical section.

## The recommended stack: Option A wins

Three plausible architectures, ordered by simplicity:

**Option A — Pure Claude Code + MAPPING.md (RECOMMENDED).** A hand-curated 1k–2k token index file describing each paper's contributions, sections, key terms, and "go here for X" pointers. Claude Code reads the mapping, selects the relevant paper(s), and invokes Read with a `pages` range. Zero infrastructure, full citation fidelity, survives total network loss after initial PDF download, and exploits Claude Code's automatic caching. Setup: 60–90 minutes for one teammate to skim the 19 papers and write `MAPPING.md`. **This is the right answer for your scenario.**

**Option B — Claude Code + local Qdrant via MCP (middle ground).** Use this only if you genuinely need semantic search across the 200 MB PyTorch code (where ripgrep falls short for "find all places that implement a LiRA-style shadow-model attack" type queries). Qdrant in Docker with `mcp-server-qdrant` (official, MIT) is the cleanest M4-ARM option: multi-arch image runs natively, 3 teammates can connect concurrently to one shared instance over gRPC, and the snapshot API gives you real dump/restore. Pair it with **`voyage-code-3` for embeddings** — the first 200M tokens are free, which fully covers the ~50M tokens in this corpus, and it beats OpenAI text-embedding-3-large by 13.8% on code retrieval benchmarks.

**Option C — Full pipeline (Marker + contextual retrieval + hybrid + rerank).** Anthropic-grade retrieval (-67% failure rate). Reserve for post-hackathon. The setup time alone (6–13 hours) eats too much of your 24h budget and the marginal accuracy gain over Option A on a 19-paper corpus is below the noise floor of question quality.

The single recommendation: **start with Option A. Add Option B's code search only if you find yourself grepping the PyTorch repos more than 10 times in the first 4 hours.**

## Setup recipe — the MAPPING.md path (under 2 hours)

```bash
# One-time prep on each M4
brew install uv ripgrep poppler   # poppler ships pdftotext
mkdir -p corpus/{pdf,txt,code,specs} && cd corpus

# Pre-extract text from PDFs in parallel (8 workers, ~5 min for 19 papers)
ls pdf/*.pdf | xargs -P 8 -I {} sh -c 'pdftotext -layout "{}" "txt/$(basename {} .pdf).txt"'

# Build MAPPING.md skeleton automatically, then hand-edit
for f in txt/*.txt; do
  echo "## $(basename $f .txt)"
  head -50 "$f" | sed -n '/[Aa]bstract/,/[Ii]ntroduction/p'
  echo ""
done > MAPPING.md
```

Then hand-edit `MAPPING.md` to be a *router*, not a summary. The format that works in practice:

```markdown
## CDI: Confidence-based Distribution Inference (2024)
- File: pdf/cdi.pdf  | Pages: 12  | Tokens: ~28k
- Topic: black-box statistical test for membership inference
- Key sections: §3 (test statistic, pp. 4-6), §4 (theoretical analysis, pp. 6-8), §5 (eval, pp. 8-11)
- Use this paper for queries about: KS-test variants, calibrated confidence, MIA without shadow models
- Cross-refs: extends LiRA (Carlini 2022) → see lira.pdf §2
```

Add to your repo's `CLAUDE.md`: *"For any question about the corpus, first read `MAPPING.md`, then use Read with `pages` parameter on the specific section. Never read more than 2 papers per turn."* That single instruction routes Claude Code's behavior reliably and exploits caching automatically.

## Setup recipe — Option B if you actually need semantic code search

```bash
# 1) Vector DB — Qdrant, multi-arch ARM image, runs natively on M4
mkdir -p ~/qdrant_storage
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# 2) Python env
uv venv && source .venv/bin/activate
uv pip install qdrant-client voyageai marker-pdf tree-sitter-language-pack

# 3) Index code with voyage-code-3 (free under 200M tokens)
export VOYAGE_API_KEY=...
python <<'PY'
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import voyageai, pathlib, hashlib

vo = voyageai.Client()
qc = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
qc.recreate_collection("code", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

# Naive baseline: file-level chunking; upgrade to tree-sitter later
files = list(pathlib.Path("repos").rglob("*.py"))
batch, points = [], []
for i, f in enumerate(files):
    text = f.read_text(errors="ignore")[:30000]
    batch.append(text)
    if len(batch) == 32 or i == len(files)-1:
        embs = vo.embed(batch, model="voyage-code-3", input_type="document",
                        output_dimension=1024).embeddings
        for t, e in zip(batch, embs):
            points.append(PointStruct(id=hashlib.md5(t.encode()).hexdigest()[:16],
                                      vector=e, payload={"text": t[:2000]}))
        batch = []
qc.upsert("code", points=points)
qc.create_snapshot(collection_name="code")  # backup
PY

# 4) Wire MCP server into Claude Code (~/.claude/mcp.json)
cat > ~/.claude/mcp.json <<'JSON'
{ "mcpServers": {
    "qdrant": { "command": "uvx", "args": ["mcp-server-qdrant"],
      "env": { "QDRANT_URL": "http://localhost:6333",
               "COLLECTION_NAME": "code",
               "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2" } } } }
JSON
```

A teammate then queries from inside Claude Code with: *"Use the qdrant tool to find code implementing LiRA-style shadow models, then read the top result."* The MCP server exposes only two tools (`qdrant-store`, `qdrant-find`), keeping the per-turn schema overhead under ~500 tokens — acceptable. **Do not use ChromaDB's PersistentClient from three laptops on a shared folder**: documented HNSW cache-thrashing and lock corruption (TribalScale's MemPalace post-mortem). If you prefer Chroma over Qdrant, run `chroma run --path ./db` once and have all three teammates use `HttpClient`.

## Embedding choice for this corpus

For mixed PyTorch code plus academic prose, **voyage-code-3 dominates** at $0.18/M tokens with the first 200M free, beating OpenAI text-embedding-3-large by 13.8% and CodeSage-large by 16.8% on 32 code-retrieval benchmarks. 32k context handles full source files without aggressive chunking. If offline operation matters, **Qwen3-Embedding-4B via mlx-embeddings** is the strongest local pick — Apache 2.0, 32k context, runs natively in MLX with Qwen3 support added in v0.0.4 (Sept 2025), hitting tens of thousands of tokens/sec on M4 with 4-bit DWQ quants. BGE-M3 is the fast lightweight fallback (568M params, MIT, hybrid dense+sparse — its sparse component is excellent for security queries with literal terms like CVE IDs and attack names). Skip OpenAI text-embedding-3-* for code-heavy corpora and skip Cohere embed-v3 (the 512-token cap is too tight for source files); Cohere embed-v4 with 128k context is competitive but more expensive than Voyage.

## PDF extraction: only matters if you go RAG

For the MAPPING.md path, `pdftotext -layout` is sufficient. **For RAG, Marker 1.9+ from Datalab is the SOTA on academic content** — 95.67% on its own benchmark, #1 on olmOCR-Bench beating GPT-4o, Mistral OCR, and Deepseek-OCR. Marker's killer feature is per-block typed output (`SectionHeader`, `Equation`, `Code`, `Caption`) that drops directly into section-aware chunking. Real M4 throughput is the catch: ~2–4 seconds per page on MPS because TableRec is CPU-only. For 19 papers averaging 30 pages, that's 30–60 minutes wall time across 4–8 worker processes — fits inside a 1-hour index budget.

```bash
uv pip install marker-pdf
export TORCH_DEVICE=mps
ls pdf/*.pdf | xargs -P 4 -I {} marker_single "{}" --output_format markdown --output_dir md/
```

`pymupdf4llm` is 30–50× faster (~0.09 s/page) but loses equations and code structure — useful as a quick first pass to validate your pipeline. Nougat is effectively unmaintained since 2023; do not use. Docling is the strong second choice (MIT license, MLX acceleration on Apple Silicon, 88.2% on opendataloader benchmark) and is the right call if Marker's GPL-3 + AI Pubs Open-RAIL-M licensing concerns you.

## Chunking, if you commit to RAG

The validated stack from Anthropic's own evaluations: **800-token chunks with 100-token overlap, retrieving top-20**, on top of section-aware splitting from Marker's typed blocks. Add Anthropic Contextual Retrieval — prepending a 50–100 token "situating" prefix generated by Haiku 4.5 with the full document prompt-cached — at a cost of roughly $1 per 1M document tokens, or about $0.001 per 80-page paper. Layered with hybrid BM25 and a reranker (Voyage rerank-2.5 at $0.05/M, free under 200M, ~40 ms latency), this stack hits Anthropic's measured 1.9% top-20 retrieval failure rate. **Late chunking (Jina v3/v4) helps when documents fit the embedder's 8k context window**; for 80-page papers it requires sliding windows that partially defeat the technique. Tree-sitter via LlamaIndex `CodeSplitter` at function-level granularity (40 lines, 15 line overlap, 512 max tokens) is the right approach for Python repos.

## Fallback plan — what survives a hackathon emergency

The MAPPING.md path is offline-resilient by construction: PDFs are local files, Read works fully offline within Claude Code if API connectivity holds, and the only outside dependency is the Anthropic API itself. Four concrete failure modes and recoveries:

1. **Vector DB corruption** — restore from `qdrant.snapshot` (created above) in ~30 seconds, or `cp -r qdrant_storage.backup qdrant_storage`. LanceDB has built-in `table.checkout(version=N)` time-travel as a stronger alternative if you chose Option B's variant.
2. **Embedding API outage** — keep Qwen3-Embedding-0.6B (4-bit DWQ, ~600 MB) downloaded locally as `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`; reindex from cached chunks in ~2 minutes for the full corpus.
3. **Claude API outage** — fall back to the teammate's external CUDA box running a local model (Llama 3.3 70B or Qwen 2.5 72B via vLLM). Keep the same MAPPING.md interface; only the inference endpoint changes.
4. **Total network loss** — `pdftotext`-extracted `.txt` files plus `MAPPING.md` plus `ripgrep` will answer 70% of corpus questions without any LLM. This is the actual floor of capability and it's surprisingly high.

Critical 24h discipline: **commit `MAPPING.md`, `qdrant_storage/`, all extracted text, and the embedding script to git every 2 hours**. Index corruption that takes 4 hours to debug is the most common silent killer in hackathon RAG setups.

## What this means in practice

The shift from 2024 to April 2026 is real and underappreciated: **prompt caching at $0.30/M reads on Sonnet 4.6 has destroyed RAG's economic moat for small corpora**. A 19-paper corpus with 100 questions costs roughly $2 with the recommended approach versus ~$1.50 with a fully-tuned RAG stack — a 25% saving that is dwarfed by the 6–10 hour engineering delta. The vector DB only earns its place when (a) you genuinely need semantic search over a code corpus where ripgrep fails, (b) the corpus exceeds 200k tokens *per query plan*, or (c) you're past the 24-hour mark and have a stable working baseline. For a 3-person ML security hackathon team on M4 Macs, **write the MAPPING.md, set `ttl: "1h"` explicitly on your scripted API calls, keep a Qdrant+Voyage path warmed up as a stretch goal, and spend the saved hours on the actual security questions you came to answer.**