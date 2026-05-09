# JURECA cluster workflow

Stan po setupie (2026-05-09):
`/p/scratch/training2615/kempinski1/Czumpers/` — shared team folder, ACL na 4 osoby
(kempinski1 + szypczyn1 + multan1 + murdzek2), 3 datasety pobrane, 3 venv'y
zbudowane (uv 0.11, Python 3.12, PyTorch 2.11 + CUDA 13).

## Mapa folderów

```
/p/scratch/training2615/kempinski1/Czumpers/
├── DUCI/                          # Task 1 dataset + venv (shared, read by all)
│   ├── DATA/                      # X.npy, Y.npy + per-model train indices
│   ├── MODELS/                    # 9 ResNet checkpoints
│   ├── task_template.py           # ORGANIZER submission boilerplate
│   ├── README.md                  # ORGANIZER spec (autoritative)
│   ├── requirements.txt
│   ├── .venv/                     # built by hackathon_setup.sh
│   └── output/                    # job outputs go here
├── P4Ms-hackathon-vision-task/    # Task 2 (PII)
│   ├── target_lmm/                # overfitted LMM (cel ataku)
│   ├── shadow_lmm/                # clean LMM (opcjonalny baseline)
│   ├── task/                      # 1000 samples × 3 questions = 3000 prompts
│   ├── validation_pii/            # 280 × 3 z ground-truth PII (lokalna walidacja)
│   ├── task2_standalone_codebase.zip  # white-box LMM source
│   ├── sample_submission.py
│   ├── sample_submission.csv
│   └── .venv/, output/
├── llm-watermark-detection/       # Task 3
│   ├── Dataset.zip                # train 360 / val 180 / test 2250
│   ├── generate_random_submission.py
│   ├── submission_template.py
│   └── .venv/, output/
├── .uv/                           # shared uv cache
├── .cache/                        # shared HF cache (HF_HOME)
├── repo-kempinski1/               # owner's git clone (per-user)
├── repo-szypczyn1/                # ← teammate (po ich setupie)
├── repo-multan1/
└── repo-murdzek2/
```

ACL pozwala pisać każdemu w każdym `repo-X/`, ALE konwencja zespołu jest twarda:
**każdy operuje TYLKO w swoim `repo-$USER/`**. Nie wchodzimy do cudzego.

## Per-user repo: kto co robi

Każdy z 4 teammate'ów ma **własny git clone** w `Czumpers/repo-<jego-username>/`.
To NIE jest 4 forki ani 4 branche — to są **4 niezależne working copies tego samego repo**
(`oszypczy/hackaton`, branch `main`).

Sync między nimi idzie **wyłącznie przez GitHub**:
- **Push** robisz **TYLKO z laptopa** (`git push origin main` z lokalnej kopii)
- **Pull** na klastrze robisz **TYLKO w swoim** `repo-$USER/` (`git pull` ściąga to co wszyscy popushali)
- **Push z klastra: NIE** — żadnego edytowania na klastrze i `git push`. To zaśmieca historię i kolej commitów.

```
Laptop teammate'a A         GitHub                Cluster /p/scratch/.../Czumpers/
─────────────────           ────────              ────────────────────────────
edytuj + commit             oszypczy/hackaton     repo-A/  ← pull przez konto A
git push          ──────►   main                  repo-B/  ← pull przez konto B
                   ◄──────                        repo-C/  ← pull przez konto C
                   ◄──────                        repo-D/  ← pull przez konto D
```

**Złote zasady:**
- ❌ Nie edytuj plików w cudzym `repo-X/` (nawet jeśli ACL pozwala — łamiesz konwencję, masakrowane historie git)
- ❌ Nie rób `git push` z klastra (push robisz z laptopa po commit'cie)
- ❌ Nie rób `git pull` w cudzym `repo-X/` jego kontem (potrzebujesz JEGO klucza SSH na JEGO koncie Jülich — i tak się nie da)
- ✅ Edycja zawsze lokalnie na laptopie → commit → push → pull w swoim klastrowym `repo-$USER/`
- ✅ Quick fix (np. `vim` na klastrze): tylko jeśli to throw-away test. Jak działa → odwzorować lokalnie + commit + push, NIE pushować z klastra.

## Branch model

| Branch | Owner | Zawartość |
|---|---|---|
| `main` | wszyscy | shared infra (docs, scripts, Justfile, requirements) |
| `task1` | owner taska 1 | wszystko związane z Task 1 (DUCI) |
| `task2` | owner taska 2 | wszystko związane z Task 2 (PII) |
| `task3` | owner taska 3 | wszystko związane z Task 3 (Watermark) |

**Polityka merge'a:**
- Edycja kodu taska → ZAWSZE na branchu `taskN`, nigdy bezpośrednio na `main`
- Edycja shared infra (Justfile, scripts, docs) → na `main` (krótkie commity)
- **Squash merge** `taskN` → `main` **DOPIERO na koniec hackathonu** (przed prezentacją).
  W trakcie 24h każdy task żyje na swoim branchu, submisje idą z brancha.
- Brak code-review/PR w trakcie 24h — bezpośredni push na branch ownera taska. Jakość weryfikowana lokalnie.

**Workflow z branchami:**

```bash
# Lokalnie (jednorazowo na początku swojego taska):
git checkout main && git pull
git checkout task1     # przełącz się na pre-bootstrapped branch
git pull origin task1  # synchro

# Lokalnie (każda iteracja):
git add code/attacks/task1_duci/main.py
git commit -m "feat(task1): describe change"
git push               # idzie na origin/task1

# Na klastrze (Twój repo-$USER, raz na sesję):
cd /p/scratch/training2615/kempinski1/Czumpers/repo-$USER
git fetch
git checkout task1     # tylko raz, potem już zostaniesz
git pull
```

**Pobieranie shared updates** (np. owner pushnął nową wersję submit.py na main):
```bash
git checkout main && git pull
git checkout task1
git merge main          # ściągasz zmiany z main na swój branch
# rozwiąż konflikty jeśli są
git push                # push do origin/task1
```

## Workflow per task (per-user ownership)

1. **Edytujesz LOKALNIE** w `code/attacks/<task>/main.py` na laptopie (na branchu `task<N>`)
2. **Commit + push** lokalnie (`git push origin task<N>`)
3. **Na klastrze:** `cd /p/scratch/training2615/kempinski1/Czumpers/repo-$USER && git pull`
   (**SWÓJ** `repo-$USER`, na branchu `task<N>` — nie cudzy)
4. **Run:** `sbatch code/attacks/<task>/main.sh` (lub `python` dla quick test)
5. **Output:** `Czumpers/<task>/output/<jobid>.out` + `submission.csv`
6. **CSV → laptop:** `scripts/juelich_exec.sh "cat Czumpers/<task>/submission.csv" > submissions/<task>.csv`
7. **Submit z laptopa** (klucz API żyje tylko w `.env` lokalnie): `just submit <task> submissions/<task>.csv`
   Patrz `docs/SUBMISSION_FLOW.md`.

## Cluster rules (z `Hackathon_Setup.md` § 7)

❌ Zakazane:
- zmiana ACL ręcznie na shared folders (`Czumpers/`, datasety)
- usuwanie shared folders / cudzych plików
- odtwarzanie `.venv` bez powodu (są pre-built przez owner setup)
- `pip install` / `uv pip install` w shared envie (`Czumpers/<task>/.venv/`) bez ostrożności — patrz uwaga niżej

✅ Dozwolone:
- modyfikacje w **swoim** `~/` (`~/.bashrc`, `~/.ssh/`, `~/.local/`) — skrypt setupowy SAM to robi
- praca w `Czumpers/repo-$USER/` (Twój clone)
- logi do `Czumpers/<task>/output/`
- używanie `Czumpers/<task>/.venv/` (read tylko dla Python interpreter)

⚠ Pip install w shared `.venv`:
- Skrypt setupowy ZBUDOWAŁ je z requirements.txt. Jeśli potrzebujesz extra paczki,
  możesz `uv pip install` ALE pamiętaj: zmienisz env dla całej czwórki.
- Bezpieczniejsze: per-user venv w swoim `repo-$USER/.venv-<task>/` jeśli pakiety mocno odmienne.
- Konsultuj z teamem zanim instalujesz cokolwiek w shared envie.

## sbatch template (per task)

```bash
#!/bin/bash
#SBATCH --account=training2615
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=30
#SBATCH --partition=dc-gpu
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/<TASK_DIR>/output/%j.out

# Activate per-task venv (z dataset folderu, NIE z repo)
cd /p/scratch/training2615/kempinski1/Czumpers/<TASK_DIR>
source .venv/bin/activate

# Code z naszego repo (per-user)
srun python /p/scratch/training2615/kempinski1/Czumpers/repo-$USER/code/attacks/<task>/main.py

echo "done!"
```

Submit job: `sbatch main.sh`. Status: `squeue -u $USER`. Cancel: `scancel <jobid>`
(`scancel` bez ID = blocked przez juelich_exec.sh).

## Quick test bez sbatch (dev iteration)

```bash
# Login node bez GPU — szybki sanity, NIE do treningu
cd /p/scratch/training2615/kempinski1/Czumpers/<TASK_DIR>
source .venv/bin/activate
python /p/scratch/training2615/kempinski1/Czumpers/repo-$USER/code/attacks/<task>/main.py --debug
```

Dla 30s smoke testu OK. Dla treningu/inference na full data — sbatch.

## Aktywacja sesji (po świeżym SSH)

```bash
jutil env activate -p training2615
# albo: source ~/.bashrc (jeśli sourcowałeś teammate.sh / hackathon_setup.sh wcześniej)
```

## Job zarządzanie (przez juelich_exec.sh)

```bash
# Status własnych jobów
scripts/juelich_exec.sh "squeue -u \$USER"

# Tail outputu live
scripts/juelich_exec.sh "tail -f /p/scratch/training2615/kempinski1/Czumpers/<task>/output/<jobid>.out"

# Submit job (ASKS [y/N] przez wrapper)
scripts/juelich_exec.sh "cd Czumpers/repo-\$USER && sbatch code/attacks/<task>/main.sh"

# Cancel pojedynczy job (ASKS [y/N])
scripts/juelich_exec.sh "scancel 12345678"
```

## Troubleshooting

**`fatal: not a git repository`** w shared folderze
→ jesteś w `Czumpers/`, nie w swoim clone. `cd repo-$USER`.

**`Permission denied`** przy pisaniu w cudzym folderze
→ ACL pozwala (rwx dla całej czwórki), ale konwencja: nie tykać `repo-<inny-user>/`.

**`uv: command not found`**
→ `~/.bashrc` nie zaktualizowany lub nie sourcowany. `source ~/.bashrc` lub re-ssh.

**`No GPUs available`** w jobie
→ `--gres=gpu:1` źle ustawione lub `--reservation=cispahack` źle. Sprawdź `sinfo -p dc-gpu`.

**Job nie startuje (`squeue` długo `PD`)**
→ kolejka pełna lub limit projektu. Sprawdź `sacct -u $USER --starttime=now-1hour`.
