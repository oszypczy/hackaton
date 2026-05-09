# JURECA setup dla teammate'ów (szypczyn1 / multan1 / murdzek2)

Owner (kempinski1) odpalił już `hackathon_setup.sh` → folder
`/p/scratch/training2615/kempinski1/Czumpers/` istnieje, ma ACL na 4 osoby
(kempinski1 + Ty), zawiera 3 datasety i 3 venv'y per-task.

Teraz ty wykonujesz **3 kroki na swoim koncie Jülich** (jednorazowo, ~5 min).

---

## Krok 1 — SSH na JURECA

```bash
ssh <twój-username>@jureca.fz-juelich.de
# MFA: TOTP z aplikacji (Authenticator)
```

Jeśli używasz `juelich_connect.sh` z naszego repo lokalnie — odpal go zamiast tego.

## Krok 2 — `source teammate.sh` (aktywacja shared env)

```bash
cd /p/scratch/training2615/kempinski1
source teammate.sh
```

Co to robi:
- `jutil env activate -p training2615` (ustawia `$PROJECT`, `$SCRATCH`)
- modyfikuje twój `~/.bashrc` (eksporty UV/HF cache → shared team folder)
- instaluje `uv` w twoim `~/.local/bin/` (jeśli brak)
- weryfikuje że masz dostęp do `Czumpers/`

## Krok 3 — SSH key dla GitHuba

JURECA blokuje outbound port 22, więc git po SSH idzie przez port 443.
Wklej całość poniżej w terminalu Jülich:

```bash
mkdir -m 700 -p ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$USER@jureca-hackathon-2026-05-09"

cat > ~/.ssh/config <<'EOF'
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519
EOF

cat ~/.ssh/id_ed25519.pub
```

Ostatnia linia wypisze twój **public key**. Skopiuj go i:
1. Otwórz https://github.com/settings/keys (twoje konto GH)
2. "New SSH key" → tytuł `jureca-<twój-username>-hackathon`, typ `Authentication Key`
3. Wklej public key → "Add SSH key"

Test:
```bash
ssh -T -o StrictHostKeyChecking=accept-new git@github.com
```
Powinno powiedzieć `Hi <twój-github-login>! You've successfully authenticated...`
(exit code może być 1 — to normalne, GitHub nie daje shell access).

## Krok 4 — Clone repo

```bash
cd /p/scratch/training2615/kempinski1/Czumpers
git clone git@github.com:oszypczy/hackaton.git repo-<twój-username>
cd repo-<twój-username>
git config user.name "<twój-username>"
git config user.email "<twój-email>"
```

Zamień `<twój-username>` na swój Jülich username (`szypczyn1` / `multan1` / `murdzek2`).
Zamień `<twój-email>` na email który chcesz używać w git commits.

---

## Workflow po setupie

**Edytujesz lokalnie** (laptop) → commit + push → na Jülichu `git pull` → run.

```bash
# na laptopie
git add code/attacks/task1_duci/main.py
git commit -m "feat: ..."
git push

# na Jülichu (twoje konto)
cd /p/scratch/training2615/kempinski1/Czumpers/repo-<twój-username>
git pull
sbatch code/attacks/task1_duci/main.sh    # albo python
```

---

## Folder structure na klastrze

```
/p/scratch/training2615/kempinski1/Czumpers/
├── DUCI/                       # Task 1 (dane + venv shared)
├── P4Ms-hackathon-vision-task/ # Task 2 (dane + venv shared)
├── llm-watermark-detection/    # Task 3 (dane + venv shared)
├── .uv/, .cache/               # shared cache
├── repo-kempinski1/            # owner's clone
└── repo-<twój-username>/       # ← TY (po kroku 4)
```

ACL: cała czwórka ma rwx do wszystkiego. Standardowo NIE edytuj cudzego
`repo-X/` — każdy ma swój clone, sync przez GitHub.

## Aktywacja venv per-task

```bash
# np. pracujesz nad Task 1 (DUCI)
cd /p/scratch/training2615/kempinski1/Czumpers/DUCI
source .venv/bin/activate
python /p/scratch/training2615/kempinski1/Czumpers/repo-<twój-username>/code/attacks/task1_duci/main.py
```

Albo `uv run` z folderu task'a.

## Submission jobu (sbatch template)

W repo `code/attacks/taskN_*/main.sh`:

```bash
#!/bin/bash
#SBATCH --account=training2615
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=30
#SBATCH --partition=dc-gpu
#SBATCH --output=/p/scratch/training2615/kempinski1/Czumpers/<task>/output/%j.out

cd /p/scratch/training2615/kempinski1/Czumpers/<task>
source .venv/bin/activate
srun python /p/scratch/training2615/kempinski1/Czumpers/repo-$USER/code/attacks/<task>/main.py
echo "done!"
```

Submit: `sbatch main.sh`. Output → `<task>/output/<jobid>.out`.

---

## Troubleshooting

**`ssh: connect to host github.com port 22: Connection refused`**
→ brak `~/.ssh/config` z `Port 443`. Wróć do Kroku 3.

**`Permission denied (publickey)` przy `git clone`**
→ klucz nie dodany w GitHub settings, lub zły identityfile:
```bash
cat ~/.ssh/config
ls -la ~/.ssh/id_ed25519*
ssh -vT git@github.com 2>&1 | grep -i identity
```

**`uv: command not found` po świeżym ssh**
→ wyloguj się i wróć (`exit` + nowe ssh), albo `source ~/.bashrc`.

**`fatal: not a git repository` w `Czumpers/`**
→ jesteś w shared folderze, nie w swoim clone. `cd repo-<twój-username>`.

**Konflikt na `git pull`**
→ ktoś inny pushnął coś sprzecznego z tym co ty edytowałeś. Najprościej:
```bash
git stash    # schowaj swoje
git pull
git stash pop    # wróć do swojego, możliwe konflikty do rozwiązania
```

---

Pingnij na grupie gdy skończysz — owner może wtedy push'nąć dalsze rzeczy
i widzieć czy ci się propaguje.
