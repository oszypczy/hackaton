# TODO — Hackathon CISPA Warsaw 2026-05-09/10

Hackathon startuje **2026-05-09 o 12:00**. Taski ogłaszane naraz.

## Przed startem (do 12:00)

- [ ] Każdy uruchamia `! scripts/juelich_connect.sh` — socket MFA na 4h
- [ ] Każdy ma `.venv` aktywny i `just eval` zielony
- [ ] Ustalić podział tasków (Data ID / Memorization / Watermark)

## Po ogłoszeniu tasków (12:00+)

1. Przeczytać PDF opisu każdego taska
2. Pobrać dane z HuggingFace / Jülich (`jutil env activate -p training2615`)
3. Easy baseline w pierwszej godzinie — coś na scoreboard szybko
4. Submit przynajmniej 3× per task (iterate)

## Submission
- REST API + CSV, team API token dostarczony na start
- Cooldown 5 min (2 min przy failed submission)
- Live scoreboard

## Compute
- Jülich: `sbatch` z `--partition=dc-gpu --account=training2615`
- Węzły: 4× A800 per node; partycja devel: `dc-gpu-devel`
- UV jako package manager na Jülich

## Kluczowe papiery do szybkiego dostępu
- Data ID: Maini et al. → `references/papers/txt/02_*.txt`
- Memorization: Carlini et al. → `references/papers/txt/01_*.txt`
- Watermark: Kirchenbauer et al. → `references/papers/txt/04_*.txt`
- CDI (diffusion): paper 09 → `references/papers/txt/09_*.txt`
