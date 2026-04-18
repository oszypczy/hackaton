# CISPA European Cybersecurity & AI Hackathon Championship - Warsaw
## Dokument przygotowawczy

### Podstawowe informacje
- **Data:** 9-10 maja 2026 (24h hackathon)
- **Miejsce:** Rektorska 4, Politechnika Warszawska
- **Format:** Zespoły 3-4 osobowe, 3 zadania do rozwiązania
- **Ocena:** Rozwiązania przesyłane na serwer CISPA, automatyczny scoring + live leaderboard + prezentacja przed jury
- **GPU:** Jülich Supercomputer Cluster (rejestracja: https://judoor.fz-juelich.de/projects/training2615)
- **Discord:** https://discord.gg/F2hPce55kK
- **Organizatorzy:** Adam Dziedzic & Franziska Boenisch (SprintML Lab, CISPA)

### Nagrody
- 1-3 miejsce: certyfikat + wejście do Grand Finale (lipiec 2026, St. Ingbert, Niemcy) z pokryciem kosztów podróży i zakwaterowania
- Grand Finale: 1. miejsce €4,000 | 2. miejsce €2,000 | 3. miejsce €1,200

---

## ANALIZA CHALLENGE'ÓW Z POPRZEDNICH EDYCJI

Na podstawie 5 poprzednich edycji (Paryż, Wiedeń, Sztokholm, Monachium, Barcelona) zidentyfikowano **powtarzające się kategorie challenge'ów**:

### Kategoria 1: ATAKI NA PRYWATNOŚĆ DANYCH TRENINGOWYCH
**Pojawiła się w:** Wiedeń (Dataset Inference), Sztokholm (Model Inversion), Barcelona (Fairness/Bias Inference)

Warianty:
- **Model Inversion** - rekonstrukcja danych treningowych z wytrenowanego modelu
- **Dataset Inference** - określenie czy konkretny zbiór danych był użyty do treningu
- **Membership Inference** - czy konkretna próbka była w zbiorze treningowym
- **Fairness Auditing** - wykrycie bias/nierówności w danych treningowych z odpowiedzi modelu (black-box)

### Kategoria 2: ATAKI ADVERSARIALNE / ROBUSTNOŚĆ MODELI
**Pojawiła się w:** Paryż (Adversarial Image Perturbation), Monachium (Model Robustness)

Warianty:
- **Adversarial Examples** - generowanie imperceptowalnych perturbacji powodujących misklasyfikację
- **Targeted Attacks** - zmuszenie modelu do klasyfikacji obrazu jako konkretna klasa
- **Techniki:** FGSM, PGD, C&W Attack, AutoAttack

### Kategoria 3: ATRYBUCJA / WATERMARKING
**Pojawiła się w:** Paryż (AI Image Attribution), Wiedeń (Watermark Removal), Monachium (Dataset Inference)

Warianty:
- **Model Attribution** - identyfikacja który model AI wygenerował dany obraz
- **Watermark Removal** - usuwanie niewidzialnych watermarków z obrazów
- **Watermark Detection** - wykrywanie watermarków w tekście generowanym przez LLM
- **Techniki:** analiza szumów, fingerprinting częstotliwościowy, tree-ring watermarks

### Kategoria 4: MODEL STEALING
**Pojawiła się w:** wywiad z organizatorami, jako jeden z głównych challenge'ów

Warianty:
- **Model Extraction** - klonowanie modelu ML przez API queries
- **Knowledge Distillation Attack** - odtworzenie funkcjonalności modelu
- **Black-box Attack** - bez wiedzy o architekturze modelu

---

## OBOWIĄZKOWE PAPERY (z maila organizatorów)

### 1. "Extracting Training Data from Diffusion Models" (Carlini et al., USENIX 2023)
- **Temat:** Modele dyfuzyjne zapamiętują i emitują dane treningowe
- **Metoda:** Dwuetapowy atak generate-and-filter; generowanie obrazów i filtrowanie tych, które odpowiadają membership inference
- **Wynik:** Wyekstrahowano >1000 przykładów treningowych z modeli SOTA
- **Wniosek:** Modele dyfuzyjne są znacznie mniej prywatne niż GANy
- **PDF:** https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf

### 2. "LLM Dataset Inference: Did you train on my dataset?" (Maini et al., NeurIPS 2024)
- **Temat:** Membership Inference Attacks (MIA) są zawodne → Dataset Inference jest lepszym podejściem
- **Problem:** MIA cierpią na distribution shift - sukces wynika z różnic w dystrybucjach, nie z faktycznej detekcji
- **Metoda:** Selektywne łączenie MIA dających pozytywny sygnał + test statystyczny na poziomie całego datasetu
- **Wynik:** p-values < 0.1 bez false positives na podzbiorach Pile
- **Link:** https://openreview.net/forum?id=Fr9d1UMc37

### 3. "Detecting Data Contamination in LLMs via In-Context Learning" (Zawalski et al., NeurIPS Workshop 2025)
- **Temat:** Detekcja kontaminacji danych treningowych w LLM
- **Metoda CoDeC:** In-context examples poprawiają wyniki na nieznanych danych, ale POGARSZAJĄ na zapamiętanych (zakłócają wzorzec memoryzacji)
- **Wynik:** Wyraźnie rozdzielone score'y contaminated vs. clean
- **Link:** https://arxiv.org/abs/2510.27055

### 4. "A Watermark for Large Language Models" (Kirchenbauer et al., ICML 2023)
- **Temat:** Watermarking outputu LLM
- **Metoda:** "Green token" selection - przed generacją każdego tokena, losowy podzbiór tokenów oznaczany jako "zielone" i promowany podczas samplowania
- **Detekcja:** Statystyczny test z p-values, bez potrzeby dostępu do modelu
- **Link:** https://proceedings.mlr.press/v202/kirchenbauer23a.html

---

## PROFIL BADAWCZY ORGANIZATORÓW (wskazówki co do challenge'ów)

### Adam Dziedzic (CISPA, SprintML Lab)
- Differential privacy, membership inference, dataset inference
- Watermarking dla modeli text-to-image
- Memoryzacja w modelach dyfuzyjnych i autoregresyjnych
- Model stealing i obrona przed nim
- Prywatność LLM
- PhD: University of Chicago; doświadczenie: Microsoft Research, Google, CERN

### Franziska Boenisch (CISPA, SprintML Lab)
- Model inversion attacks (rekonstrukcja danych treningowych)
- Membership inference attacks
- Differential privacy
- Prywatność w foundation models (grant ERC: Privacy4FMs)
- Praktyczna ewaluacja prywatności

---

## KLUCZOWE NARZĘDZIA I FRAMEWORKI DO PRZYGOTOWANIA

### Adversarial Attacks
- **Foolbox** - biblioteka Python do ataków adversarialnych
- **ART (Adversarial Robustness Toolbox)** - IBM, kompleksowy framework
- **CleverHans** - Goodfellow et al.
- **PyTorch FGSM Tutorial** - https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
- **TensorFlow FGSM Tutorial** - https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

### Privacy Attacks
- **ML Privacy Meter** - membership inference i model inversion
- **OpenDP** - differential privacy
- **Opacus** (PyTorch) - trenowanie z DP-SGD

### Watermarking
- Implementacja z paperu Kirchenbauer et al. (green-list watermarking)
- **Tree-Ring Watermarks** - bardziej robustne watermarki w frequency domain

### Model Stealing
- **Knockoff Nets** - framework do model extraction
- **Steal-ML** (Tramèr et al.) - https://github.com/ftramer/Steal-ML

---

## STRATEGIA NA HACKATHON

### Podział ról w zespole 3-osobowym
1. **Osoba A:** Specjalista od ataków adversarialnych + model robustness
2. **Osoba B:** Specjalista od privacy attacks (membership/dataset inference, model inversion)
3. **Osoba C:** Specjalista od watermarking/attribution + model stealing

### Przygotowanie przed hackathnem
1. ✅ Przeczytać wszystkie 4 papery z maila
2. ✅ Przećwiczyć podstawowe ataki (FGSM, PGD) na prostych modelach
3. ✅ Zapoznać się z narzędziami (ART, Foolbox, ML Privacy Meter)
4. ✅ Przygotować środowisko z GPU (zarejestrować się na Jülich: https://judoor.fz-juelich.de/projects/training2615)
5. ✅ Dołączyć na Discord: https://discord.gg/F2hPce55kK
6. ✅ Przygotować boilerplate code do szybkiego startu
7. ✅ Przećwiczyć watermark detection/removal
8. ✅ Przećwiczyć model extraction przez API

### Podczas hackatonu
- Szybko przeczytać opisy zadań i przypisać osoby
- Korzystać z AI (Claude) do szybkiego prototypowania i debugowania
- Iterować szybko - submisje na serwer dają natychmiastowy feedback
- Nie spędzać zbyt dużo czasu na perfekcji - liczy się leaderboard score
- Przygotować się na prezentację przed jury - umieć wytłumaczyć podejście
