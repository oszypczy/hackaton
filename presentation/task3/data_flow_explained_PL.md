# Wyjaśnienie kroków diagramu `data_flow_task3.png`

Diagram jest **wysokopoziomowy** (angielski, bez nazw plików z repo). Poniżej — co oznacza każdy element **w kolejności przepływu**.

---

## 1. Input data

- **Labeled corpus (~540 texts)**  
  Teksty z **znaną etykietą**: „clean” vs „watermarked”. To zwykle połączenie zbiorów treningowego i walidacyjnego. Służy **wyłącznie** do uczenia dwóch głowic (parametrów skalowania i klasyfikatora liniowego).

- **Test set (2250 texts, no labels)**  
  Teksty bez etykiet — na nich produkujesz **predykcje** do wysłania organizatorowi. Identyczna kolejność przetwarzania co dla części zlabelowanej zapewnia spójność macierzy cech.

---

## 2. Shared black-box feature layer

**Idea:** jeden szeroki wektor liczb na tekst, zbudowany z **wielu niezaleznych „spojrzeń”** na ten sam tekst — bez zakładania jednego konkretnego algorytmu watermarku. Obie głowice (A i B) korzystają z **tego samego** zestawu tych cech (plus własny, wąski dodatek później).

- **Small surrogate LM statistics**  
  Pod **małym** modelem językowym liczone są zgrubne statystyki typu prawdopodobieństw tokenów, rang, zróżnicowania n-gramów, prostych miar „klimatu” tekstu — szybki, ogólny sygnał.

- **Observer–performer pairs (GPT-2, Pythia scales)**  
  Metoda typu *binoculars*: dwa modele (np. mniejszy „obserwator”, większy „wykonawca”) i relacja między ich perplexity / log-probami. Tekst wygenerowany maszynowo często ma **charakterystyczny profil** tej relacji; różne skale (GPT-2, Pythia) dają różną czułość.

- **OLMo-7B-Instruct sequence scores**  
  Zgrubne statystyki **dopasowania** tekstu do dużego modelu **OLMo-7B-Instruct** (np. średnie log-prawdopodobieństwa, perplexity po sekwencji). W praktyce zespołu ta rodzina modeli była szczególnie informacyjna na tych danych.

- **OLMo-1B-Instruct sequence scores**  
  To samo w mniejszej skali **OLMo** — sygnał **względem** dużego OLMo i innych rodzin (np. Pythia) pomaga w „odczycie” pochodzenia stylu generacji.

- **Token-level curvature under a causal LM**  
  Na każdym kroku: jak **nietypowy** jest faktyczny token względem rozkładu przewidywanego przez model (idea krzywizny / DetectGPT). Watermark może systematycznie przesuwać wybór w stronę „mniej zaskakujących” tokenów.

- **Semantic sentence embeddings**  
  Zdania → wektory; miary **spójności semantycznej** (np. podobieństwo sąsiednich zdań, proste statystyki w przestrzeni embeddingów). Kierunek na watermarky **semantyczne**, niewidoczne tylko na poziomie liczenia tokenów.

- **Optional: train-derived token frequency priors**  
  Z części **zlabelowanej** można wyciągnąć preferencje „które tokeny częściej przy watermark” i zamienić je w dodatkowe cechy. Logicznie to **wyciek informacji z treningu** — na diagramie jest „optional”, bo zależy od konfiguracji i od tego, czy chcesz to mieć w wektorze.

**Notatka na diagramie (*same text order; wide feature vector*):**  
Wszystkie ekstraktory liczą cechy w **tej samej kolejności tekstów**; wyniki są **sklejane w jeden długi wektor** na tekst.

---

## 3. Dwie gałęzie (Head A i Head B)

Po warstwie wspólnej ścieżka **rozgałęzia się**. Uczone są **dwa osobne** modele scoringu; różnią się **jednym typem** dodatkowych cech.

### Head A

- **Per-step full softmax under OLMo-7B-Instruct**  
  Dla każdej pozycji bierzesz **pełny rozkład** następnego tokenu (nie tylko średnie log-prob). Agregujesz m.in. **entropię**, **rangę** faktycznego tokenu, **serie** pozycji o niskiej entropii. Watermark często **zmienia kształt** rozkładu, nie tylko jego średnią.

### Head B

- **Multiple tokenizer variants of green-list z-tests (Kirchenbauer-style)**  
  Dla wielu konfiguracji (różne tokenizery, udział „zieleni”, kontekst seeda) liczysz statystyki w stylu **„czy tokeny zbyt często wpasowują się w hipotetyczną zieloną listę”** (z-score). Nawet gdy prawdziwy watermark konkursu **nie** jest bit-w-kopię Kirchenbauer, takie cechy mogą nadal coś nieść — klasyfikator wybiera wagi.

### Oba heady — dalej identycznie (na schemacie)

- **Cross-model discrepancy features**  
  Kilka liczb z **różnic / ilorazów** między już policzonymi sygnałami (np. OLMo-7B vs GPT-2 medium, vs Pythia, vs OLMo-1B). To nie jest nowe „przepuszczenie” tekstu — tylko **pochodne** z tego samego wektora, żeby podkreślić **względną** dziwność pod różnymi LM.

- **Scaled logistic regression**  
  Cechy są **standaryzowane** (średnia/wariancja z treningu w foldzie lub na całości), potem **logistyczna regresja** z regularyzacją. Wynik to prawdopodobieństwo klasy „watermarked” na **teście** — **Score vector A** lub **B**.

---

## 4. Rank fusion on test set

- **average ranks of A and B**  
  Na 2250 próbkach: z osobna zamieniasz score’y A i B na **rangi** (porządek od najmniej do najbardziej „podejrzanych” według danej głowicy — przy typowej konwencji wyższy surowy score → wyższa ranga). Potem **uśredniasz** te dwie rangi po próbce.

- **linear rescale**  
  Uśredniony wektor jest **liniowo rozciągnięty** na sensowny zakres i lekko **przycięty**, żeby uniknąć dokładnie 0 lub 1 — tak powstaje **jeden** końcowy ranking zgodny z metryką **TPR przy ustalonym FPR na clean** (decyduje głównie **porządek**, nie skala surowa).

---

## 5. Final CSV

Plik **id, score** dla 2250 wierszy — gotowy do API konkursu.

---

## Jak to się ma do nazwy pliku „final blend”?

**Finalny plik CSV** to po prostu zapis tego samego przepływu: rankowa fuzja **Head A** (pełny rozkład OLMo-7B-Instruct) i **Head B** (wielowariantowe z-testy zielonej listy), na wspólnym tle cech black-box.
