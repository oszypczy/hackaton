# Task 3 — logika najlepszego rozwiązania (bez szczegółów implementacji)

**Co wysyłamy:** jeden plik ze skoringiem każdej z 2250 próbek testowych — im wyższy wynik, tym silniejsze przekonanie modelu, że tekst jest oznakowany watermarkem.

**Jak ocenia organizator:** TPR przy 1% fałszywych alarmów na tekstach „czystych”. Praktycznie chodzi więc o **dobry ranking** próbek: musimy ustawić kolejność tak, żeby przy takim progu jak najwięcej prawdziwie watermarked trafiało powyżej progu, a fałszywe alarmy na clean były rzadkie. Absolutna wartość liczby (np. czy 0.6 „znaczy” watermark) jest drugorzędna względem **porządku** względem innych próbek.

---

## Od danych do wyniku — krok po kroku

Poniżej **jedna spójna ścieżka**: od plików organizatora po finalne 2250 liczb w pliku submisji. Dwa środkowe przebiegi (gałęzie) różnią się tylko **rodzajem dodatkowego bloku cech**; reszta logiki jest taka sama.

1. **Wejście.** Dostajesz zbiory tekstów: część **oznaczona** (typowo 360 trening + 180 walidacji, etykieta: clean vs watermarked) oraz **test** (2250 tekstów bez etykiety, każdy ma identyfikator `id`). To jedyne informacje nadzorowane używane do uczenia głowic.

2. **Jedna wspólna kolejność tekstów.** Wszystkie ekstraktory liczą cechy w tej samej kolejności: najpierw **wszystkie zlabelowane** (w ustalonym spięciu train+val), potem **wszystkie testowe**. Dzięki temu łatwo uciąć macierz: początek wierszy = uczenie, koniec = tylko predykcja.

3. **Warstwa „ogólnych” cech (wcześniejsza praca).** Dla każdego tekstu z tej listy obliczasz (lub wczytujesz z cache) **ten sam zestaw** sygnałów black-box: surrogate LM, Binoculars w kilku wariantach, OLMo-7B i OLMo-1B (statystyki log-prob / PPL), krzywizna tokenowa, proxy semantyczne, opcjonalnie heurystyki list wyuczone na treningu itd. Wynik: **wiele liczb na tekst**, wspólnych dla obu głowic — by obie „widziały ten sam świat”, zestaw musi być **identyczny** między dwoma przebiegami (ten sam cache / ta sama wersja ekstrakcji).

4. **Różnica między przebiegami: jeden dodatkowy blok.**  
   - *Przebieg A (gałąź entropii):* do powyższego dokładasz cechy z **pełnego rozkładu następnego tokenu** pod OLMo-7B-Instruct (entropie, rangi, bursty niskiej entropii itd.).  
   - *Przebieg B (gałąź list KGW-style):* zamiast tego dokładasz **wiele wariantów z-score** „czy tokeny zbyt często trafiają na hipotetyczną zieloną listę” pod różnymi tokenizacjami i ustawieniami.  
   W jednym uruchomieniu trenujesz **albo** A **albo** B — nie oba naraz w jednym modelu.

5. **Cechy pochodne cross-LM.** Z już obliczonych kolumn baseline (nie z nowych modeli) budujesz **kilka różnic i jeden iloraz** PPL między OLMo-7B a GPT-2 medium / Pythiami / OLMo-1B. Doklejasz je do wektora jako zwykłe dodatkowe kolumny.

6. **Jeden wektor liczb na tekst.** Dla każdego wiersza masz teraz długi wektor: **[ ogólne cechy | cross-LM v1 | blok A lub B ]**. Wartości absurdalne (nieskończoność, brak danych) są zamieniane na bezpieczne zera, żeby klasyfikator nie wybuchał.

7. **Uczenie głowicy (powtórzone osobno dla A i B).**  
   - Bierzesz **tylko wiersze zlabelowane** (te ~540).  
   - **Normalizujesz** cechy (średnia/var w obrębie zbioru treningowego w foldzie).  
   - Uczysz **liniowy** klasyfikator z **mocną regularyzacją** — ma ograżyć przeuczenie przy wielu skorelowanych kolumnach.  
   - Opcjonalnie: **walidacja krzyżowa** na tych 540 wierszach daje szacunek „TPR przy 1% FPR na clean” tylko do porównań wewnętrznych (nie musi zgadzać się z leaderboardem).  
   - **Model końcowy:** ponownie normalizacja + ta sama rodzina klasyfikatora, ale fit na **całych** zlabelowanych danych naraz.

8. **Predykcja na teście.** Bierzesz **wiersze testowe** (2250), przepuszczasz przez ten sam skaler i nauczony klasyfikator, otrzymujesz dla każdego `id` **prawdopodobieństwo** klasy „watermarked”. Lekko **obcinasz** do otwartego przedziału (żeby unikać dokładnie 0 lub 1). Masz **pierwszy wektor score’ów** (wynik gałęzi A) lub **drugi** (wynik gałęzi B).

9. **Fuzja rankingowa.** Dla każdej z 2250 próbek masz teraz **dwie liczby** (z A i z B). Osobno dla całego testu: zamieniasz score’y gałęzi A na **rangi** (najwyższy score → najwyższa ranga w sensie „bardziej podejrzany”), to samo dla B. **Uśredniasz** obie rangi (np. wagi równe). Z uśrednionego wektora robisz jeszcze **rozciągnięcie liniowe** na rozsądny przedział i znowu delikatny klip — to jest **ostateczny score** submisji.

10. **Wyjście.** Zapisujesz tabelę **id + score** dla 2250 wierszy. To jest plik wysyłany do organizatora; **BEST3** to ta sama logika co powyżej, ewentualnie pod inna nazwą pliku dla wersji „zaakceptowanej” przez zespół.

---

## Rdzeń pomysłu: dwa „głowy” + połączenie rankingowe

Zamiast jednego dużego detektora budujemy **dwa oddzielne modele** prawdopodobieństwa „watermarked”. Oba uczą się na tych samych **około pięciuset** oznakowanych tekstach (trening + walidacja) i korzystają z **tego samego szerokiego zestawu ogólnych cech** z wcześniejszej pracy zespołu — opisanego poniżej. Do tego dochodzą **kilka ręcznie zdefiniowanych różnic między modelami** (tzw. cross-LM v1), liczonych z tych samych liczb źródłowych.

### Zestaw ogólnych cech — co to znaczy logicznie

**Uściślenie:** „Ogólne” to **wiele niezależnych ekstraktorów** na ten sam tekst (każdy może zwrócić **wiele** liczb), a wyniki są **łączone w jeden szeroki wektor** — to nie jest dokładnie „6 sposobów”. Liczba **6** odnosi się wyłącznie do **pochodnych cross-LM v1** (wzory na różnice / iloraz z już obliczonych kolumn), a nie do liczby osobnych metod ekstrakcji.

Chodzi o **„black-box” sygnały**, które nie zakładają jednego konkretnego algorytmu watermarku, tylko łączą obserwacje „ten tekst jest statystycznie dziwny pod różnymi LM / metrykami”.

- **Lekki surrogate LM (styl „branch A”):** pod małym modelem (np. GPT-2) liczymy typowe wskaźniki generacji: jak wyglądają log-prawdopodobieństwa i **kubełki rang** (poziom „GLTR”), **różnorodność n-gramów**, **burstiness**, kompresja, typowy stosunek typów do tokenów. To szybki, schemat-agnostyczny „klimat” tekstu.

- **Binoculars w kilku rozmiarach:** metoda oparta na **stosunku / niespójności perplexity** (lub powiązanych log-prob) między **dwoma** modelami — „obserwatorem” i „wykonawcą”. Tekst z maszyny (w tym często watermarked) ma charakterystyczny **profil** tej relacji; mocniejsze pary modeli (np. większe Pythie) dają ostrzejszy sygnał niż sama para małych GPT-2.

- **OLMo-7B (średnie log-proby / PPL):** duży model z **rodziny OLMo instruct**, na którym zespół wcześniej uzyskał najsilniejszy trend na leaderboardzie. Z tekstu wyciągane są zagregowane statystyki **jak bardzo i jak konsekwentnie** ten model „przewiduje” kolejne tokeny (średnia i rozrzut log-prob, percentyle, perplexity).

- **OLMo-1B (drugi rozmiar tej samej rodziny):** podobna idea co wyżej, ale **mniejszy** model — razem z sygnałami Pythia tworzy **odcisk „rodziny Allen AI vs Eleuther”** na danym tekście. Watermark lub generator danych może faworyzować jedną stronę.

- **Krzywizna / DetectGPT-style:** na pozycjach tokenów patrzymy, jak **atypowy** jest faktycznie wybrany token względem rozkładu, który model przewiduje analitycznie z softmaxa (średnia „dziwności”, rozrzut, ekstrema). Watermark może **systematycznie pchać** wybór w stronę bardziej „przewidywalnych” odpowiedzi względem tego rozkładu.

- **Proxy semantyczne (styl „branch D”):** tekst dzielony na zdania, embeddingi zdaniowe; liczymy **podobieństwo sąsiednich zdań** oraz prosty obraz rozkładu embeddingów (np. przez koszykowanie / porównanie histogramów). To kierunek na watermarky **semantyczne** (invariant w przestrzeni znaczeń), których nie widać tylko po częstości tokenów.

- **Heurystyki „zielonej listy” dopasowane do treningu (branche B/C w jednym bloku):** z oznakowanych tekstów treningowych wyciągane są **preferencje tokenów** typu „które ID pojawiają się częściej przy watermark vs clean”, potem używane do score’ów w stylu listy zielonej / okien. Logicznie to **atak na scenariusz unigramowego watermarku** i pokrewne; ponieważ część informacji pochodzi bezpośrednio z etykiet treningu, w dokumentacji zespołu bywała traktowana jako ryzykowna do **przeceny** na lokalnej walidacji względem leaderboardu — mimo to może zostać w wektorze cech, jeśli cache był zbudowany w tej konfiguracji.

**Sześć cech „cross-LM v1”** to nie osobne modele, tylko **pochodne** z już obecnych kolumn: różnice średnich log-prob OLMo-7B względem GPT-2 medium i Pythii (różne rozmiary), różnica między dwiema Pythiami, oraz iloraz perplexity OLMo-7B do Pythii. Intuicja: często ważniejsze od absolutnego PPL jest to, **jak bardzo** tekst „źle” wygląda pod jednym LM w porównaniu z drugim — watermark i sposób generacji mogą to przesuwać **spójnie** na całej próbie.

**Jedyna różnica między naszymi dwoma głowicami końcowymi** wobec tego wspólnego tła to dodatkowy, wąski blok cech:

1. **Pierwsza gałąź** patrzy na to, jak pod dużym modelem językowym (OLMo-7B-Instruct) wygląda **pełny rozkład następnego tokenu** na każdej pozycji — nie tylko „jak dziwny jest średnio log-prob”, ale **jak rozrzucona jest masa prawdopodobieństwa** (entropia), **jak nisko w rankingu** ląduje faktycznie wybrany token, jak często pojawiają się **fragmenty** z niską entropią. Intuicja: wiele watermarków **zawęża** losowanie tokenów, co zmienia kształt rozkładu, niekoniecznie w ten sam sposób co czysty tekst z modelu.

2. **Druga gałąź** dodaje wiele wariantów klasycznego **testu „zielona lista”** (styl Kirchenbauera): przy różnych tokenizacjach i ustawieniach liczymy, czy kolejne tokeny zbyt często wpasowują się w hipotetyczną listę dozwolonych tokenów zależną od kontekstu, i zamieniamy to na **statystyki typu z-score**. Intuicja: nawet jeśli organizatorzy **nie** użyli dokładnie tej samej funkcji skrótu i parametrów, takie liczby mogą dalej być **słabym lub silnym sygnałem** skorelowanym z regularnością generacji — a liniowy klasyfikator na całym wektorze cech sam wybiera, ile im ufa wobec reszty.

Każda gałąź na końcu przypisuje testowemu tekstowi **jedną liczbę** (prawdopodobieństwo klasy „watermarked”); **łączenie** obu liczb na 2250 próbkach jest opisane **w kroku 9** sekcji „Od danych do wyniku” (średnia **rang**, nie surowych score’ów — spójnie z metryką TPR przy ustalonym FPR na clean).

---

## Dlaczego to ma sens jako „najlepsze” podejście w tej rodzinie

- **Dwa niezależne spojrzenia:** jedna gałąź opiera się na **zachowaniu dużego LM na tekście**, druga na **abstrakcyjnych statystykach list tokenowych** bez ponownego przepuszczania tekstu przez ten sam duży forward w taki sam sposób. Błędy i mocne strony jednej gałęzi częściowo **komplementują** drugą.

- **Wspólny grunt:** obie gałęzie korzystają z **tego samego bogatego tła** cech zespołowych (w tym sprawdzonych różnic między modelami), więc nie zaczynamy od zera — tylko dokładamy **wąskie, wyspecjalizowane** rozszerzenia.

- **Fusion pod ranking:** świadomie łączymy wyniki sposobem zależnym od **porządku**, nie od surowej probabilistyki — spójne z tym, jak działa ewaluacja z ustalonym FPR.

---

## Nazwa pliku BEST3

Finalny plik to po prostu **wybrana** wersja tej samej logiki: **rankowe połączenie** wyniku gałęzi „entropie / rozkład OLMo” z wynikiem gałęzi „wielowariantowy test listowy”. Treść może być bitowo identyczna z wcześniejszym blendem o tej samej parze gałęzi — chodzi o **zamrożenie** decyzji zespołu, którą wersję uznać za ostateczną submisję.
