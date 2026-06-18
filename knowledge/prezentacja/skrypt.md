# Skrypt prezentacji - ocena eksperymentu A/B/C (przycisk w e-commerce)

> Scenariusz mówiony do prezentacji `prezentacja.html`.
> Układ: przechodzimy przez cały eksperyment PUNKT PO PUNKCIE (1-6). Każdy punkt ma dwa slajdy: najpierw ORYGINAŁ (cytat z projektu), potem OCENA (zielony/czerwony/żółty).
> Nawigacja: strzałki ← / →, scroll, ESC = przegląd slajdów.
> Kolory: zielony = dobrze · czerwony = błąd · żółty = poprawka.

---

## Slajd 1 - Okładka

Dzień dobry. Oceniam projekt eksperymentu A/B/C ze sklepu internetowego - test wyglądu przycisku „Dodaj do koszyka”. Żeby nic nie pominąć, przejdę przez cały projekt dokładnie w jego oryginalnej kolejności: jest sześć punktów. Dla każdego punktu najpierw pokażę, co napisali autorzy - cytat z oryginału - a zaraz potem moją ocenę tego punktu. Kolory są kluczem: zielony to elementy dobre, czerwony to błędy, żółty to propozycje poprawek. Z prawej strony widać plan: scenariusz, grupy, hipoteza, zmienna zależna, randomizacja i analiza.

---

## Slajd 2 - Punkt 1: scenariusz (ORYGINAŁ)

Punkt pierwszy - scenariusz i cel. Autorzy zauważyli, cytuję, że „wielu użytkowników odwiedza stronę z opisem produktu, ale stosunkowo niewielu z nich decyduje się na dodanie go do koszyka”. I dalej: „chcemy przetestować, czy zmiana jego wyglądu zwiększy sprzedaż”. Czyli problem jest jasny - mała konwersja na stronie produktu - a pomysłem na rozwiązanie jest zmiana przycisku.

## Slajd 3 - Punkt 1: scenariusz (OCENA)

Moja ocena tego punktu. Na plus: problem jest konkretny i mierzalny, to świetny scenariusz pod test A/B/C. Ale jest pierwszy błąd: celem jest „sprzedaż”, a całe badanie mierzy dodanie do koszyka. To nie to samo - część osób doda produkt do koszyka i nigdy nie zapłaci. Poprawka: albo mierzyć faktyczny zakup, albo jasno zaznaczyć, że badamy tylko etap koszyka, i ostrożniej formułować wnioski.

---

## Slajd 4 - Punkt 2: grupy A/B/C (ORYGINAŁ)

Punkt drugi - grupy. Mamy grupę kontrolną A: szary przycisk „Dodaj do koszyka”. Grupę B: duży, zielony przycisk, ten sam tekst. I grupę C: duży, pomarańczowy przycisk z innym tekstem - „Kup teraz”. Czyli jedna kontrola i dwie wersje eksperymentalne.

## Slajd 5 - Punkt 2: grupy A/B/C (OCENA)

I tu jest najpoważniejsza wada całego projektu. Spójrzmy na macierz zmian. W grupie B zmienia się naraz kolor i rozmiar. W grupie C - kolor, rozmiar oraz tekst. To znaczy, że warianty różnią się kilkoma rzeczami jednocześnie. Jeśli C wygra, nie będziemy wiedzieć, co właściwie zadziałało: kolor, rozmiar, słowa „Kup teraz”, czy ich kombinacja. To jest konfundacja zmiennych. Brakuje też czwartej kombinacji - na przykład szary przycisk z napisem „Kup teraz” - która pozwoliłaby oddzielić efekt tekstu od efektu koloru.

---

## Slajd 6 - Punkt 3: hipoteza (ORYGINAŁ)

Punkt trzeci - hipoteza badawcza. Autorzy spodziewają się, cytuję, że „wariant C osiągnie najwyższe wyniki ze względu na budowanie poczucia pilności zakupu”. Czyli zakładają, że wyrazisty kolor i bezpośredni komunikat zwiększą konwersję, a najlepszy będzie wariant C.

## Slajd 7 - Punkt 3: hipoteza (OCENA)

Ocena. Spójrzmy na pełną hipotezę z poprzedniego slajdu. Na plus: jest jasno wyrażony kierunek oczekiwań - wiadomo, czego autorzy się spodziewają, że wygra C. Ale są trzy problemy. Po pierwsze: nie ma formalnej pary hipotez - brakuje hipotezy zerowej i alternatywnej, a to podstawa wnioskowania statystycznego. Hipoteza zerowa to założenie, że żaden wariant nie różni się od pozostałych; to właśnie ją test ma próbować obalić. Po drugie: sama hipoteza miesza dwa mechanizmy naraz - w jednym zdaniu mówi o „wyrazistym kolorze ORAZ bezpośrednim komunikacie Kup teraz”. Jeśli C wygra, z samej hipotezy nie wynika, co zadziałało: kolor czy tekst. Po trzecie: brak progu - autorzy piszą tylko „wyższy współczynnik konwersji”, ale nie mówią, o ile; nie wiadomo, jaki wzrost uznają za sukces. (Osobny problem, że hipoteza jest kierunkowa, a chi-kwadrat bezkierunkowy, wróci jeszcze przy punkcie 6.)

---

## Slajd 8 - Punkt 4: zmienna zależna (ORYGINAŁ)

Punkt czwarty - zmienna zależna, czyli to, co mierzymy. Autorzy definiują współczynnik konwersji jako, cytuję, „stosunek liczby użytkowników, którzy kliknęli przycisk i dodali produkt do koszyka, do całkowitej liczby unikalnych użytkowników, którzy odwiedzili daną stronę produktu”.

## Slajd 9 - Punkt 4: zmienna zależna (OCENA)

Ocena. Na plus: konwersja to trafny typ metryki - jest mierzalna i policzalna. Ale w tej definicji są dwa problemy. Po pierwsze, łączy ona dwa różne zdarzenia w jedno - „kliknęli i dodali do koszyka” - a dobra metryka powinna opisywać jedno czyste zdarzenie. Po drugie, to wciąż nie jest zakup. Poprawka: wybrać jedno jednoznaczne zdarzenie i określić okno konwersji - czy liczymy je w tej samej sesji, w ciągu doby, czy w dłuższym oknie.

---

## Slajd 10 - Punkt 5: randomizacja (ORYGINAŁ)

Punkt piąty - randomizacja, czyli sposób przydziału do grup. Autorzy piszą, cytuję, że „algorytm wykonuje na User ID lub zahashowanym identyfikatorze sesji operację modulo 3”, gdzie reszta zero to grupa A, jeden to B, dwa to C. I dodają, że jeśli użytkownik wróci następnego dnia, zobaczy ten sam wariant.

## Slajd 11 - Punkt 5: randomizacja (OCENA)

To najważniejszy techniczny zarzut. Na plus: stały przydział użytkownika jest dobry - po powrocie ta sama osoba widzi ten sam wariant, nie miesza wersji. Ale jest poważny problem z samym „modulo 3”. Ono daje równe i losowe grupy tylko wtedy, gdy identyfikatory są równomierne i nieuporządkowane. Jeśli ID są sekwencyjne albo ten sam hash jest używany w kilku eksperymentach, podział przestaje być losowy i grupy dostają systematyczne obciążenie. Poprawka: najpierw zahashować identyfikator z solą specyficzną dla tego eksperymentu, dopiero potem brać modulo, kontrolować balans grup - czyli Sample Ratio Mismatch - i wykluczyć boty oraz ruch testowy.

---

## Slajd 12 - Punkt 6: analiza (ORYGINAŁ)

Punkt szósty - analiza wyników. Test ma trwać trzy tygodnie. Autorzy planują, cytuję, „najpierw test Chi-kwadrat dla wielu proporcji” przy poziomie istotności 0,05, a potem porównania parami - A z B, A z C oraz B z C - z poprawką Bonferroniego.

## Slajd 13 - Punkt 6: analiza (OCENA)

Ocena. Kierunkowo to dobry plan. Na plus: chi-kwadrat dla wielu proporcji to właściwy test, a poprawka Bonferroniego przy porównaniach parami jest jak najbardziej na miejscu. Ale są problemy. Po pierwsze - i tu wraca wątek z hipotezy - chi-kwadrat jest testem omnibus, bezkierunkowym: mówi tylko „czy w ogóle jest jakaś różnica między grupami”, a nie „która grupa jest lepsza”. A autorzy postawili hipotezę kierunkową, że wygra C. Sam chi-kwadrat tego nie potwierdzi - dopiero porównania parami. Po drugie - brak analizy mocy: nie policzono wymaganej liczebności próby, czyli alfy, bety i minimalnego wykrywalnego efektu. Po trzecie - reguła zakończenia: sztywne „trzy tygodnie” bez zasady, kiedy wolno przerwać, otwiera drogę do peekingu. I pamiętajmy: sam czas nie gwarantuje mocy - przy małym ruchu trzy tygodnie i tak nie wystarczą.

---

## Slajd 14 - Werdykt

Podsumowując, po przejściu wszystkich sześciu punktów: moja ocena to mniej więcej sześć na dziesięć. Projekt ma sensowny temat, grupę kontrolną, randomizację na poziomie użytkownika i dobry kierunek analizy - to realne plusy. Największy minus to trzy rzeczy: mieszanie kilku zmian naraz w wariantach, „modulo 3” użyte zamiast prawdziwej randomizacji oraz mylenie sprzedaży z dodaniem do koszyka. W obecnej formie eksperyment pokaże najwyżej, który wariant wygrał, ale nie wyjaśni, dlaczego. Po poprawkach - jedna zmiana na raz, metryka zakupowa, analiza mocy i poprawny przydział - może być wartościowym badaniem. Dziękuję za uwagę.

---

### Ściąga awaryjna (kolejność punktów)

1. **Scenariusz** - dobry problem ✓ · ale „sprzedaż” vs koszyk ✗
2. **Grupy A/B/C** - są 3 grupy ✓ · ale kilka zmian naraz ✗ (główna wada)
3. **Hipoteza** - jest kierunek ✓ · brak H₀/H₁, brak progu sukcesu ✗
4. **Zmienna zależna** - konwersja OK ✓ · łączy 2 zdarzenia, brak okna ✗
5. **Randomizacja** - stały przydział ✓ · „mod 3” ≠ losowanie, brak soli/SRM ✗
6. **Analiza** - chi-kwadrat + Bonferroni ✓ · test omnibus vs hipoteza kierunkowa, brak mocy/reguły STOP ✗

→ **Werdykt: 6/10** - dobry szkic, niegotowy do realizacji.
