# Diagnostyka różnicowa chorób erytemato-łuskowych metodami uczenia maszynowego

**Autorzy:**
1. Mateusz Mróz, 251190
2. Dawid Kośka, XXXXXX
3. Wiktor Grzyb, XXXXXX

**Data:** DD.MM.2026
**Przedmiot:** Wprowadzenie do uczenia maszynowego (dr inż. Krzysztof Smółka)
**Temat:** 6. Choroby skóry (łuszczyca, egzema, trądzik, alergie kontaktowe)

---

## Streszczenie

> *(150–250 słów)*
>
> Raport przedstawia zastosowanie klasycznych metod uczenia maszynowego w zadaniu diagnostyki różnicowej niezłośliwych chorób skóry. Zespół zebrał trzy niezależne zbiory danych: (1) UCI Dermatology — tabelaryczny zbiór 366 przypadków sześciu chorób erytemato-łuskowych, (2) DermNet — 19 500 obrazów dermatologicznych z 23 kategoriami chorób skóry oraz (3) Fitzpatrick17k — 16 577 obrazów oznakowanych typem skóry Fitzpatrick'a, obejmujący 114 dermatoz. Każdy członek zespołu wybrał jeden zbiór do własnych eksperymentów modelowych.
>
> W części eksperymentalnej osoby 1 zastosowano pipeline Random Forest z strojeniem hiperparametrów metodą GridSearchCV i walidacją 5-fold StratifiedKFold. Wynik na zbiorze testowym: **accuracy 95.95%, macro-F1 94.31%** — poziom zgodny z literaturą (Güvenir 1998: 99.2%, Xie 2011: 98.6%). Analiza SHAP potwierdziła, że model korzysta z cech medycznie sensownych (histopatologia dominuje nad kliniką). Wyniki osób 2 i 3 zostaną uzupełnione po zakończeniu ich eksperymentów.
>
> Projekt pokazuje, że klasyczne tabelaryczne metody ML pozostają konkurencyjne w dermatologii dla dobrze scharakteryzowanych danych klinicznych, a ich interpretowalność jest kluczowa w kontekście medycznym.

**Słowa kluczowe:** choroby skóry; klasyfikacja wieloklasowa; Random Forest; SHAP; UCI Dermatology

---

## 1. Wstęp

Choroby skóry stanowią jedną z najczęstszych przyczyn wizyt u lekarza pierwszego kontaktu. Według WHO przewlekłe dermatozy (łuszczyca, atopowe zapalenie skóry, trądzik) dotykają ponad 900 milionów ludzi na świecie. Wczesna i precyzyjna diagnoza różnicowa jest trudna, ponieważ wiele chorób z grupy **erytemato-łuskowej** (łuszczyca, łojotokowe zapalenie skóry, liszaj płaski, łupież różowy Giberta, przewlekłe zapalenie skóry, łupież rumieniowaty mieszkowy) dzieli podobne objawy kliniczne — rumień i złuszczanie — i często wymaga biopsji dla potwierdzenia.

W ostatnich dekadach opracowano wiele klasycznych i głębokich modeli uczenia maszynowego wspierających dermatologów w diagnostyce. Celem niniejszego raportu jest:

1. Zebranie i opisanie trzech publicznych zbiorów danych dotyczących chorób skóry.
2. Przegląd literatury i benchmarków dla każdego zbioru.
3. Przeprowadzenie własnych eksperymentów ML na wybranych zbiorach.
4. Porównanie uzyskanych wyników z publikowanymi w literaturze i sformułowanie wniosków.

---

## 2. Materiały i metody

### 2.1. Zbiory danych

**Tabela 1.** Zestawienie wszystkich trzech zbiorów danych zebranych przez zespół.

| Lp. | Nazwa | Typ | Rozmiar | Objętość | Klasy / etykieta | Źródło (licencja) | Braki | Osoba |
|-----|-------|-----|---------|----------|------------------|-------------------|-------|-------|
| 1.1 | **UCI Dermatology** | tabularny | 366 × 34 cechy | ~110 KB (CSV) | 6 chorób erytemato-łuskowych (łuszczyca, łojotokowe ZS, liszaj płaski, łupież różowy, przewlekłe ZS, łupież rumieniowaty) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/33/dermatology) (CC BY 4.0) | 8 braków w `age` | Mateusz Mróz |
| 2.1 | **DermNet (Kaggle mirror)** | obrazy 2D | ~19 500 obrazów | ~1.6 GB (JPG) | 23 kategorie (łuszczyca, egzema, trądzik, alergie kontaktowe, infekcje grzybicze, bakteryjne...) | [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) (educational use) | — | Dawid Kośka |
| 3.1 | **Fitzpatrick17k** | obrazy + metadane | 16 577 obrazów | ~12 GB (JPG + meta) | 114 chorób skóry × 6 typów skóry Fitzpatrick'a | [GitHub mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) (CC BY-NC-SA) | — | Wiktor Grzyb |

#### 2.1.1. UCI Dermatology (dataset osoby 1, wybrany do modelowania)

Zbiór stworzony przez Nilsel Ilter (Gazi University, Ankara) i H. Altay Güvenir (Bilkent University) w 1998 roku. Cechy dzielą się na:

- **11 cech klinicznych** (wartości 0–3 lub binarne) — erytema, łuski, świąd, fenomen Koebnera, łupież grudkowy, zajęcie błon śluzowych, kolan/łokci, skóry głowy, wywiad rodzinny itp.
- **22 cechy histopatologiczne** (wartości 0–3) — nacieki, parakeratoza, spongioza, hiperkeratoza, fibroza warstwy brodawkowatej skóry właściwej itp.
- **wiek pacjenta** (kolumna `age`)

Klasy (patologiczne rozpoznania) są nierównolicznie reprezentowane:

| Klasa | Liczba próbek | Udział |
|-------|---------------|--------|
| psoriasis | 112 | 30.6% |
| lichen planus | 72 | 19.7% |
| seboreic dermatitis | 61 | 16.7% |
| chronic dermatitis | 52 | 14.2% |
| pityriasis rosea | 49 | 13.4% |
| pityriasis rubra pilaris | 20 | 5.5% |

Stosunek klasy większościowej do mniejszościowej wynosi 5.6× — *łagodne* niezbalansowanie, ale istotne przy wyborze metryki (macro-F1 zamiast accuracy).

#### 2.1.2. DermNet *(opisuje osoba 2 — Dawid Kośka)*

*Placeholder — opis do uzupełnienia przez osobę 2.*

#### 2.1.3. Fitzpatrick17k *(opisuje osoba 3 — Wiktor Grzyb)*

*Placeholder — opis do uzupełnienia przez osobę 3.*

### 2.2. Metody uczenia maszynowego

#### 2.2.1. Osoba 1 (Mateusz Mróz) — UCI Dermatology

**Pipeline:**
1. Podział stratyfikowany 80/20 (train/test, `random_state=42`)
2. Imputacja braków (`SimpleImputer(strategy="median")`)
3. Skalowanie (`StandardScaler`)
4. Klasyfikacja: baseline (`DummyClassifier`, `LogisticRegression`) + modele porównawcze (`RandomForestClassifier`, `SVC` z kernelem RBF, `XGBClassifier`)
5. Walidacja: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
6. Strojenie hiperparametrów: `GridSearchCV` z `scoring="f1_macro"` dla Random Forest
7. Interpretacja: feature importance (impurity-based) + SHAP (`TreeExplainer`)

**Użyte metryki:** accuracy, macro-F1, weighted-F1 — raportowane jako `mean ± std` z 5 foldów.

**Biblioteki:** scikit-learn 1.8.0, xgboost 3.2.0, shap 0.51.0.

**Reprodukcja:** `notebooks/mateusz_mroz_uci_dermatology.ipynb` (seed = 42).

#### 2.2.2. Osoba 2 (Dawid Kośka) — DermNet

*Placeholder — uzupełnia osoba 2.*

#### 2.2.3. Osoba 3 (Wiktor Grzyb) — Fitzpatrick17k

*Placeholder — uzupełnia osoba 3.*

---

## 3. Eksperymenty i wyniki

### 3.1. Benchmarki z literatury

**Tabela 2.** Wyniki publikowane dla zbioru UCI Dermatology (3 prace peer-review zweryfikowane co do DOI i metryki).

| Źródło | Rok | Model | Walidacja | Accuracy |
|--------|-----|-------|-----------|----------|
| Güvenir, Demiröz, Ilter [1] | 1998 | Voting Feature Intervals (VFI5) | 10-fold CV | 99.20% |
| Xie & Wang [2] | 2011 | SVM + hybrid feature selection (IFSFFS) | 10-fold CV | 98.61% |
| Abdi & Giveki [3] | 2013 | PSO-SVM + association rules (AR) | 10-fold CV | 98.91% |

**Legenda skrótów modelowych:**

- **VFI5** — Voting Feature Intervals (klasyfikator głosujący na przedziałach cech, Demiröz & Güvenir 1997).
- **IFSFFS** — Improved F-score + Sequential Forward Floating Selection (hybrydowa selekcja cech).
- **PSO-SVM + AR** — Particle Swarm Optimization tuning SVM + Association Rules do selekcji cech.

*Benchmarki dla DermNet i Fitzpatrick17k — uzupełniają osoby 2 i 3.*

### 3.2. Wyniki zespołu

**Tabela 3.** Zbiorcza tabela wyników eksperymentów zespołu.

| Lp.* | Dataset | Zadanie | Wyniki z literatury (najlepszy model) | Model zespołu (parametry) | Wynik zespołu |
|------|---------|---------|---------------------------------------|---------------------------|---------------|
| 1.1 | UCI Dermatology | klasyfikacja wieloklasowa (6 klas) | VFI5 — 99.20% acc (Güvenir 1998) | Random Forest (n=200, max_depth=None, min_samples_leaf=4, max_features=sqrt) | **acc CV: 97.59 ± 1.82%**<br>**f1_macro CV: 96.92 ± 2.30%** *(baseline)*<br>**f1_macro CV: 0.9810** *(GridSearch best_score_)*<br>**acc test: 95.95%**<br>**f1_macro test: 94.31%** |
| 2.1 | DermNet | *(placeholder — osoba 2)* | *(placeholder)* | *(placeholder)* | *(placeholder)* |
| 3.1 | Fitzpatrick17k | *(placeholder — osoba 3)* | *(placeholder)* | *(placeholder)* | *(placeholder)* |

*\* nr_osoby_wg_listy_autorów.nr_zbioru_danych*

### 3.3. Szczegóły wyników osoby 1 (UCI Dermatology)

**Porównanie modeli w walidacji 5-fold CV (metryki: `mean ± std`):**

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Dummy (most_frequent) | 0.3048 ± 0.0063 | 0.0779 ± 0.0012 | 0.1424 ± 0.0052 |
| Logistic Regression (balanced) | 0.9794 ± 0.0129 | 0.9768 ± 0.0145 | 0.9793 ± 0.0129 |
| Random Forest (300 drzew, domyślne) | 0.9759 ± 0.0182 | 0.9692 ± 0.0230 | 0.9757 ± 0.0182 |
| SVM RBF (balanced) | 0.9760 ± 0.0130 | 0.9736 ± 0.0145 | 0.9759 ± 0.0129 |
| XGBoost (300 drzew) | 0.9657 ± 0.0106 | 0.9604 ± 0.0125 | 0.9656 ± 0.0106 |

**Wybrany model:** Random Forest — porównywalna jakość do LR i SVM, a znacznie lepsza interpretowalność (feature importance + SHAP).

**Raport per-klasa (na zbiorze testowym n=74):**

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| psoriasis | 1.0000 | 1.0000 | 1.0000 | 23 |
| seboreic dermatitis | 0.9091 | 0.8333 | 0.8696 | 12 |
| lichen planus | 1.0000 | 1.0000 | 1.0000 | 15 |
| pityriasis rosea | 0.9000 | 0.9000 | 0.9000 | 10 |
| chronic dermatitis | 1.0000 | 1.0000 | 1.0000 | 10 |
| pityriasis rubra pilaris | 0.8000 | 1.0000 | 0.8889 | 4 |
| **macro avg** | **0.9348** | **0.9556** | **0.9431** | **74** |

**Najważniejsze cechy (top 10, z Random Forest feature importance):**

1. `fibrosis_papillary_dermis` (0.1064) — fibroza warstwy brodawkowatej
2. `perifollicular_parakeratosis` (0.0707) — parakeratoza okołomieszkowa
3. `thinning_suprapapillary_epidermis` (0.0674) — ścieńczenie naskórka nadbrodawkowego
4. `koebner_phenomenon` (0.0601) — fenomen Koebnera
5. `clubbing_rete_ridges` (0.0543) — maczugowate poszerzenie pasm naskórka
6. `vacuolisation_damage_basal_layer` (0.0488)
7. `follicular_horn_plug` (0.0477)
8. `spongiosis` (0.0460)
9. `elongation_rete_ridges` (0.0409)
10. `age` (0.0392)

8 z 10 top-cech to cechy **histopatologiczne**, co pokrywa się z praktyką kliniczną — biopsja jest standardem diagnostyki różnicowej tych chorób.

---

## 4. Dyskusja

### 4.1. Porównanie z literaturą

Nasze wyniki (accuracy **97.59 ± 1.82% CV / 95.95% test**) są porównywalne z publikacjami (98.61%–99.20%) — różnica ~1–3 pp mieści się w przedziale standardowego odchylenia naszego CV (±1.82 pp), więc nie jest statystycznie istotna. Cytowane publikacje raportują pojedynczą metrykę bez przedziałów ufności, co dodatkowo utrudnia rygorystyczne porównanie.

Świadome różnice metodologiczne względem top paperów:

1. **5-fold CV zamiast 10-fold** — mniejsza liczba foldów daje wyższą wariancję per fold, ale również bardziej konserwatywną estymację na małej próbce (n=366); 10-fold zwykle podnosi raportowany wynik o 0.5–1 pp.
2. **Ograniczony grid hiperparametrów** (54 kombinacje) — świadome unikanie over-tuningu na n=293 próbek treningowych.
3. **Brak zaawansowanej selekcji cech** — top papery używają IFSFFS (Xie 2011) lub PSO + association rules (Abdi 2013), wybierając 10–24 z 34 cech; my pracujemy na pełnym wektorze, co preferuje interpretowalność kosztem ~1–2 pp accuracy.
4. **Random state i podział train/test** — zbiór testowy (74 próbki, 20%) jest dedykowanym hold-outem; niektóre publikacje używają CV na pełnym zbiorze bez wydzielonego testu, co podnosi raportowany wynik.

### 4.2. Analiza błędów

Test accuracy 95.95% = 71 trafnych / 3 błędy na 74 próbkach (4.05% error rate). Pomyłki koncentrują się w parach klas o nakładających się objawach klinicznych (łuski + rumień) — `seboreic_dermatitis` ↔ `pityriasis_rosea` — oraz dla klasy najmniej licznej (`pityriasis_rubra_pilaris`, n=20). Pattern jest zgodny z literaturą — bez biopsji histopatologicznej rozróżnienie tych chorób jest klinicznie trudne (Güvenir et al. 1998), więc trudność modelu odzwierciedla realną trudność medyczną.

### 4.3. Interpretowalność modelu

Cechy o najwyższym wkładzie SHAP per klasa pokrywają się z klasycznymi markerami histopatologicznymi opisanymi w dermatopatologii:

- `saw_tooth_retes`, `band_like_infiltrate` → **lichen planus** (charakterystyczne pasmo limfocytarne na granicy skórno-naskórkowej)
- `perifollicular_parakeratosis`, `follicular_horn_plug` → **pityriasis rubra pilaris** (czopy rogowe wokół mieszków włosowych)
- `fibrosis_papillary_dermis`, `clubbing_rete_ridges` → **chronic dermatitis** (przewlekłe włóknienie brodawkowate)
- `koebner_phenomenon`, `knee_and_elbow_involvement` → **psoriasis** (objaw Köbnera + lokalizacja kolana/łokcie)

Zbieżność cech SHAP z klasycznymi markerami histopatologicznymi sugeruje, że model opiera decyzje na cechach dziedzinowo poprawnych, a nie na korelacjach przypadkowych w danych treningowych.

### 4.4. Ograniczenia

- Mała próbka (366) → duża wariancja wyników między foldami.
- Niezbalansowanie klasy 6 (20 próbek) — nawet z `class_weight="balanced"` wyniki dla tej klasy mają większą niepewność.
- Brak cech obrazowych i multimodalnych (wywiad, zdjęcie zmiany, dane laboratoryjne).
- Dane zebrane w jednym ośrodku (Gazi University, Ankara) → możliwy bias populacyjny.

---

## 5. Wnioski

1. **Klasyczne metody ML (Random Forest, Logistic Regression, SVM) pozostają konkurencyjne** na tabelarnych danych dermatologicznych, osiągając >96% accuracy.
2. **Niewielka różnica między modelami** (0.9692 vs 0.9810 macro-F1) sugeruje, że dla tego zbioru czynnikiem limitującym jest jakość i ilość danych, nie wybór algorytmu.
3. **Cechy histopatologiczne dominują** — potwierdzenie roli biopsji w diagnostyce różnicowej.
4. **Model jest interpretowalny** — SHAP + feature importance pokazują medycznie sensowne wzorce.
5. **Propozycje ulepszeń:**
   - Stacking (LR + RF + XGBoost + meta-learner).
   - SMOTE dla klasy 6.
   - Rozszerzenie o cechy obrazowe (fuzja tabular + CNN embeddingi z DermNet).
   - Walidacja na niezależnym zbiorze (np. zbiór Xie 2011 o 500+ próbkach).

---

## 6. Bibliografia

[1] H. A. Güvenir, G. Demiröz, and N. Ilter, "Learning differential diagnosis of erythemato-squamous diseases using voting feature intervals," *Artificial Intelligence in Medicine*, vol. 13, no. 3, pp. 147–165, 1998. DOI: [10.1016/S0933-3657(98)00028-1](https://doi.org/10.1016/S0933-3657(98)00028-1).

[2] J. Xie and C. Wang, "Using support vector machines with a novel hybrid feature selection method for the diagnosis of erythemato-squamous diseases," *Expert Systems with Applications*, vol. 38, no. 5, pp. 5165–5172, 2011. DOI: [10.1016/j.eswa.2010.10.050](https://doi.org/10.1016/j.eswa.2010.10.050).

[3] M. J. Abdi and D. Giveki, "Automatic detection of erythemato-squamous diseases using PSO–SVM based on association rules," *Engineering Applications of Artificial Intelligence*, vol. 26, no. 1, pp. 603–608, 2013. DOI: [10.1016/j.engappai.2012.01.017](https://doi.org/10.1016/j.engappai.2012.01.017).

*Bibliografia dla DermNet i Fitzpatrick17k — uzupełniają osoby 2 i 3.*

---

## 7. Załączniki

Struktura oddania (ZIP na WIKAMP):

```
PROJEKT/
├── raport.pdf                                      # eksport niniejszego pliku
├── data/
│   ├── uci_dermatology/
│   │   ├── dermatology.data
│   │   └── dermatology.names
│   ├── dermnet/                                    # (osoba 2)
│   └── fitzpatrick17k/                             # (osoba 3)
├── notebooks/
│   ├── mateusz_mroz_uci_dermatology.ipynb          # pełny pipeline
│   ├── dawid_koska_dermnet.ipynb                   # (osoba 2)
│   └── wiktor_grzyb_fitzpatrick17k.ipynb           # (osoba 3)
├── results/
│   ├── mateusz_mroz_uci_dermatology_cv_results.csv
│   └── mateusz_mroz_uci_dermatology_final.csv
└── env/
    └── requirements.txt
```

---

## 8. Aneksy

### A. Rozkład klas (UCI Dermatology)

Generowany w notebooku — sekcja 3.2 `mateusz_mroz_uci_dermatology.ipynb`.

### B. Macierz korelacji cech

Sekcja 3.4 notebooka.

### C. Macierz pomyłek na zbiorze testowym

Sekcja 8 notebooka.

### D. Wykresy SHAP summary

Sekcja 9.2 notebooka.

### E. Boxploty top-9 cech per klasa i rozkład wieku (KDE)

Sekcje 3.5 i 3.6 notebooka. Boxploty potwierdzają, że cechy silnie dyskryminują poszczególne klasy (np. `band_like_infiltrate` → lichen_planus, `clubbing_rete_ridges` → psoriasis). KDE wieku pokazuje, że `pityriasis_rubra_pilaris` występuje głównie u dzieci (~10 lat) — zgodne z literaturą medyczną dla tej rzadkiej dermatozy pediatrycznej.

### F. Krzywe ROC One-vs-Rest

Sekcja 8.1 notebooka. Macro AUC = 0.998, wszystkie klasy powyżej 0.99.

### G. Konfiguracja Random Forest (po GridSearchCV)

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
```
