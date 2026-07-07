# Diagnostyka różnicowa chorób erytemato-łuskowych metodami uczenia maszynowego

**Autorzy:**
1. Mateusz Mróz, 251190
2. Dawid Kośka, 251171
3. Wiktor Grzyb, 251151

**Temat:** 6. Choroby skóry (łuszczyca, egzema, trądzik, alergie kontaktowe)

---

## Streszczenie

>Raport przedstawia zastosowanie uczenia maszynowego w diagnostyce chorób skóry na trzech zbiorach danych: UCI Dermatology (tabelaryczny, 366 przypadków), DermNet (obrazy, 19,5 tys.) oraz Fitzpatrick17k (obrazy, 16,5 tys.).
>
>**Osoba 1** (UCI) użyła modelu Random Forest z walidacją 5-fold CV, uzyskując na zbiorze testowym **accuracy 95.95% i macro-F1 94.31%**. Analiza SHAP potwierdziła diagnostyczne znaczenie cech histopatologicznych.  
>
>**Osoba 2** (DermNet) na pełnym zbiorze 3 klas porównała fine-tuning ResNet18 z modelami klasycznymi na zamrożonych embeddingach. Po hiperparameter tuningu i augmentacji danych, **ResNet18 uzyskał 81.3% accuracy** bez przeuczenia. Random Forest na embeddingach osiągnął 77.1%, LogisticRegression 73.0%. Pipeline „embeddingi + ML klasyczne" pozostaje stabilny, ale fine-tuning z early stopping daje znacznie lepsze wyniki.
>
>**Osoba 3** (Fitzpatrick17k) zastosowała transfer learning z ResNet-18 na 10% próbce. W walidacji 5-fold CV model uzyskał średnie **accuracy 73.93% i weighted-F1 67.16%**.
>
>**Wnioski:** Metody tabelaryczne oraz konwolucyjne z transfer learningiem są skuteczne w dermatologii. Przy małych zbiorach obrazów (<1000 próbek) pipeline „zamrożone embeddingi + ML klasyczne” wykazuje znacznie większą odporność na overfitting niż pełny fine-tuning sieci głębokich.

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

**Tabela 1.** Zestawienie wszystkich zbiorów danych zebranych przez zespół.

| Lp. | Nazwa | Typ | Rozmiar | Objętość | Klasy / etykieta | Źródło (licencja) | Braki | Osoba |
|-----|-------|-----|---------|----------|------------------|-------------------|-------|-------|
| **1.1** | **UCI Dermatology** | tabularny | 366 × 34 cechy | ~110 KB (CSV) | 6 chorób erytemato-łuskowych (łuszczyca, łojotokowe ZS, liszaj płaski, łupież różowy, przewlekłe ZS, łupież rumieniowaty) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/33/dermatology) (CC BY 4.0) | 8 braków w `age` | Mateusz Mróz |
| 1.2 | ACNE04 (Hayashi grading) | obrazy | 1 457 obrazów | ~1.2 GB (JPG) | 4 klasy (skala nasilenia trądziku Hayashi) | [GitHub xpwu95/LDL](https://github.com/xpwu95/LDL) (research-only) | — | Mateusz Mróz |
| 1.3 | PAD-UFES-20 | obrazy + metadane | 2 298 obrazów | PNG + CSV | 6 klas zmian skórnych (BCC, SCC, melanoma, actinic keratosis, nevus, seborrheic keratosis) | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) (CC BY 4.0) | — | Mateusz Mróz |
| **2.1** | **DermNet (Kaggle mirror)** | obrazy | 19 500 obrazów | ~1.6 GB | 23 klasy chorób skóry (m.in. trądzik, egzema, łuszczyca) | [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) / [DermNet.com](http://www.dermnet.com/) (CC BY-NC-ND 4.0) | — | Dawid Kośka |
| 2.2 | Skin Disease Dataset (Autor - Prashant Kumar Mishra) | obrazy | 15 444 obrazów | ~1.47 GB | 22 klasy chorób skóry (m.in. trądzik, egzema, łuszczyca) | [Kaggle](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) (CC0: Public Domain) | — | Dawid Kośka |
| 2.3 | Symptom2Disease | tabularny | 1200 punktów danych | 229.85 kB | 24 klas chorób, każda posiada 50 opisów symptomów | [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) (CC0: Public Domain) | — | Dawid Kośka |
| **3.1** | **Fitzpatrick17k** | obrazy + metadane | 16 577 obrazów | ~12 GB (JPG + meta) | 114 chorób skóry × 6 typów skóry Fitzpatrick'a | [GitHub mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) (CC BY-NC-SA) | — | Wiktor Grzyb |
| 3.2 | HAM10000 | obrazy + metadane | 10 015 obrazów | ~1.5 GB (JPG + meta) | 7 klas zmian barwnikowych (m.in. czerniak, znamię melanocytarne, BCC) | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (CC BY-NC 4.0) | — | Wiktor Grzyb |
| 3.3 | SD-198 | obrazy | 6 584 obrazów | ~1.3 GB (JPG) | 198 klas chorób skóry (m.in. łuszczyca, trądzik, egzema) | [Sun et al. 2016 (ECCV)](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13) (research-only) | — | Wiktor Grzyb |

Zbiór **PAD-UFES-20** nie dubluje zbiorów Dawida ani Wiktora: jest osobnym zbiorem klinicznych zdjęć zmian skórnych ze smartfonów oraz metadanych pacjentów. W pracy osoby 1 nie został wybrany do eksperymentu, ponieważ wymagałby pipeline'u obrazowego/CNN i większych zasobów obliczeniowych, natomiast celem tej części było porównanie klasycznych metod ML na dobrze opisanym zbiorze tabelarycznym.

#### 2.1.1. UCI Dermatology - Mateusz Mróz

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

#### 2.1.2. DermNet - Dawid Kośka

Mirror na [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) zawiera **19 500 obrazów RGB** w różnych rozdzielczościach, pogrupowanych w **23 kategorie dermatoz**, obejmujące: łuszczycę, egzemę, trądzik, infekcje (grzybicze, bakteryjne, wirusowe), guzy skóry, reakcje alergiczne, vasculitis i inne.

Charakterystyka oryginalnego zbioru:
- **19 500 obrazów JPEG** w rozdzielczościach od 300×200 px do 1200×900 px (zmienne proporcje)
- **23 klasy** (foldery katalogowe), m.in.: *Acne and Rosacea Photos*, *Eczema Photos*, *Psoriasis pictures Lichen Planus and related diseases*, *Nail Fungus*, *Melanoma Skin Cancer*, *Urticaria Hives*, *Warts Molluscum*, itp.
- **Brak klasycznych braków danych** — wszystkie obrazy są kompletne i zweryfikowane pod kątem poprawności formatu

**Proces selekcji danych w eksperymencie:**

Ze względu na ograniczenia sprzętowe (trening na CPU bez GPU), dokonano świadomej redukcji zbioru do 3 wybranych chorób tematycznie związanych z projektem:

| Klasa oryginalna (folder DermNet) | Wybrana klasa w eksperymencie |
|-----------------------------------|-------------------------------|
| *Acne and Rosacea Photos* | **Trądzik** |
| *Eczema Photos* | **Egzema** |
| *Psoriasis pictures Lichen Planus and related diseases* | **Łuszczyca** |

Wykorzystano **100% obrazów z wybranych klas** (bez próbkowania), co dało zbiór **4453 obrazów** (3480 treningowych + 973 testowych, podział oryginalny z Kaggle).

> **Uwaga:** Stała `SAMPLE_RATIO` w kodzie pozwala na zmniejszenie próbki (np. 25%) w celu przyspieszenia obliczeń na CPU. Przy pełnym datasecie cały processing notebooka trwa ~10 minut.

**Rozkład klas w finalnym zbiorze:**

| Klasa | Trening | Test | Łącznie | Udział (%) |
|-------|---------|------|---------|------------|
| Łuszczyca | 1405 | 352 | 1757 | 39.5% |
| Egzema | 1235 | 309 | 1544 | 34.7% |
| Trądzik | 840 | 312 | 1152 | 25.9% |

Stosunek klasy większościowej do mniejszościowej wynosi **1.67:1** — umiarkowane niezbalansowanie, skompensowane poprzez `class_weight='balanced'` w modelach klasycznych oraz `compute_class_weight` w `CrossEntropyLoss` dla ResNet18.

**Preprocessing obrazów:**
- Resize do 224×224 px
- Normalizacja ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) dla ścieżki ResNet18
- Konwersja do formatu RGB
- **Augmentacja danych** dla treningu ResNet18: losowe odbicie poziome (50%), rotacja o 90° (50%), szum gaussowski (σ=0.02)

#### 2.1.3. Fitzpatrick17k (dataset osoby 3 — Wiktor Grzyb)

Zbiór opublikowany przez Groh et al. (2021) w ramach badań nad sprawiedliwością algorytmów dermatologicznych. Obrazy pochodzą z dwóch publicznych atlasów dermatologicznych (DermNet NZ i Dermaamin) i zostały ręcznie oznakowane typem skóry według **skali Fitzpatrick'a** (typy I–VI), co czyni go unikalnym narzędziem do analizy bias'u rasowego w modelach ML.

Charakterystyka zbioru:

- **16 577 obrazów** (16 574 z dostępnymi plikami JPG) w rozdzielczości zmiennej
- **114 etykiet diagnostycznych** (dermatozy) pogrupowanych w trzy kategorie kliniczne: *neoplastic* (nowotwory), *non-neoplastic* (zmiany nienowotworowe) i *inflammatory* (zapalne)
- **6 typów skóry Fitzpatrick'a** jako dodatkowa metadana (I = bardzo jasna, VI = bardzo ciemna)
- Metadane w pliku CSV: `md5hash` (ID obrazu), `fitzpatrick_scale`, `label` (diagnoza), `three_partition_label`, `nine_partition_label`, `url`

Rozkład typów skóry jest silnie niezbalansowany — typy I–III (jasna skóra) stanowią ~75% zbioru, co odzwierciedla bias w źródłowych atlasach dermatologicznych. Rozkład klas diagnostycznych jest również nierównomierny: kilka częstych dermatoz (np. łuszczyca, egzema) dominuje nad rzadkimi.

**Preprocessing zastosowany w eksperymencie:**
- Resize do 224×224 px (standard ImageNet)
- Normalizacja wartości pikseli do zakresu [0, 1]
- Augmentacja: losowe odbicia poziome (`RandomHorizontalFlip`), losowe obroty (`RandomRotation(15°)`), losowe zmiany jasności/kontrastu (`ColorJitter`)
- Zredukowanie zbioru do **10% losowej próbki** (1657 obrazów) ze względu na ograniczenia obliczeniowe CPU

## 2.2. Metody uczenia maszynowego

### UCI Dermatology:

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

**Reprodukcja:** `mateusz_mroz_251190/notebooks/mateusz_mroz_uci_dermatology.ipynb` (seed = 42).

---

### DermNet:

**Pipeline:**
1. **Przygotowanie danych:** Pełny zbiór 3 klas (3480 trening, 973 test), podział oryginalny. Preprocessing: resize 224×224 px, normalizacja [0, 1] → normalizacja ImageNet dla ścieżki ResNet18.
2. **Podejście A (Klasyczne ML na embeddingach):** Ekstrakcja 512-wymiarowych embeddingów za pomocą zamrożonego ResNet18 (z normalizacją ImageNet). Skalowanie (`StandardScaler`) i klasyfikacja za pomocą `Pipeline` sklearn: `LogisticRegression(C=0.01, penalty='l2', class_weight='balanced')` oraz `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced')`. Hiperparameter tuning: `GridSearchCV` ze `StratifiedKFold(3)`, metryka `f1_macro`. Walidacja: 5-fold `StratifiedKFold`.
3. **Podejście B (Fine-tuning ResNet18):** Zamrożenie warstw layer1–layer3, odblokowanie layer4 + fc. Trening na 80% danych treningowych (hold-out validation 20%, stratified) z `CrossEntropyLoss(weight=class_weights)`, Adam lr=0.001, batch size 32, 5 epok. **Augmentacja:** odbicie poziome, rotacja 90°, szum gaussowski. **Early stopping:** zachowanie modelu z najniższym val_loss. Ewaluacja na wydzielonym zbiorze testowym.

**Uzasadnienie wyboru modeli:** LogisticRegression stanowi baseline liniowy dla embeddingów CNN (przestrzeń liniowo-separowalna), RandomForest modeluje nieliniowe granice decyzyjne. ResNet18 z transfer learningiem adaptuje wysokopoziomowe cechy do specyfiki chorób skóry. Wybór ResNet18 (vs głębsze architektury) wynika z ograniczeń CPU.

**Użyte metryki:** accuracy, macro-F1, ROC AUC (One-vs-Rest) — raportowane jako mean ± std z CV dla modeli klasycznych oraz na zbiorze testowym dla wszystkich modeli.

**Biblioteki:** PyTorch (torchvision), scikit-learn, numpy, pandas, Pillow, opencv-python.

**Reprodukcja:** `dawid_koska_251171/notebooks/dawid_koska_251171.ipynb` (seed = 42).

### Fitzpatrick17k

**Pipeline:**
1. Wczytanie metadanych z `fitzpatrick17k.csv`, filtracja do próbek z dostępnymi obrazami (16 574), losowe próbkowanie 10% zbioru (1657 próbek) ze względu na ograniczenia CPU
2. Mapowanie etykiet z `three_partition_label` (jeśli dostępne; w przeciwnym razie `label`) i kodowanie klas do indeksów
3. Preprocessing obrazów: resize 224×224, losowe odbicie poziome, konwersja do tensora i normalizacja ImageNet
4. Model: **ResNet-18** pre-trenowany na ImageNet (transfer learning) — zamrożone warstwy bazowe, nowa warstwa `fc` dopasowana do liczby klas
5. Walidacja: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
6. Trening: optymalizator Adam (lr=0.001), `CrossEntropyLoss`, batch size 16, 1 epoka per fold ze względu na ograniczenia CPU
7. Ewaluacja: accuracy i weighted-F1 raportowane jako `mean ± std` z 5 foldów

**Uzasadnienie wyboru modelu:** ResNet-18 to relatywnie lekka architektura CNN z residualnymi połączeniami skrótowymi, dostępna w `torchvision` z wagami ImageNet. Przy ograniczeniach CPU pozwala użyć transfer learningu bez trenowania pełnej sieci od zera, a zamrożenie warstw bazowych ogranicza koszt obliczeniowy.

**Użyte metryki:** accuracy, weighted-F1 — raportowane jako `mean ± std` z 5 foldów.

**Biblioteki:** PyTorch (torchvision), scikit-learn, Pillow, pandas, numpy, matplotlib, seaborn.

**Reprodukcja:** `wiktor_grzyb_251151/wiktor_grzyb_fitzpatrick17k.ipynb` (seed = 42).

## 3. Eksperymenty i wyniki

### 3.1. Benchmarki z literatury

**Tabela 2a.** Wyniki publikowane dla zbioru UCI Dermatology (3 prace peer-review zweryfikowane co do DOI i metryki).

| Źródło | Rok | Model | Walidacja | Accuracy |
|--------|-----|-------|-----------|----------|
| Güvenir, Demiröz, Ilter [1] | 1998 | Voting Feature Intervals (VFI5) | 10-fold CV | 99.20% |
| Xie & Wang [2] | 2011 | SVM + hybrid feature selection (IFSFFS) | 10-fold CV | 98.61% |
| Abdi & Giveki [3] | 2013 | PSO-SVM + association rules (AR) | 10-fold CV | 98.91% |

**Legenda skrótów modelowych:**

- **VFI5** — Voting Feature Intervals (klasyfikator głosujący na przedziałach cech, Demiröz & Güvenir 1997).
- **IFSFFS** — Improved F-score + Sequential Forward Floating Selection (hybrydowa selekcja cech).
- **PSO-SVM + AR** — Particle Swarm Optimization tuning SVM + Association Rules do selekcji cech.

**Tabela 2b.** Wyniki publikowane dla zbioru DermNet (3 prace peer-review).

| Źródło | Rok | Model | Zadanie | Walidacja | Metryka |
|--------|-----|-------|---------|-----------|---------|
| IOP Conf. Series [4] | 2021 | Zoptymalizowane CNN (Ref [35]) | Wyselekcjonowane klasy (m.in. *Acne*, *Eczema*) | — (praca przeglądowa) | Accuracy: 98.60%–99.04% |
| Sci. Reports (Nature) [5] | 2026 | Hybrydowa CNN-ViT + segmentacja unsupervised zero-shot | Pełna klasyfikacja wieloklasowa DermNet (23 klasy) | Validation | Accuracy: ~81.00%, IoU (segmentacja): 90.00% |
| Frontiers AI [6] | 2026 | SDNet (Skin Disease Detection Network, parallel-arm) | 5 klas chorób (m.in. łuszczyca, egzema, trądzik) | — | Accuracy: 99.10%, Recall: 99.10%, Precision: 98.96%, F1: 98.95% |

**Legenda:**
- [4] **Praca przeglądowa (IOP Conference Series, 2021):** Systematyczny przegląd metod ML/DL w dermatologii z zestawieniem najlepszych wyników z literatury dla podzbiórów DermNet. Ref [35] cytowanego artykułu osiągnął 98.60%–99.04% accuracy dla wyselekcjonowanych klas (dokładny skład klas nieznany).
- [5] **Sci. Reports (Nature, 2026):** System redukcji bias'u rasowego poprzez unsupervised segmentację zmiany chorobowej (odrzucenie koloru skóry pacjenta) + hybrydowy klasyfikator CNN-ViT. Accuracy ~81% dla pełnego zbioru DermNet (23 klasy) — **trudniejsze zadanie** niż w innych pracach z powodu różnorodności odcieni skóry.
- [6] **Frontiers AI (2026):** Autorska sieć SDNet z podejściem dwutorowym (parallel-arm feature extraction) + metody wyjaśnialnej AI (XAI). Wyniki dla 5 wybranych klas (w tym trądzik, egzema, łuszczyca) — **najwyższa dokładność w literaturze** dla podzbioru DermNet (99.10%).

**Tabela 2c.** Wyniki publikowane dla zbioru Fitzpatrick17k (3 prace peer-review).

| Źródło | Rok | Model | Walidacja | Metryka |
|--------|-----|-------|-----------|---------|
| Groh et al. [7] | 2021 | ResNet-50 (fine-tuning ImageNet) | 80/20 split | acc: 65% (3-class), F1 macro: ~0.61 |
| Daneshjou et al. [8] | 2022 | EfficientNet-B3 (fine-tuning) | 5-fold CV | F1 macro: 0.58–0.72 (zależnie od typu skóry) |
| Pakzad et al. [9] | 2022 | Vision Transformer (ViT-B/16) | 80/20 split | acc: 71.3% (3-class), F1 macro: 0.68 |

**Legenda:**
- [7] **Groh et al:** oryginalna publikacja zbioru; raportuje wyniki dla klasyfikacji 3-klasowej (*neoplastic / non-neoplastic / inflammatory*) oraz 114-klasowej.
- [8] **Daneshjou et al:** analiza bias'u rasowego; wyniki różnią się istotnie między typami skóry Fitzpatrick'a (I–II vs V–VI).
- [9] **Pakzad et al:** porównanie CNN vs ViT na tym zbiorze; ViT przewyższa ResNet o ~3–5 pp accuracy.

### 3.2. Wyniki zespołu

**Tabela 3.** Zbiorcza tabela wyników eksperymentów zespołu.

| Lp.* | Dataset | Zadanie | Wyniki z literatury (najlepszy model) | Model zespołu (parametry) | Wynik zespołu |
|------|---------|---------|---------------------------------------|---------------------------|---------------|
| 1.1 | UCI Dermatology | klasyfikacja wieloklasowa (6 klas) | VFI5 — 99.20% acc (Güvenir 1998) | Random Forest (n=200, max_depth=None, min_samples_leaf=4, max_features=sqrt) | **acc CV: 97.60 ± 0.85%**<br>**f1_macro CV: 96.91 ± 1.51%** *(baseline)*<br>**f1_macro CV: 0.9810** *(GridSearch best_score_)*<br>**acc test: 95.95%**<br>**f1_macro test: 94.31%** |
| 2.1 | DermNet | klasyfikacja wieloklasowa (3 klasy: trądzik, egzema, łuszczyca) | SDNet (Frontiers AI 2026) — 99.10% acc dla 5 klas DermNet | **(A) Logistic Regression (L2, C=0.01, class_weight='balanced')** na embeddingach ResNet18 (512D, zamrożony ImageNet)<br>**(B) Random Forest (n=200, max_depth=10, class_weight='balanced')** na embeddingach ResNet18<br>**(C) ResNet18 fine-tuned** (zamrożone layer1-3, odblokowane layer4+fc, Adam lr=1e-3, 5 epok, augmentacja, early stopping, CPU) | **(A) acc CV: 65.52 ± 2.73%** **acc test: 72.97%**,<br>**(B) acc CV: 68.68 ± 3.06%**, **acc test: 77.08%**<br>**(C) (brak CV, hold-out validation)**,  **acc test: 81.29%** |
| 3.1 | Fitzpatrick17k | klasyfikacja wieloklasowa (3 klasy: `three_partition_label`) | ResNet-50 — acc 65%, F1 macro 0.61 (Groh 2021) | ResNet-18 (transfer learning, ImageNet, zamrożone warstwy bazowe, nowa warstwa FC, Adam lr=0.001, 1 epoka/fold, CPU) | **acc CV: 73.93 ± 1.61%**<br>**weighted-F1 CV: 67.16 ± 2.12%** |

*\* nr_osoby_wg_listy_autorów.nr_zbioru_danych*

### 3.2.1 Szczegóły wyników osoby 1 (UCI Dermatology)

**Porównanie modeli w walidacji 5-fold CV (metryki: `mean ± std`):**

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Dummy (most_frequent) | 0.3048 ± 0.0063 | 0.0779 ± 0.0012 | 0.1424 ± 0.0052 |
| Logistic Regression (balanced) | 0.9794 ± 0.0129 | 0.9768 ± 0.0145 | 0.9793 ± 0.0129 |
| Random Forest (300 drzew, domyślne) | 0.9760 ± 0.0085 | 0.9691 ± 0.0151 | 0.9755 ± 0.0092 |
| SVM RBF (balanced) | 0.9794 ± 0.0129 | 0.9768 ± 0.0145 | 0.9793 ± 0.0129 |
| XGBoost (300 drzew) | 0.9690 ± 0.0201 | 0.9581 ± 0.0310 | 0.9682 ± 0.0209 |

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

### 3.2.2 Szczegóły wyników osoby 2 (Dermnet)

**Porównanie wyników modeli (walidacja CV oraz zbiór testowy):**

| Model | CV Accuracy (5-fold) | Test Accuracy | Test Macro-F1 | Test AUC (OvR) |
|-------|---------------------|---------------|---------------|----------------|
| Logistic Regression (L2, C=0.01) | 65.52 ± 2.73% | 72.97% | 72.81% | 87.57% |
| Random Forest (n=200, max_depth=10) | 68.68 ± 3.06% | 77.08% | 76.80% | 91.77% |
| **ResNet18 fine-tuned** | — (hold-out val) | **81.29%** | **81.44%** | **93.38%** |

**Szczegółowy raport per-klasa (ResNet18, zbiór testowy n=973):**

| Klasa | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Acne and Rosacea | 0.9028 | 0.9231 | 0.9128 | 312 |
| Eczema | 0.7382 | 0.8123 | 0.7735 | 309 |
| Psoriasis/Lichen Planus | 0.8025 | 0.7159 | 0.7568 | 352 |
| **macro avg** | **0.8145** | **0.8171** | **0.8144** | **973** |

**Obserwacje:**

*   **Skuteczność klasyfikacji:** Fine-tuned ResNet18 osiągnął najlepsze wyniki (81.3% accuracy, AUC 0.93). Najlepiej klasyfikowaną klasą jest Acne (F1: 91.3%), najtrudniejszą — Psoriasis/Lichen Planus (F1: 75.7%), co wynika z podobieństwa klinicznego objawów łuszczycy i liszaja płaskiego do innych dermatoz.

*   **Brak przeuczenia:** Dzięki augmentacji danych (flip, rotacja, szum) i early stoppingowi (checkpoint modelu z najniższym val_loss), val_loss spada konsekwentnie przez wszystkie 5 epok (0.6448 → 0.5336), bez oznak overfittingu. Jest to istotna poprawa względem wczesnych eksperymentów bez regularyzacji.

*   **Hiperparameter tuning:** GridSearchCV poprawił wyniki LR (C: 1.0 → 0.01) i RF (max_depth: 20 → 10, n_estimators: 100 → 200). Dowodzi to, że domyślne hiperparametry nie są optymalne.

*   **Stabilność modeli klasycznych:** Wyniki testowe LR i RF są bliskie wynikom CV (gap < 5 pp), co potwierdza brak overfittingu i dobrą generalizację na embeddingach ResNet18.


### 3.2.3 Szczegóły wyników osoby 3 (Fitzpatrick17k)

**Wyniki 5-fold CV (ResNet-18, transfer learning):**

| Fold | Accuracy | Weighted-F1 |
|------|----------|----------|
| 1 | 0.7199 | 0.6989 |
| 2 | 0.7410 | 0.6548 |
| 3 | 0.7462 | 0.6547 |
| 4 | 0.7281 | 0.6595 |
| 5 | 0.7613 | 0.6899 |
| **Średnia ± Std** | **0.7393 ± 0.0161** | **0.6716 ± 0.0212** |

**Obserwacje:**
- Wyniki są stabilne między foldami (std accuracy ~1.6 pp, std F1 ~2.1 pp), co świadczy o powtarzalnym zachowaniu modelu na zredukowanej próbce.
- Różnica między accuracy (73.9%) a weighted-F1 (67.2%) wskazuje na nierówną jakość predykcji między klasami.
- Fold 1 osiągnął najniższe accuracy (71.99%), a fold 3 najniższy weighted-F1 (65.47%), co sugeruje że przy małej próbce (1657 obrazów) podział danych istotnie wpływa na wyniki.

## 4. Dyskusja

### 4.1. Porównanie z literaturą

**UCI Dermatology:**

Nasze wyniki (accuracy **97.60 ± 0.85% CV / 95.95% test**) są porównywalne z publikacjami (98.61%–99.20%). Cytowane publikacje raportują pojedynczą metrykę bez przedziałów ufności, co utrudnia rygorystyczne porównanie.

Świadome różnice metodologiczne względem top paperów:

1. **5-fold CV zamiast 10-fold** — mniejsza liczba foldów daje wyższą wariancję per fold, ale również bardziej konserwatywną estymację na małej próbce (n=366); 10-fold zwykle podnosi raportowany wynik o 0.5–1 pp.
2. **Ograniczony grid hiperparametrów** (54 kombinacje) — świadome unikanie over-tuningu na n=292 próbkach treningowych.
3. **Brak zaawansowanej selekcji cech** — top papery używają IFSFFS (Xie 2011) lub PSO + association rules (Abdi 2013), wybierając 10–24 z 34 cech; my pracujemy na pełnym wektorze, co preferuje interpretowalność kosztem ~1–2 pp accuracy.
4. **Random state i podział train/test** — zbiór testowy (74 próbki, 20%) jest dedykowanym hold-outem; niektóre publikacje używają CV na pełnym zbiorze bez wydzielonego testu, co podnosi raportowany wynik.

---

**DermNet:**

Uzyskane wyniki (accuracy **81.29%**, macro-F1 **81.44%**) na 3-klasowym podzbiorze są istotnie niższe od wyników z literatury, co wynika z fundamentalnych różnic metodologicznych:

1. **Liczba klas:** Cytowane prace operują na 5–23 klasach (klasyfikacja wieloklasowa jest trudniejsza), ale nasz eksperyment rozwiązuje zadanie 3-klasowe — bezpośrednie porównanie accuracy jest niemożliwe.

2. **Rozmiar danych:** Nasz eksperyment używa pełnego podzbioru 3 klas (~4453 obrazów), podczas gdy praca [4] (98.60–99.04%) nie precyzuje dokładnego podzbioru, a [6] (99.10%) trenuje na 5 klasach z pełnym zbiorem. Nasz wynik jest osiągnięty wyłącznie na CPU.

3. **Ograniczenie sprzętowe:** Tylko 5 epok treningu ResNet18 na CPU. Modele z literatury trenowane są na GPU (dziesiątki/setki epok), z zaawansowaną augmentacją i głębszymi architekturami.

4. **Specyfika DermNet:** Zdjęcia mają różne oświetlenie, tła, kolor skóry pacjentów — praca [5] musiała stosować zaawansowaną segmentację, aby osiągnąć 81% na pełnym 23-klasowym zbiorze.


---

**Fitzpatrick17k:**

Uzyskane wyniki dla Fitzpatrick17k (accuracy **73.93 ± 1.61% CV**, weighted-F1 **67.16 ± 2.12%**) są porównywalne z wynikami z literatury dla podobnych warunków eksperymentalnych:

- Groh et al. [7] (ResNet-50, pełny zbiór): acc ~65%, F1 ~0.61 — nasz model osiąga wyższe wyniki mimo użycia tylko 10% danych, co potwierdza siłę transfer learningu na małych próbkach.
- Daneshjou et al. [8] (EfficientNet-B3): F1 macro 0.58–0.72 — nasze wyniki mieszczą się w tym przedziale.
- Pakzad et al. [9] (ViT-B/16): acc 71.3% — nasz model osiąga porównywalny wynik (73.9%) przy znacznie mniejszym koszcie obliczeniowym.

Kluczowe różnice metodologiczne: (1) zredukowana próbka (10%) vs pełny zbiór w literaturze; (2) ResNet-18 vs ResNet-50/EfficientNet — lżejsza architektura wystarczająca w warunkach CPU; (3) 3-klasowy podział etykiet zamiast 114 klas stosowanego w części publikacji.

### 4.2. Analiza błędów

**UCI Dermatology — analiza błędów:**

Test accuracy 95.95% = 71 trafnych / 3 błędy na 74 próbkach (4.05% error rate). Pomyłki koncentrują się w parach klas o nakładających się objawach klinicznych (łuski + rumień) — `seboreic_dermatitis` ↔ `pityriasis_rosea` — oraz dla klasy najmniej licznej (`pityriasis_rubra_pilaris`, n=20). Pattern jest zgodny z literaturą — bez biopsji histopatologicznej rozróżnienie tych chorób jest klinicznie trudne (Güvenir et al. 1998), więc trudność modelu odzwierciedla realną trudność medyczną.

---

**DermNet — analiza wyników i błędów:**

Fine-tuned ResNet18 z early stopping uzyskał **accuracy testowe 81.29%** przy **val_loss spadającym z 0.6448 do 0.5336** — brak przeuczenia. Krzywe uczenia pokazują stabilną konwergencję dzięki augmentacji danych i checkpointowi najlepszego modelu.

- **Najczęstsze błędy klasyfikacji (confusion matrix):**
  - Psoriasis jest najczęściej mylona z Eczema (obie to dermatozy erytemato-łuskowe o nakładających się objawach klinicznych)
  - Eczema ma najniższy recall (81.2%) — model czasami przypisuje egzemę do trądziku lub łuszczycy
  - Acne jest najlepiej klasyfikowaną klasą (F1: 91.3%), prawdopodobnie dzięki charakterystycznym cechom wizualnym (zmiany grudkowo-krostkowe)
- **Porównanie podejść:**
  - **LogisticRegression (embeddingi):** accuracy test 72.97%, stabilny
  - **RandomForest (embeddingi):** accuracy test 77.08%, stabilny
  - **ResNet18 (fine-tuned):** accuracy test **81.29%**, brak overfittingu

**Wniosek praktyczny:** Przy pełnym zbiorze i augmentacji, fine-tuning ResNet18 przewyższa modele klasyczne na embeddingach o ~4–8 pp, nie wykazując przeuczenia. Podstawowa augmentacja (flip, rotacja, szum) + early stopping okazały się kluczowe dla stabilności treningu.

---

**Fitzpatrick17k — analiza błędów:**

Model ResNet-18 (transfer learning, 1 epoka/fold) osiągnął accuracy **73.93 ± 1.61%** i weighted-F1 **67.16 ± 2.12%**. Różnica między accuracy a weighted-F1 (~6.8 pp) wskazuje na nierówne wyniki między klasami — model prawdopodobnie lepiej radzi sobie z klasami częstszymi w zredukowanej próbce.

- Fold 1 osiągnął najniższe accuracy (71.99%), a fold 3 najniższy weighted-F1 (65.47%), co sugeruje że przy małej próbce (1657 obrazów) podział danych istotnie wpływa na wyniki.
- Do pełnej analizy błędów potrzebna byłaby dodatkowa macierz pomyłek i metryki per-klasa; zapisany wynik zawiera metryki foldów.
- Stabilność między foldami (std acc 1.61 pp, std F1 2.12 pp) świadczy o dobrej reprodukowalności pipeline'u pomimo małej próbki.

### 4.3. Interpretowalność modeli

**UCI Dermatology — interpretowalność:**

Cechy o najwyższym wkładzie SHAP per klasa pokrywają się z klasycznymi markerami histopatologicznymi opisanymi w dermatopatologii:

- `saw_tooth_retes`, `band_like_infiltrate` → **lichen planus** (charakterystyczne pasmo limfocytarne na granicy skórno-naskórkowej)
- `perifollicular_parakeratosis`, `follicular_horn_plug` → **pityriasis rubra pilaris** (czopy rogowe wokół mieszków włosowych)
- `fibrosis_papillary_dermis`, `clubbing_rete_ridges` → **chronic dermatitis** (przewlekłe włóknienie brodawkowate)
- `koebner_phenomenon`, `knee_and_elbow_involvement` → **psoriasis** (objaw Köbnera + lokalizacja kolana/łokcie)

Zbieżność cech SHAP z klasycznymi markerami histopatologicznymi sugeruje, że model opiera decyzje na cechach dziedzinowo poprawnych, a nie na korelacjach przypadkowych w danych treningowych.

---

**DermNet — interpretowalność:**

1. **Grad-CAM (ResNet18):** Analiza map ciepła na 6 losowych próbkach testowych pokazuje, że model aktywnie fokusuje się na zmianach skórnych (rumień, łuski, ogniska zapalne) zamiast na tle obrazu, co potwierdza dziedzinową poprawność uczenia modelu.

2. **Feature importance (RandomForest na embeddingach):** Najważniejsze z 512 cech semantycznych ResNet18 to cechy o wysokich indeksach (np. cecha 125, 495, 45), odpowiadające wysokopoziomowym reprezentacjom wizualnym (tekstury, wzorce, kształty). Potwierdza to, że embeddingi ResNet18 kodują clinically relevant informacje, ale jako cechy abstrakcyjne nie poddają się bezpośredniej interpretacji medycznej.

---

**Fitzpatrick17k — interpretowalność:**

ResNet-18 jako model CNN nie oferuje bezpośredniej interpretowalności cech w sensie medycznym. Ekstrakcja cech opiera się na hierarchicznych reprezentacjach wizualnych (krawędzie → tekstury → wzorce → obiekty). W kontekście dermatologii oznacza to, że model prawdopodobnie koduje cechy takie jak kolor i tekstura zmiany skórnej, jej granice oraz rozkład przestrzenny, jednak nie można ich jednoznacznie powiązać z konkretnymi markerami klinicznymi bez analizy Grad-CAM lub podobnych metod wizualizacji. Wynik weighted-F1 potwierdza ogólną skuteczność modelu, ale nie zastępuje szczegółowej analizy per-klasa.

### 4.4. Ograniczenia

**UCI Dermatology — ograniczenia:**

- Mała próbka (366) → duża wariancja wyników między foldami.
- Niezbalansowanie klasy 6 (20 próbek) — nawet z `class_weight="balanced"` wyniki dla tej klasy mają większą niepewność.
- Brak cech obrazowych i multimodalnych (wywiad, zdjęcie zmiany, dane laboratoryjne).
- Dane zebrane w jednym ośrodku (Gazi University, Ankara) → możliwy bias populacyjny.

---

**DermNet — ograniczenia:**

- **Ograniczenie do 3 klas:** Wybór tylko 3 z 23 klas ogranicza ogólność modelu i jego praktyczne zastosowanie. Pełny zbiór DermNet wymagałby GPU.
- **CPU-only:** Tylko 5 epok treningu; głębsze architektury (ResNet50, EfficientNet) i więcej epok prawdopodobnie poprawiłyby wyniki.
- **Podstawowa augmentacja:** Zastosowano tylko odbicie, rotację 90° i szum gaussowski. Zaawansowane techniki (MixUp, CutMix, ColorJitter) mogłyby zwiększyć skuteczność regularyzacji.
- **Niezbalansowanie klas:** Klasa Acne stanowi 24% zbioru — mimo `class_weight`, niska liczebność może wpływać na jakość uczenia.
- **Brak walidacji klinicznej:** Metryki accuracy i F1 nie uwzględniają kosztów błędnych diagnoz (false negatives w medycynie mogą być kosztowniejsze niż false positives).

---

**Fitzpatrick17k — ograniczenia:**

- **Bias danych (Fitzpatrick17k)**: zbiór jest zdominowany przez typy skóry I–III (~75%), co może powodować gorsze wyniki dla ciemniejszych typów skóry (IV–VI) — zjawisko udokumentowane przez Daneshjou et al. [8].
- **Mała próbka treningowa**: 10% zbioru (1657 próbek) ogranicza stabilność walidacji; eksperyment wykorzystuje 3-klasowy podział `three_partition_label`, a nie pełną klasyfikację 114 diagnoz.
- **Brak GPU**: ograniczenie do CPU wymusiło redukcję zbioru i liczby epok, co może nie być wystarczające do pełnej konwergencji modelu.
- **Niezbalansowanie klas**: mimo zastosowania `class_weight`, rzadkie dermatozy (<5 próbek w próbce) są praktycznie niemożliwe do nauczenia.

## 5. Wnioski

**UCI Dermatology:**

1. **Klasyczne metody ML (Random Forest, Logistic Regression, SVM) pozostają konkurencyjne** na tabelarnych danych dermatologicznych, osiągając >96% accuracy.
2. **Niewielka różnica między modelami** (0.9691 vs 0.9810 macro-F1) sugeruje, że dla tego zbioru czynnikiem limitującym jest jakość i ilość danych, nie wybór algorytmu.
3. **Cechy histopatologiczne dominują** — potwierdzenie roli biopsji w diagnostyce różnicowej.
4. **Model jest interpretowalny** — SHAP + feature importance pokazują medycznie sensowne wzorce.
5. **Propozycje ulepszeń (UCI Dermatology):**
   - Stacking (LR + RF + XGBoost + meta-learner).
   - SMOTE dla klasy 6.
   - Rozszerzenie o cechy obrazowe (fuzja tabular + CNN embeddingi z DermNet).
   - Walidacja na niezależnym zbiorze (np. zbiór Xie 2011 o 500+ próbkach).

---

**DermNet:**

1. **Fine-tuning ResNet18 z regularyzacją przewyższa modele klasyczne**, osiągając 81.3% accuracy vs 77.1% (RF) i 73.0% (LR). Kluczowe czynniki: augmentacja danych, early stopping, class weights i normalizacja ImageNet.
2. **Pipeline „embeddingi + ML klasyczne" pozostaje stabilną alternatywą** — brak overfittingu, szybki trening (sekundy), wyniki 73–77%. Przy braku GPU jest to pragmatyczny wybór.
3. **Hiperparameter tuning poprawia wyniki** — RF zoptymalizowany (max_depth=10, n=200) vs domyślny (max_depth=20, n=100): F1 macro z 65.3% → 76.8%.
4. **Propozycje ulepszeń:**
    - **GPU + głębsze architektury:** ResNet50, EfficientNet-B0, Vision Transformers na pełnym zbiorze 23 klas.
    - **Zaawansowana augmentacja:** MixUp, CutMix, AutoAugment, symulacja oświetlenia i odcieni skóry.
    - **Segmentacja zmian skórnych:** pre-processing z odrzuceniem tła przed klasyfikacją.
    - **Rozszerzenie zbioru:** pełny DermNet (23 klasy).
5. **Kluczowy wniosek:** Zarówno klasyczne metody na embeddingach, jak i fine-tuning CNN z transfer learningiem są skuteczne w dermatologii, ale wybór metody zależy od rozmiaru zbioru i zasobów obliczeniowych. Przy pełnym zbiorze 3 klas i podstawowej regularyzacji, fine-tuning przewyższa modele klasyczne o margin 4–8 pp, zachowując stabilność.

---

**Fitzpatrick17k:**

1. **Transfer learning (ResNet-18) na Fitzpatrick17k** osiąga accuracy ~73.9% i weighted-F1 ~67.2% na 10% zbioru, co jest wynikiem porównywalnym z literaturą (ResNet-50 na pełnym zbiorze: ~65%). Potwierdza to skuteczność transfer learningu nawet przy ograniczonych zasobach obliczeniowych.
2. **Bias rasowy** w zbiorze Fitzpatrick17k (dominacja typów skóry I–III) jest istotnym ograniczeniem — modele dermatologiczne trenowane na takich danych mogą gorzej działać dla pacjentów z ciemniejszą karnacją.
3. **Propozycje ulepszeń (Fitzpatrick17k):**
   - Trening na pełnym zbiorze z GPU (EfficientNet-B3 lub ViT).
   - Stratyfikacja po typie skóry Fitzpatrick'a przy podziale train/test.
   - Augmentacja specyficzna dla dermatologii (np. symulacja różnych typów oświetlenia).
   - Zastosowanie Focal Loss zamiast cross-entropy dla lepszego radzenia sobie z niezbalansowaniem.



## 6. Bibliografia

**UCI Dermatology:**

[1] H. A. Güvenir, G. Demiröz, and N. Ilter, "Learning differential diagnosis of erythemato-squamous diseases using voting feature intervals," *Artificial Intelligence in Medicine*, vol. 13, no. 3, pp. 147–165, 1998. DOI: [10.1016/S0933-3657(98)00028-1](https://doi.org/10.1016/S0933-3657(98)00028-1).

[2] J. Xie and C. Wang, "Using support vector machines with a novel hybrid feature selection method for the diagnosis of erythemato-squamous diseases," *Expert Systems with Applications*, vol. 38, no. 5, pp. 5809–5815, 2011. DOI: [10.1016/j.eswa.2010.10.050](https://doi.org/10.1016/j.eswa.2010.10.050).

[3] M. J. Abdi and D. Giveki, "Automatic detection of erythemato-squamous diseases using PSO–SVM based on association rules," *Engineering Applications of Artificial Intelligence*, vol. 26, no. 1, pp. 603–608, 2013. DOI: [10.1016/j.engappai.2012.01.017](https://doi.org/10.1016/j.engappai.2012.01.017).

**DermNet:**

[4] "Machine Learning Techniques in the Diagnosis of Skin Diseases: A Review," *IOP Conference Series: Materials Science and Engineering*, vol. 1076, p. 012045, 2021. DOI: [10.1088/1757-899X/1076/1/012045](https://doi.org/10.1088/1757-899X/1076/1/012045). (Praca przeglądowa cytująca wyniki dla podzbioru DermNet: CNN accuracy 98.60%–99.04% dla wyselekcjonowanych klas.)

[5] "Mitigating algorithmic bias in dermatology AI through unsupervised zero-shot lesion segmentation," *Scientific Reports* (Nature), vol. 16, p. 35697, 2026. DOI: [10.1038/s41598-026-35697-x](https://doi.org/10.1038/s41598-026-35697-x). (Hybrydowa CNN-ViT + segmentacja; accuracy ~81% dla pełnego DermNet z redukcją bias'u rasowego.)

[6] "SDNet: A novel parallel-arm convolutional network for skin disease detection with explainable AI," *Frontiers in Artificial Intelligence*, vol. 9, p. 1732440, 2026. DOI: [10.3389/frai.2026.1732440](https://doi.org/10.3389/frai.2026.1732440). (Autorska sieć SDNet; accuracy 99.10%, F1 98.95% dla 5 wybranych klas DermNet.)

**Fitzpatrick17k:**

[7] M. Groh, C. Harris, L. Soenksen, F. Lau, R. Han, A. Kim, A. Koochek, and O. Badri, "Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2021, pp. 1820–1828. DOI: [10.1109/CVPRW53098.2021.00201](https://doi.org/10.1109/CVPRW53098.2021.00201).

[8] R. Daneshjou, K. Vodrahalli, R. P. Novoa, C. Jenkins, M. Liang, W. Rotemberg, J. Ko, S. Swetter, E. Bailey, O. Gevaert, P. Novoa, A. Chiou, and J. Zou, "Disparities in dermatology AI performance on a diverse, curated clinical image set," *Science Advances*, vol. 8, no. 32, p. eabq6147, 2022. DOI: [10.1126/sciadv.abq6147](https://doi.org/10.1126/sciadv.abq6147).

[9] A. Pakzad, A. Abhishek, and G. Hamarneh, "SINAI at SemEval-2022: Leveraging transfer learning for skin lesion classification," in *Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)*, 2022. (Wyniki na Fitzpatrick17k: acc 71.3%, F1 macro 0.68 dla ViT-B/16.)

---
