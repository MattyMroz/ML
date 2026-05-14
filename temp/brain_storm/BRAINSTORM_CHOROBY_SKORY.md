# BRAINSTORM: Projekt ML — Choroby skóry (temat 6)

> **Tryb:** small brainstorm
> **Autor:** Mateusz Mróz
> **Data:** 22.04.2026
> **Cel:** Zaplanować projekt grupowy ML na temat *Choroby skóry* (łuszczyca, egzema, trądzik, alergie kontaktowe) — 3 datasety per osoba, 1 wybrany do modelowania, raport wg template prowadzącego.

---

## 🎯 FAZA 1 — Definicja Problemu

**Co robimy:** Projekt z przedmiotu *Wprowadzenie do uczenia maszynowego* (dr Smółka, PŁ). Zespół do 3 osób. Każdy członek:

1. Zbiera 3 różne zbiory danych na temat *Choroby skóry* (tabelarne preferowane lub obrazy).
2. Opisuje je (źródło, cechy, licencja, rozmiar, preprocessing).
3. Znajduje ≥3 publikacje naukowe per dataset (z mierzalnymi wynikami).
4. Wybiera 1 dataset i trenuje model (klasyfikacja lub regresja).
5. Raport zbiorczy zespołu + tabela porównawcza wyników.

**Nasz zakres:** User robi całość "z naszej strony" + template raportu dla zespołu. Zakładam, że robimy **solidny bazowy setup** (struktura + 1 pełny notebook dla osoby 1 + template raportu + instrukcje dla osób 2-3).

---

## 📐 FAZA 1b — Tablica Prawdy (Constraints)

| # | Święta Zasada | Źródło | Status |
|---|---------------|--------|--------|
| 1 | Temat: Choroby skóry (łuszczyca, egzema, trądzik, alergie) — **NIE** nowotwory skóry (to temat 16) | instrukcja prowadzącego | 🔒 ABSOLUTNA |
| 2 | Preferowane dane tabelarne, clean, bez braków | user + instrukcja | 🔒 ABSOLUTNA |
| 3 | Jupyter notebook samowystarczalny, dane z lokalnego `/data/`, bez GPU | instrukcja (pkt 9) | 🔒 ABSOLUTNA |
| 4 | ≥3 publikacje per dataset z mierzalnymi metrykami | instrukcja (pkt 4) | 🔒 ABSOLUTNA |
| 5 | Walidacja min. k-fold (k≥5), stratyfikacja, techniki dla niezbalansowanych jeśli potrzeba | instrukcja (pkt 5) | 🔒 ABSOLUTNA |
| 6 | Struktura oddania: raport.pdf + /data/ + /notebooks/ + /env/requirements.txt | instrukcja (pkt 8) | 🔒 ABSOLUTNA |
| 7 | Komentarze Markdown wyjaśniające kroki (dla usera — "żebym się poprawił") | user | 🔒 ABSOLUTNA |
| 8 | Python 3.x, scikit-learn, pandas, matplotlib/seaborn, istniejące `.venv` w repo | repo | 🔒 ABSOLUTNA |

---

## 💡 FAZA 2 — Dywergencja: Kandydaci na Datasety

### Tabelarne (priorytet — user chce clean, bez braków)

#### Pomysł 1: **Dermatology Dataset (UCI)** 🏆
- **Opis:** 6 chorób erytemato-łuskowych (łuszczyca, łojotokowe zapalenie skóry, liszaj płaski, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris), 366 próbek, 34 cechy (11 klinicznych + 22 histopatologiczne + wiek).
- **Link:** https://archive.ics.uci.edu/dataset/33/dermatology
- **Licencja:** CC BY 4.0
- **Braki:** tylko 8 NaN w kolumnie `age` — łatwo sobie poradzić.
- **Mocne:** Klasyczny benchmark (cytowany w dziesiątkach prac), idealny do klasyfikacji wieloklasowej, tabelarny, mały → szybki trening, mocno pasuje do tematu (łuszczyca w core).
- **Słabe:** Mała próbka (366), nieco niezbalansowany (klasa 6 ma ~20 próbek).
- **Ryzyko:** overfitting przy złożonych modelach; mitigacja: k-fold CV, proste modele (LR, RF, SVM).
- **Ocena:** ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ (10/10) — wzorcowy wybór dla tego tematu.
- **Test tablicy prawdy:** ✅ Wszystkie zasady spełnione.

#### Pomysł 2: **HAM10000 — metadata-only (tabelaryczna wersja)** ⚠️
- **Opis:** 10015 dermoskopowych obrazów zmian skórnych, 7 klas — **ALE 5 z 7 klas to nowotwory/zmiany przednowotworowe** (melanoma, BCC, AKIEC, NV). Tylko `bkl` (benign keratosis) i `df` (dermatofibroma) to zmiany niezłośliwe.
- **Problem z naszym tematem:** HAM10000 to głównie czerniak → **temat 16, nie 6**. ❌
- **Ocena:** ⭐⭐⭐☆☆☆☆☆☆☆ (3/10) — zły temat.
- **Test tablicy prawdy:** ❌ Narusza zasadę #1 (to nowotwory, nie dermatozy).

#### Pomysł 3: **Acne Severity Dataset (Kaggle — ACNE04)**
- **Opis:** Zdjęcia twarzy z oceną nasilenia trądziku w skali Hayashi (4 poziomy: mild/moderate/severe/very severe), ~1457 obrazów.
- **Link:** https://github.com/xpwu95/LDL / mirrors na Kaggle
- **Licencja:** research only
- **Mocne:** Dokładnie trądzik → temat 6, ordinal classification.
- **Słabe:** Obrazy (nie tabelarne) → wymaga CNN lub transfer learning → więcej zachodu.
- **Ocena:** ⭐⭐⭐⭐⭐⭐⭐☆☆☆ (7/10) — dobry drugi/trzeci dataset dla różnorodności.

#### Pomysł 4: **Atopic Dermatitis / Eczema — EASI scores (research datasets)**
- **Opis:** Datasety z ocen nasilenia egzemy (EASI, SCORAD). Dostępne np. przez Mendeley Data, PhysioNet.
- **Status:** większość to obrazy + metadata tabelarne. Są jakieś, ale rzadkie i często licencja research-only.
- **Ocena:** ⭐⭐⭐⭐⭐⭐☆☆☆☆ (6/10) — trudniejsze do znalezienia, ale silnie pasuje do tematu.

#### Pomysł 5: **DermNet NZ Image Dataset (Kaggle mirror)**
- **Opis:** ~19500 obrazów, 23 kategorie chorób skóry (w tym łuszczyca, egzema, trądzik, dermatitis, alergie). Idealny tematycznie.
- **Link:** https://www.kaggle.com/datasets/shubhamgoel27/dermnet
- **Licencja:** educational use
- **Mocne:** Szerokie spektrum dermatoz → temat 6 w core, dużo publikacji używa tego setu (lub DermNet skin disease atlas).
- **Słabe:** Obrazy → CNN potrzebny, klasy niezbalansowane, część obrazów niskiej jakości.
- **Ocena:** ⭐⭐⭐⭐⭐⭐⭐⭐☆☆ (8/10) — bardzo dobry obrazowy kandydat.

#### Pomysł 6: **PAD-UFES-20 (tabelarny + obrazy)** ⚠️
- **Opis:** 2298 próbek zmian skórnych, 6 klas z Brazylii. Zawiera dane kliniczne **tabelarne** (wiek, lokalizacja, historia rodzinna, itp.) + obrazy.
- **Problem:** 3 z 6 klas to nowotwory (BCC, SCC, MEL). Klasy niezłośliwe (NEV, SEK, ACK) nie są w core tematu 6.
- **Ocena:** ⭐⭐⭐⭐☆☆☆☆☆☆ (4/10) — niepasujący temat.

#### Pomysł 7: **Psoriasis Severity Dataset (PASI scores)**
- **Opis:** Datasety do oceny PASI (łuszczyca). Dostępne np. przez publikacje (Meienberger 2020, PsoriasisPic).
- **Status:** raczej research-only, często trudne do pobrania.
- **Ocena:** ⭐⭐⭐⭐⭐⭐☆☆☆☆ (6/10) — idealny tematycznie, ale dostępność?

#### Pomysł 8: **Monkeypox / Mpox Skin Lesion Dataset**
- **Opis:** Obrazy zmian skórnych mpox vs inne choroby skóry.
- **Ocena:** ⭐⭐⭐⭐☆☆☆☆☆☆ (4/10) — niszowy, głównie infekcyjne, nie core tematu.

#### Pomysł 9: **Skin Cancer MNIST / Skin Disease Image Dataset (Kaggle)**
- Podobne do HAM10000 — głównie onkologia. ❌

#### Pomysł 10: **Fitzpatrick17k**
- **Opis:** 16577 obrazów, 114 chorób skóry oznaczonych typami skóry Fitzpatrick. Zawiera wiele dermatoz niezłośliwych.
- **Link:** https://github.com/mattgroh/fitzpatrick17k
- **Licencja:** research, CC BY-NC-SA
- **Mocne:** Szeroki wachlarz dermatoz, przydatne do badania bias (skin tone).
- **Słabe:** Obrazy, ogromny, wymaga preprocessing.
- **Ocena:** ⭐⭐⭐⭐⭐⭐⭐☆☆☆ (7/10) — solidny obrazowy z bonus (fairness angle).

---

## 📊 FAZA 3 — Konwergencja: Ranking

### Matryca (kryteria: temat-fit, tabelarność, clean, publikacje, rozmiar, feasibility bez GPU)

| Dataset | Temat-fit (25%) | Tabular (20%) | Clean (15%) | Publikacje (15%) | Feasibility (15%) | Rozmiar OK (10%) | SUMA |
|---------|-----------------|---------------|-------------|------------------|-------------------|------------------|------|
| **1. UCI Dermatology** | 10 | 10 | 9 | 10 | 10 | 9 | **9.7** |
| 5. DermNet | 9 | 2 | 6 | 8 | 5 | 8 | **6.4** |
| 10. Fitzpatrick17k | 8 | 3 | 7 | 8 | 4 | 7 | **6.1** |
| 3. Acne (ACNE04) | 8 | 2 | 7 | 7 | 5 | 8 | **6.0** |
| 7. Psoriasis PASI | 9 | 5 | 5 | 6 | 5 | 5 | **5.9** |
| 4. Eczema/AD | 8 | 5 | 5 | 6 | 5 | 5 | **5.8** |
| 2. HAM10000 | 2 | 3 | 9 | 10 | 7 | 9 | **5.0** (zły temat) |
| 6. PAD-UFES-20 | 3 | 7 | 8 | 8 | 7 | 8 | **5.8** (zły temat) |

### Strategie decyzyjne

**1. Eliminacja negatywna:** Odrzucam HAM10000, PAD-UFES-20, Skin Cancer MNIST (naruszają zasadę #1 — to nowotwory). ✂️

**2. Pareto 80/20:** UCI Dermatology daje 80% wartości projektu za 20% wysiłku (mały, clean, klasyczny benchmark, dużo literatury). MUST PICK.

**3. Premortum:** "Projekt oceniony słabo — dlaczego?"
- Datasety zbyt podobne → brak różnorodności. 🚨 Mitigacja: mix tabular + obrazowy.
- Brak publikacji benchmark → mitigacja: UCI ma setki cytowań, DermNet też.
- Overfitting → mitigacja: k-fold, proste modele + 1 złożony dla porównania.

---

## 🏆 FAZA 6 — Rekomendacja Finalna: 3 Datasety

### Osoba 1 (user) — **UCI Dermatology** ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
**Dlaczego:** Tabelarny, clean, idealnie pasuje do tematu (łuszczyca + 5 innych dermatoz), klasyczny benchmark → łatwo znaleźć 3+ publikacje, szybki trening bez GPU, dużo możliwości edukacyjnych (klasyfikacja wieloklasowa, feature importance, SHAP, CV, niezbalansowanie). **To będzie nasz DO-MODELOWANIA.**

### Osoba 2 — **DermNet (Kaggle mirror)** ⭐⭐⭐⭐⭐⭐⭐⭐
**Dlaczego:** Obrazowy, szerokie spektrum dermatoz (łuszczyca, egzema, trądzik, dermatitis), daje kontrast do tabularnego. Można użyć transfer learning (Lab12). Dobra narracja "obraz vs tabular".

### Osoba 3 — **Fitzpatrick17k** ⭐⭐⭐⭐⭐⭐⭐
**Dlaczego:** Obrazy + metadata (Fitzpatrick skin type) → można zrobić tabelaryczną ekstrakcję cech albo czyste klasyfikacja obrazów + analiza bias. Dodaje wymiar fairness/ML ethics do raportu.

**Plan B (fallback jeśli któryś nie pobierze się):**
- Zamiast DermNet → ACNE04 (Kaggle)
- Zamiast Fitzpatrick17k → Eczema/AD dataset (Mendeley)

---

## 🧭 FAZA 4 — Deep Dive: UCI Dermatology (dataset główny usera)

### Plan implementacji notebooka

```
1. Import + wczytanie danych → verify: df.shape == (366, 35)
2. EDA
   - Rozkład klas → countplot
   - Missing values (age column) → imputacja medianą
   - Distribucja cech → heatmap korelacji, histogramy
   verify: brak NaN po preprocessing
3. Preprocessing
   - Train/test split stratified 80/20
   - StandardScaler (dla modeli wrażliwych na skalę)
   verify: stratyfikacja potwierdzona
4. Baseline
   - Dummy classifier (most_frequent) → baseline accuracy
   - Logistic Regression (one-vs-rest, L2) → main baseline
   verify: LR accuracy > Dummy
5. Modele
   - Random Forest (n_estimators=200)
   - SVM (RBF kernel, C/gamma tuned)
   - Gradient Boosting (XGBoost/LightGBM)
   verify: wszystkie modele zbiegają, brak błędów
6. Walidacja
   - StratifiedKFold (k=5)
   - Metryki: accuracy, macro-F1, weighted-F1, confusion matrix
   - CI przez std z k-fold
   verify: CI reportowany w tabeli
7. Strojenie hiperparametrów
   - GridSearchCV lub RandomizedSearchCV dla najlepszego modelu
   verify: poprawa vs default
8. Interpretacja
   - Feature importance (RF)
   - SHAP summary plot
   - Confusion matrix per-class
   verify: wizualizacje czytelne
9. Porównanie z literaturą
   - Tabela: nasze wyniki vs 3+ publikacji
   verify: tabela w notebooku + raporcie
10. Wnioski
    - Które cechy najbardziej wpływają, ograniczenia, propozycje ulepszeń
```

### Zależności
- Pakiety: scikit-learn, pandas, numpy, matplotlib, seaborn, shap, xgboost (opcjonalnie lightgbm), ucimlrepo (opcjonalnie dla wygody)
- Wszystko dostępne w `.venv` (sprawdzić pyproject.toml)

### Ryzyka
| Ryzyko | P | Wpływ | Mitygacja |
|--------|---|-------|-----------|
| Niezbalansowanie klas (klasa 6 ~5%) | ŚREDNIE | ZNACZĄCY | stratified CV, class_weight='balanced', może SMOTE |
| Mała próbka → duża wariancja | WYSOKIE | ZNACZĄCY | k=5 CV, raportuj std, powtórzenia |
| SHAP może być wolny dla small data — OK | NISKIE | MAŁY | n/a |

### Metryki sukcesu
- Accuracy > 95% (UCI Dermatology zwykle daje 97-99% w literaturze)
- Macro-F1 > 0.90 (odporność na niezbalansowanie)
- Zgodność z benchmarks z literatury (ref: Güvenir 1998, Cestnik, i liczne nowsze prace)

---

## ✅❌ FAZA 5 — Podział Kontekstowy

### ✅ Robimy w MVP (osoba 1 / user)
- Full pipeline notebook dla UCI Dermatology (EDA → model → interpretacja)
- Metadata table dla 3 datasetów (wspólna)
- Template raportu Markdown zgodny z `template.docx`
- `requirements.txt` w `/env/`
- `README.md` w `PROJEKT/` z instrukcją uruchomienia
- Placeholdery dla osoby 2 i 3 (stub notebooks + README co robić)

### ❌ NIE robimy (poza scope user'a jako 1 osoby)
- Pełny trening CNN na DermNet (osoba 2 zrobi sama — dajemy stub)
- Pełny trening na Fitzpatrick17k (osoba 3 zrobi sama — dajemy stub)
- Pobieranie wszystkich 3 datasetów — tylko UCI Dermatology jest małe i automatyczne; pozostałe wymagają Kaggle API / manual download → instrukcja w README

### ⚠️ Zależy od decyzji usera
- Czy pobieramy DermNet i Fitzpatrick17k lokalnie, czy zostawiamy instrukcję dla osób 2/3?
- Czy user chce sam zrobić notebooki osób 2 i 3 (full)? → BIG scope.
- Szukanie publikacji per dataset — 3 per UCI zrobię na 100%; dla pozostałych sugestie + linki (osoby 2/3 mogą rozszerzyć).

---

## 🎯 FAZA 6 — Finalna Architektura Projektu

```
PROJEKT/
├── DOC/                                  # (istnieje) instrukcje + template
├── README.md                             # (nowy) jak uruchomić, kto co robi
├── raport.md                             # (nowy) główny raport (template wypełniony)
├── data/
│   ├── uci_dermatology/
│   │   ├── dermatology.data              # pobrane automatycznie
│   │   └── dermatology.names
│   ├── dermnet/                          # (placeholder — manual download)
│   │   └── README.md
│   └── fitzpatrick17k/                   # (placeholder — manual download)
│       └── README.md
├── notebooks/
│   ├── mateusz_mroz_uci_dermatology.ipynb    # ← PEŁNY pipeline
│   ├── osoba2_dermnet.ipynb                   # ← STUB z instrukcją
│   └── osoba3_fitzpatrick17k.ipynb            # ← STUB z instrukcją
└── env/
    └── requirements.txt
```

---

## 🔄 FAZA 7 — Plan Wykonawczy (Iteracyjny)

Pełna lista kroków z kryteriami sukcesu — w pliku **`BRAINSTORM_CHOROBY_SKORY_SUMMARY.md`**.
