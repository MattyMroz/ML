# 📋 SUMMARY: Projekt ML — Choroby Skóry

> **Źródło:** [BRAINSTORM_CHOROBY_SKORY.md](BRAINSTORM_CHOROBY_SKORY.md)
> **Data:** 22.04.2026 | **Tryb:** small brainstorm
> **Temat:** 6. Choroby skóry (łuszczyca, egzema, trądzik, alergie kontaktowe)

---

## TL;DR

Projekt grupowy ML (do 3 osób) — każdy zbiera 3 datasety, wybiera 1 do modelowania, raport zbiorczy. User robi **pełną część osoby 1** (UCI Dermatology — tabelarny, clean, 6 klas dermatoz) + **template całego raportu** + **stuby dla osób 2 i 3** (DermNet obrazowy, Fitzpatrick17k obrazy+metadata). Notebook z komentarzami Markdown żeby user zrozumiał każdy krok.

---

## 🏆 Rekomendacja — 3 Datasety

| # | Osoba | Dataset | Typ | Klasy | Ocena | Rola |
|---|-------|---------|-----|-------|-------|------|
| 1.1 | **User (osoba 1)** | **UCI Dermatology** | tabular | 6 (łuszczyca, seborrheic, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris) | 10/10 | **GŁÓWNY DO MODELOWANIA** |
| 2.1 | Osoba 2 | DermNet (Kaggle) | obrazy | 23 (łuszczyca, egzema, trądzik, dermatitis, alergie...) | 8/10 | Stub + instrukcja |
| 3.1 | Osoba 3 | Fitzpatrick17k | obrazy + metadata | 114 | 7/10 | Stub + instrukcja |

**Uzasadnienie wyboru UCI Dermatology do modelowania:**
- Clean, małe (366), tabelarny → szybki trening bez GPU
- Klasyczny benchmark (setki cytowań) → łatwo znaleźć ≥3 publikacje z metrykami
- Pasuje dokładnie do tematu (łuszczyca w core)
- Pełne spectrum wyzwań: klasyfikacja wieloklasowa + lekkie niezbalansowanie + mała próbka

---

## 🧩 Kluczowe Insights

1. **Musimy uważać na temat 6 vs 16** — HAM10000, PAD-UFES-20, ISIC to głównie czerniak → temat 16 (nowotwory skóry), **nie** nasz. Odrzucone.
2. **Tabelarny > obrazowy dla osoby 1** — instrukcja preferuje tabular, user chce clean, a UCI Dermatology to trafienie w dziesiątkę.
3. **Różnorodność dla zespołu** — miks tabular (os.1) + obrazy (os.2) + obrazy z metadata (os.3) → bogata narracja w raporcie.
4. **Literatura dla UCI Dermatology jest obfita** — Güvenir 1998 (oryginalny paper), dziesiątki nowszych (RF, SVM, XGBoost, DL benchmarki na tym zbiorze).

---

## 📝 Lista Zadań (Actionable Steps)

### 🔴 KRYTYCZNY (musi być na koniec projektu)

- [ ] **Krok 1:** Założenie struktury folderów `PROJEKT/{data,notebooks,env}` + `.gitkeep` gdzie trzeba → **Verify:** `ls PROJEKT/` pokazuje `data/ notebooks/ env/ raport.md README.md`
- [ ] **Krok 2:** Pobranie UCI Dermatology dataset automatycznie (via `ucimlrepo` lub URL → `data/uci_dermatology/`) → **Verify:** `pandas.read_csv` ładuje 366 wierszy × 35 kolumn
- [ ] **Krok 3:** Notebook `mateusz_mroz_uci_dermatology.ipynb` — sekcje:
  - [ ] 3.1 Import + load → **Verify:** `df.shape == (366, 35)`
  - [ ] 3.2 EDA (rozkład klas, missing, korelacje) → **Verify:** brak NaN po imputacji, wizualizacje czytelne
  - [ ] 3.3 Preprocessing (stratified split 80/20, StandardScaler) → **Verify:** rozkład klas zachowany w train/test
  - [ ] 3.4 Baseline (Dummy + Logistic Regression) → **Verify:** LR > Dummy na accuracy
  - [ ] 3.5 Modele (RF, SVM, XGBoost) z StratifiedKFold k=5 → **Verify:** 3 modele z mean±std metryk
  - [ ] 3.6 Strojenie (GridSearchCV na najlepszym) → **Verify:** poprawa vs default
  - [ ] 3.7 Interpretacja (feature importance + SHAP) → **Verify:** top-10 cech wyświetlone
  - [ ] 3.8 Confusion matrix + per-class metrics → **Verify:** wizualizacja + tabela
  - [ ] 3.9 Tabela porównawcza z literaturą (3 publikacje) → **Verify:** tabela w notebooku
  - [ ] 3.10 Wnioski Markdown → **Verify:** co najmniej 5 punktów (cechy, ograniczenia, ulepszenia)
- [ ] **Krok 4:** Template raportu `raport.md` wg `DOC/Projekt z uczenia maszynowego (template).txt` — sekcje: Tytuł, Autorzy, Streszczenie, Wstęp, Materiały i metody, Eksperymenty, Dyskusja, Wnioski, Bibliografia, Aneksy → **Verify:** wszystkie 9 sekcji z placeholderami lub wypełnionymi fragmentami dla osoby 1

### 🟡 WYSOKI (ważne dla jakości)

- [ ] **Krok 5:** `env/requirements.txt` z pinami wersji (sklearn, pandas, numpy, matplotlib, seaborn, shap, xgboost, jupyter, ucimlrepo) → **Verify:** `pip install -r` działa w świeżym venv
- [ ] **Krok 6:** `README.md` w `PROJEKT/` — jak uruchomić, kto co robi (osoba 1 done, 2/3 stubs), mapa plików → **Verify:** nowy członek zespołu wie co robić po 5 min czytania
- [ ] **Krok 7:** Metadata-table (wspólna, 3 datasety) w `raport.md` (Tabela 1) — nazwa/link/licencja/rozmiar/typ/cechy → **Verify:** tabela kompletna dla wszystkich 3
- [ ] **Krok 8:** Przegląd literatury dla UCI Dermatology (≥3 publikacje z metrykami) → **Verify:** w raporcie tabela "benchmarki z literatury" z 3 wpisami (autor, rok, model, metryka, wynik)

### 🟢 NORMALNY (nice-to-have, stuby dla osób 2/3)

- [ ] **Krok 9:** Stub notebook dla osoby 2 (DermNet) — z instrukcją pobrania z Kaggle + szkielet EDA/CNN transfer learning → **Verify:** markdown z krokami + linki + sugerowane publikacje
- [ ] **Krok 10:** Stub notebook dla osoby 3 (Fitzpatrick17k) — analogicznie → **Verify:** jak wyżej
- [ ] **Krok 11:** Placeholder `data/dermnet/README.md` i `data/fitzpatrick17k/README.md` z instrukcją manual download (Kaggle API / URL) → **Verify:** osoby 2/3 mogą pobrać same

---

## ⚠️ Ryzyka do Monitorowania

| Ryzyko | Trigger | Akcja |
|--------|---------|-------|
| `ucimlrepo` niedostępne | błąd importu | fallback: ręczne pobranie z `https://archive.ics.uci.edu/static/public/33/dermatology.zip` |
| Niezbalansowanie klas UCI | klasa 6 <10% | `class_weight='balanced'` + macro-F1 jako główna metryka |
| Mała próbka → wysokie std | std > 5% na CV | powtarzany k-fold (RepeatedStratifiedKFold) |
| SHAP zbyt wolny | n/a — 366 wierszy, będzie OK | brak akcji |
| Brak publikacji dla DermNet/Fitzpatrick | trudno znaleźć 3 | stuby dla os. 2/3 + sugestie w README; oni szukają sami |

---

## ❓ Otwarte Pytania (do usera — HITL)

1. **Zakres:** Robimy tylko część osoby 1 (full) + stuby dla 2/3, czy user chce też pełne notebooki dla osób 2 i 3? (Mój default: stuby, bo to oszczędza czas i respektuje "projekt grupowy").
2. **Autorzy w raporcie:** Jakie imiona wpisać dla osób 2 i 3, czy placeholder `<Imię Nazwisko 2>`?
3. **Kaggle API:** User ma skonfigurowane Kaggle API (`kaggle.json`)? Jeśli tak — mogę zautomatyzować download DermNet/Fitzpatrick. Jeśli nie — tylko instrukcja manualna.
4. **Styl komentarzy w notebooku:** User pisał "dawj kometrze co się dzieje i dalczgo dla mnie najwszj mózniej się poprawi" — czyli **dużo wyjaśnień dlaczego robimy dany krok** (edukacyjnie, jak do ściągi). ✅ Tak robię. Potwierdzenie?

---

## 🚀 Następny Krok

Po zatwierdzeniu planu przez usera (HITL) — startuję od **Kroku 1** (struktura folderów) i lecę iteracyjnie: **szukam → wybieram → uzasadniam → koduję → raportuję** — commit-less do momentu end-to-end działania notebooka.
