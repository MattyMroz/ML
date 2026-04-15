# Lab 10: AutoML — Automatyczne Uczenie Maszynowe

> **Tematyka:** Automatyczne uczenie maszynowe (AutoML) — narzędzia i biblioteki do automatyzacji całego pipeline'u ML (preprocessing, dobór modelu, hyperparameter tuning, ensemble). Lab obejmuje trzy popularne frameworki: H2O AutoML, AutoGluon i MLJAR AutoML, oraz wyjaśnienie metryki logloss.
>
> **Notebooki:** `00 autoML.ipynb`, `01 autoML - H2O.ipynb`, `02 autoML - autoGluon.ipynb`, `03 MLJAR AutoML.ipynb`, `Dodatek (Logloss).ipynb`, `Zadanie.ipynb`
>
> **Kluczowe biblioteki:** h2o, autogluon, mljar-supervised, pandas, sklearn

---

## TL;DR

AutoML to zbiór narzędzi automatyzujących cały proces budowy modeli ML — od preprocessingu danych, przez selekcję których algorytmów używać i strojenia hiperparametrów, aż po tworzenie zespołów modeli (ensemble) i ocenę. Zamiast ręcznie testować dziesiątki konfiguracji, przekazujesz dane do biblioteki (np. H2O, AutoGluon, MLJAR), a ona automatycznie trenuje wielomodel i wybiera najlepszy. Idealne do szybkiego prototypowania i baseline'ów, ale też do bieżących projektów — gdzie chcemy maksymalnej dokładności bez godzin ręcznego tuningu. Lab rozpatruje trzy główne frameworki i ich unique selling points.

---

## Spis Treści

- [Koncept 1: Co to jest AutoML?](#koncept-1-co-to-jest-automl)
- [Koncept 2: H2O AutoML](#koncept-2-h2o-automl)
- [Koncept 3: AutoGluon](#koncept-3-autogluon)
- [Koncept 4: MLJAR AutoML](#koncept-4-mljar-automl)
- [Koncept 5: Metryka LogLoss](#koncept-5-metryka-logloss)
- [Koncept 6: Praktyczne porównanie i wybór](#koncept-6-praktyczne-porównanie-i-wybór)

---

## Koncept 1: Co to jest AutoML?

### Co to jest?

AutoML to zautomatyzowana procedura budowy modeli ML, która zdejmuje z analityka boleśniejsze zadania:

1. **Wstępne przetwarzanie** — imputacja braków, skalowanie, kodowanie zmiennych kategorycznych, usuwanie kolinearności
2. **Inżynieria cech** — tworzenie nowych zmiennych, redukcja wymiarowości, selekcja cech
3. **Dobór modelu** — testowanie wielu algorytmów (drzewa, SVM, sieci neuronowe, itp.)
4. **Hyperparameter tuning** — automatyczne strojenie parametrów (grid search, random search, bayesowska optymalizacja)
5. **Ensemble learning** — kombinowanie najlepszych modeli w jeden zespół
6. **Walidacja i ocena** — k-fold CV, metryki (accuracy, AUC, RMSE)

Zamiast ręcznie pisać kod dla każdego kroku, AutoML robi to za Ciebie.

### Kiedy używać?

- **Prototypowanie** — szybko chcesz zobaczyć, czy problem jest rozwiązywalny
- **Baseline** — potrzebujesz punktu odniesienia dla bardziej zaawansowanych modeli
- **Bieżące projekty** — chcesz maksymalnej dokładności bez godzin ręcznego tuningu
- **Brak eksperta ML** — zespół nie ma specjalisty, a musisz delivered model

### Kod

```python
# Minimalny working example — AutoGluon
from autogluon.tabular import TabularPredictor
import pandas as pd

# Wczytaj dane
df = pd.read_csv('dane.csv')
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# Uruchom AutoML
predictor = TabularPredictor(label='target').fit(train_data, time_limit=60)

# Ocena
performance = predictor.evaluate(test_data)

# Predykcja
y_pred = predictor.predict(test_data)
```

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `time_limit` | Limit czasu na całą procedurę (sekundy) | Zawsze — nie masz czasu? ustaw 60, masz godzinę? ustaw 3600 |
| `presets` | Gotowe profile: `"fastest"`, `"best_quality"`, `"medium_quality_faster_train"` | Zależy od Twoich priorytetów: szybkość vs dokładność |
| `eval_metric` | Metryka optymalizacji: `accuracy`, `roc_auc`, `logloss`, `f1` | Dla niezbalansowanych danych: `roc_auc` lub `f1`, dla regresji: `rmse` |
| `num_bag_folds` | Głębokość baggingu w ensemble (domyślnie ~8) | Zwiększ dla lepszej jakości, zmniejsz aby przyspieszyć |
| `num_stack_levels` | Liczba poziomów stackingu (0=wyłączony, 1+=włączony) | Wtedy gdy masz czas i chcesz squeeze'a maksymalnej jakości |

### Gotchas / Tips

- ⚠️ **Wiele bibliotek zajmuje dużo RAMu** — H2O inicjalizuje own JVM, AutoGluon ładuje CatBoost + LightGBM + Neural Networks
- 💡 **Zawsze podaj `time_limit`** — bez niego AutoML może trenować godzinami na dużych zbiorach
- 🔑 **Metryka to warto dostosować** — dla niezbalansowanych danych `accuracy` to pułapka, użyj `roc_auc` lub `f1`
- 🔑 **Ensemble > single model** — AutoML tworzy zestawy modeli, które prawie zawsze biją pojedynczych modeli
- ⚠️ **Brak Silver Bullet** — niektóre problemy wymuszą ręcznego tuningu, szczególnie custom feature engineering

---

## Koncept 2: H2O AutoML

### Co to jest?

H2O AutoML to otwartoźródłowa platforma (licencja Apache 2.0) stworzona przez H2O.ai. Specjalizuje się w **dużych zbiorach danych** (skalowanie na Hadoopie, Spark'u, Kubernetes'ie) i **wyjaśnialności modeli**. Trenuje następujące algorytmy:

- GLM (uogólniony model liniowy)
- Random Forest (DRF)
- H2O GBM (Gradient Boosting Machine)
- Deep Learning (sieci neuronowe)
- XGBoost (opcjonalnie)
- Stacked Ensemble

Następnie wybiera najlepszy model i automatycznie tworzy ensemble.

### Kiedy używać?

- **Duże zbiory danych** (miliony wierszy)
- **Produkcja w Java / Scala ecosystem**
- **Wymagana wyjaśnialność** (ważność cech, SHAP)
- **Czasy obliczeń > kilka minut** (opłaca się zainwestować w setup)

### Kod

```python
import h2o
from h2o.automl import H2OAutoML

# Inicjalizacja klastra
h2o.init()

# Wczytaj dane jako H2OFrame
df = h2o.import_file('./data.csv', sep=',')

# Przygotowanie: zmienna docelowa jako factor (dla klasyfikacji)
df['y'] = df['y'].asfactor()

# Podział
train, test = df.split_frame(ratios=[0.8], seed=42)

# Trenowanie
aml = H2OAutoML(
    max_models=25,           # ile modeli trenować
    max_runtime_secs=120,    # limit czasu
    balance_classes=True,    # zbalansuj dane niezbalansowane
    seed=42
)
aml.train(training_frame=train, y='y')

# Ocena
best_model = aml.get_best_model()
print(best_model.model_performance(test))

# Shutdown
h2o.cluster().shutdown(prompt=False)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `max_models` | 0 (wszystkie) | Liczba modeli do trenowania | Zwiększ dla większej różnorodności, zmniejsz by przyspieszyć |
| `max_runtime_secs` | (żaden limit) | Sekund na całą procedurę | **Zawsze ustaw**, inaczej mogą to być godziny |
| `balance_classes` | `False` | Czy balansować klasy w niezbalansowanych danych | Dla churn/fraud/anomalii = `True` |
| `exclude_algos` | `[]` | Algorytmy do wyłączenia, np. `["XGBoost"]` | Na Windows wyłącz XGBoost, jeśli nie działał |
| `seed` | (losowy) | Seed do reprodukowalności | Zawsze wstaw dla reproducibility |

### Gotchas / Tips

- ⚠️ **H2O wymaga Java (JVM)** — `h2o.init()` uruchamia własny serwer JVM w tle
- ⚠️ **XGBoost na Windows to problem** — H2O wymaga natywnych bibliotek Linux, Colab jest lepszy
- 💡 **Zawsze shutdown sesję** — H2O rezerwuje dużo RAMu, `h2o.cluster().shutdown()` to krytyczne
- 🔑 **H2O Flow** — dostępny interfejs webowy (localhost:54321) do eksploracji wyników
- 🔑 **Leaderboard** — przede wszystkim chcesz zobaczyć ranking modeli: `aml.leaderboard.head()`
- 💡 **Cross-validation metrics są bardziej wiarygodne** — niż metryki treningowe

---

## Koncept 3: AutoGluon

### Co to jest?

AutoGluon to biblioteka od Amazon'a (licencja Apache 2.0) zaprojektowana dla **maksymalnej prostoty** i **wieloformat danych** (tabular, obraz, tekst). W praktyce:

- Wystarczy kilka linii kodu
- Automatycznie obsługuje braki i zmienne kategoryczne
- Tworzy stacked ensemble z LightGBM, CatBoost, XGBoost, Neural Networks, Random Forest'u
- Szybko dostarcza baseline'a

### Kiedy używać?

- **Szybki prototyp** — chcesz pracować sekund, nie godziny
- **Tabular data** — dane strukturalne (CSV, SQL)
- **Brak setup'u** — nie chcesz konfigurować JVM czy starych bibliotek
- **Auto feature engineering** — AutoGluon obsługuje kategoryczne czy braki bez doinżynierii

### Kod

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Wczytaj dane
df = pd.read_csv('dane.csv')

# Konwersja: yes/no -> 0/1 (jeśli docelowa jest tekstowa)
if df['target'].dtype == 'object':
    df['target'] = df['target'].map({'no': 0, 'yes': 1})

# Podział
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# Trenowanie — dwie linie!
predictor = TabularPredictor(label='target', eval_metric='roc_auc').fit(
    train_data,
    time_limit=60,
    presets='best_quality'
)

# Ocena
predictor.evaluate(test_data)

# Predykcja
y_pred = predictor.predict(test_data)
y_proba = predictor.predict_proba(test_data)  # prawdopodobieństwa
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `label` | (required) | Nazwa kolumny docelowej | Zawsze musz być podana |
| `eval_metric` | 'log_loss' | Metryka optymalizacji: `accuracy`, `roc_auc`, `f1`, `log_loss` | Zmień dla niezbalansowanych danych: `roc_auc` |
| `time_limit` | 0 (no limit) | Sekund na trenowanie | **Zawsze ustaw**, np. 300 (5 minut) |
| `presets` | automatu | `'best_quality'`, `'high_quality_fast_inference_only_refit'`, `'medium_quality_faster_train'`, `'fastest'` | Zależy od priorytetów |
| `hyperparameters` | (auto) | Dict z konfig. modeli, np. `{'GBM': {}, 'CAT': {}}` | Dla pełnej kontroli |
| `excluded_model_types` | `[]` | Modele do wyłączenia, np. `['KNN', 'NN_TORCH']` | Wyłącz powolnych dla szybkości |
| `num_bag_folds` | (auto) | Folds w baggingu (domyślnie 8) | Zwiększ dla lepszej jakości |
| `num_stack_levels` | 1 | Poziomy stackingu | 2+ dla squeeze'a jakości |

### Gotchas / Tips

- 💡 **Rozmiar: 600 MB** — zawiera LightGBM, CatBoost, PyTorch/MXNet
- 💡 **Auto-encoding kategorycznych** — nie musisz ręcznie, AutoGluon to robi
- 💡 **WeightedEnsemble to zwycięzca** — w leaderboard'zie prawie zawsze top model to ensemble
- ⚠️ **Import_file nie istnieje** — AutoGluon pracuje z pandas DataFrame'ów, nie jak H2O
- 🔑 **Leaderboard jest bogaty** — pokaż kolumny: `['model', 'score_val', 'score_test', 'pred_time_test', 'fit_time']`
- 💡 **Feature importance via permutation** — AutoGluon pokazuje wpływ usunięcia cechy (Similar to SHAP)

---

## Koncept 4: MLJAR AutoML

### Co to jest?

MLJAR-Supervised to biblioteka (licencja MIT) zoptymalizowana pod **raportowanie** i **white-box'a** (modele zrozumiałe). Szczególnie doceniana za:

- Generowanie **szczegółowych raportów** (Markdown → można commitować do repo)
- Tryby: `"Explain"` (szybko), `"Perform"` (balans), `"Compete"` (maksimum)
- Wsparcie dla danych niezbalansowanych
- Działanie na Windows, Linux, macOS

### Kiedy używać?

- **Raportowanie** — chcesz dokumentu do przyszłego wykorzystania
- **White-box ponad czarną skrzynię** — lepiej zrozumiały model niż squeeze'a 0.1% AUC
- **Szybki eksperyment** — tryb "Explain" daje wynik w sekundach
- **Brak Linux'a** — działa na Windows, nie wymaga JVM

### Kod

```python
from supervised.automl import AutoML
import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytaj dane
df = pd.read_csv('dane.csv')
df['target'] = df['target'].map({'no': 0, 'yes': 1})

# Podział
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Trenowanie
automl = AutoML(
    mode="Compete",           # inne: "Explain", "Perform", "Optuna"
    total_time_limit=60,      # sekund
    algorithms=["Xgboost", "LightGBM", "Random Forest"],  # opcjonalnie
    eval_metric="logloss"
)
automl.fit(X_train, y_train)

# Ocena
y_pred = automl.predict(X_test)
y_proba = automl.predict_proba(X_test)[:, 1]

# Raport (zapisany do MLJAR_AutoML/)
print("Ścieżka raportów:", automl._results_path)
automl.report()
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `mode` | - | `"Explain"` (szybko), `"Perform"` (balans), `"Compete"` (maksimum), `"Optuna"` (hyperopt) | `"Explain"` do testów, `"Compete"` do serii |
| `total_time_limit` | (żaden) | Sekund na całą procedurę | **Zawsze ustaw**, np. 300 |
| `algorithms` | (auto) | Lista do użycia, np. `["Xgboost", "LightGBM"]` | Zmniejsz listy dla szybkości |
| `eval_metric` | (auto) | `accuracy`, `roc_auc`, `logloss`, `f1` | Dla niezbalansowanych: `roc_auc` lub `f1` |
| `explain_level` | 1 | 0, 1, 2 — poziom detalizacji raportów | Zwiększ dla lepszych raportów |
| `validation_strategy` | (auto) | Dict z konfig. walidacji | Zwykle default'owy jest OK |

### Gotchas / Tips

- 💡 **Raporty to killer feature** — folder `MLJAR_AutoML/` zawiera `.md` z wykresami, metrykami, konfig.
- ⚠️ **White-box > black-box dla biznesu** — czasami 2% gorszy model ale zrozumiały > 2% lepszy ale czarna skrzynią
- 💡 **Stratified split** — MLJAR automatycznie robi stratified k-fold dla niezbalansowanych danych
- 🔑 **"Explain" mode == baseline** — 30 sekund do działającego modelu, potem możesz iść w "Perform" czy "Compete"
- ⚠️ **Nie generuje piklu domyślnie** — musisz `automl.save()` jeśli chcesz model na potem
- 💡 **Optuna mode** — jeśli chcesz zaawansowanej hyperoptymalizacji (bayesowska), użyj `mode="Optuna"`

---

## Koncept 5: Metryka LogLoss

### Co to jest?

**LogLoss** (logarytmiczna strata, binary cross-entropy) to metryka do oceny **jakości predykcji probabilistycznych** w klasyfikacji. Zamiast pytać "czy trafił w klasę?", pyta "**jak pewny był i czy miał rację?**"

$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Gdzie:
- $y_i$ = prawdziwa etykieta (0 lub 1)
- $p_i$ = przewidywane prawdopodobieństwo klasy 1
- $N$ = liczba próbek

**Kluczowa zasada**: Metryka **karze nadmierną pewność**, jeśli jesteś w błędzie.

### Kiedy używać?

- **Klasyfikacja binarna** — prawie zawsze (zamiast accuracy dla niezbalansowanych)
- **Probabilistyczne predykcje** — model musi outputować prawdopodobieństwa, nie tylko klasy
- **Ranking modeli** — AutoML domyślnie optymalizuje logloss
- **Interpretacja ryzyka** — czasami chcesz "być mniej pewny" niż "trafić w klasę"

### Kod

```python
from sklearn.metrics import log_loss
import numpy as np

# Prawdziwe etykiety
y_true = [1, 0, 1, 1, 0]

# Model 1: bardzo pewny i dobrze
y_proba_1 = [0.9, 0.1, 0.8, 0.95, 0.05]  

# Model 2: pewny i źle
y_proba_2 = [0.1, 0.9, 0.2, 0.05, 0.95]  

print("Model 1 (dobrze, pewny):", log_loss(y_true, y_proba_1))  # ~0.18
print("Model 2 (źle, pewny):", log_loss(y_true, y_proba_2))     # ~4.61
```

### Interpretacja

| Wartość | Co to znaczy |
|---------|-------------|
| 0.0 | Idealna predykcja (niemożliwe w praktyce) |
| 0.1 - 0.3 | Bardzo dobra kalibracja |
| 0.3 - 0.5 | Dobra kalibracja |
| 0.5 - 1.0 | Umiarkowana |
| > 1.0 | Słaba (model gubimy klasę) |

### Gotchas / Tips

- ⚠️ **Przycina do $\epsilon$ aby uniknąć log(0)** — sklearn automatycznie robi to, ale ręcznie: `np.clip(p, 1e-15, 1-1e-15)`
- 💡 **vs Accuracy** — accuracy nie karze pewności, logloss karze. Na niezbalansowanych danych logloss jest lepszy
- 🔑 **Kalibracja model'i ważna** — dobry model z loglossu = dobrze skalibrowany (predykcje 0.7 ~ 70% pozytywów)
- 💡 **Dla multiclass**: $\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log(p_{ij})$ (softmax entrop krzyżowa)

---

## Koncept 6: Praktyczne porównanie i wybór

### Porównanie H2O vs AutoGluon vs MLJAR

| Cecha | H2O | AutoGluon | MLJAR |
|-------|-----|-----------|-------|
| **Startup time** | Wolny (JVM) | Szybki | Szybki |
| **Skalowalność** | Ogromne zbiory | Tabular do GB | Tabular do GB |
| **Wyjaśnialność** | Odliczna (SHAP) | Dobra (permutation) | Najlepsza (raport) |
| **Raportowanie** | HTML, Flow UI | Leaderboard | Markdown (git-friendly) |
| **Windows** | Ograniczenia (XGB) | Pełne | Pełne |
| **Setup** | Wymaga config JVM | Pip, zero config | Pip, zero config |
| **Czas prototypu** | +30min | 5min | 2min |
| **Jakość baseline** | Bardzo dobra | Doskonała (ensemble) | Dobra (white-box) |
| **Cena** | Open (Apache) | Open (Apache) | MIT |

### Schemat decyzyjny

```
Czy masz > 10M wierszy?
  Tak -> H2O AutoML
  Nie -> dalej

Czy chcesz raport dokumentujący model?
  Tak -> MLJAR AutoML
  Nie -> dalej

Czy chcesz najwyższej jakości baseline?
  Tak -> AutoGluon
  Nie -> MLJAR (szybciej)
```

---

## 📊 Przydatne do Projektu

Co z tego laba bezpośrednio wpasowuje się w projekt na zaliczenie:

1. **Metryki wymagane projektem** — AutoML domyślnie oblicza AUC, accuracy, F1. W raporcie pokażemy:
   - AUC (ROC-AUC dla klasyfikacji)
   - Accuracy (ogólny odsetek trafień)
   - F1-score (harmoniczna średnia precision/recall — ważna dla niezbalansowanych)
   - LogLoss (jeśli używamy probabilistyczne predykcje)

2. **Preprocessing i feature engineering** — AutoML obsługuje:
   - Imputacja braków (średnia, mediana, forward fill)
   - Skalowanie (StandardScaler na regresji)
   - Encoding kategorycznych (one-hot, label encoding)
   - Redukcja wymiarowości (opcjonalnie)

3. **K-fold walidacja** — AutoML domyślnie robi k-fold CV (zwykle 5-10 folds) i reports średnie metryki + std

4. **Porównanie modeli** — Leaderboard'y (H2O, AutoGluon) albo raporty (MLJAR) pozwalają zobaczyć ranking algorytmów

5. **Do raportu**:
   - Najlepszy model (z jakimi hiperparamami?)
   - Feature importance — które cechy miały większy wpływ?
   - Confusion matrix czy ROC curve
   - Wniosek: czy model jest użyteczny? Czy zbijać się z literaturą?

### Praktyczny workflow:

1. Wczytaj dane, konwertuj target na 0/1 (lub y/n)
2. Ustawić metryka = `roc_auc` (dla niezbalansowanych danych)
3. Ustaw `time_limit` na rozsądny (np. 300 sekund = 5 minut)
4. Uruchom AutoML: `predictor.fit(train_data, ...)`
5. Ocena: `predictor.evaluate(test_data)` → wyciągnij AUC, Accuracy, F1
6. Feature importance: zobaczysz które cechy liczyły się
7. Zapisz model: `predictor.save()` — do potem

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy na przyszłość, NIE jako materiał do raportu dla prowadzącego.

### 🔄 Aktualność

- **AutoML to nie moda — to standard** (2024-2026). Każda duża firma ML używa czegoś na kształt AutoML.
- **H2O był early player** (2015+), ale traci do AutoGluon'a i MLJAR'a na prostotę
- **AutoGluon od Amazon** — podrażnił konkurencję, ale jest najtrudniejszy do deploymentu (PyTorch, MXNet)
- **MLJAR — niedoceniany** — ma najlepsze raportowanie, ale mniej known w opensource communitycie

### ⚡ Nowsze podejścia (2025+)

- **Large Language Models do AutoML** — OpenAI API i Claude mogą generować konfiguracje AutoML (meta-learning)
- **Neural Architecture Search (NAS)** — AutoML dla sieci neuronowych (AutoKeras, NAS-FCOS)
- **Few-shot learning** — zamiast milionów danych, trenuj z 100 przykładów
- **Edge AutoML** — TinyML, quantization — AutoML dla urządzeń IoT
- **Reinforcement Learning for AutoML** — używają RL do wyboru best algorithms dynamicznie

### 🆚 Porównania

- **AutoML vs Feature Store** — AutoML mówi jak budować model, Feature Store mówi skąd brać dane
- **AutoML vs MLOps** — AutoML to trenowanie, MLOps to deployment + monitorowanie
- **AutoML vs Hyperopt** — AutoML = pełny pipeline, Hyperopt = tylko tuning parametrów
- **AutoML vs Manual Tuning** — ručna: +2-3 dni pracy, AutoML: +30 minut, ale gorsze o ~0.5-2%

### 🔐 Production considerations

Jeśli wdrażasz model z AutoML do produkcji:

1. **Model encoding** — MLJAR najlepiej (kod Exp = jasny), AutoGluon trudniej (black-box ensemble)
2. **Reprodukowanie** — zawsze zachowaj seed, konfigurację i wersje bibliotek
3. **Data drift** — model trenowany dzisiaj może być gorsze za miesiąc, monitoruj metryki live
4. **Retraining schedule** — co 2 tygodnie? czyle? zależy od branży
5. **Inference speed** — AutoGluon ensemble = wolne (~1s per sample), MLJAR/H2O → lepsze

### Złote zasady

- ✅ **Zawsze ustaw seed** — reproducibility
- ✅ **Zawsze ustaw time_limit** — inaczej ryzyko, że trenowanie potrwa noc
- ✅ **Zawsze waliduj na odrębnym zbiorze test** — nie na treningowym czy walidacyjnym
- ✅ **Nie overfit na leaderboard metrice** — czasami logloss↓ = accuracy↑ = AUC↓ (sprzeczne cele)
- ❌ **Nie deploy'uj bez baseline** — zawsze patrz: czy AutoML lepsze od prostego random forest?

---

**Koniec notatki Lab 10: AutoML**
