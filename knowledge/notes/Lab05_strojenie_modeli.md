# Lab 5: Strojenie Modeli

> **Tematyka:** Systematyczne strojenie hiperparametrów modeli ML przez Grid Search, Random Search, podejście bayesowskie i halving search.
> **Notebooki:** `Strojenie modeli.ipynb`, `Zadania.ipynb`
> **Kluczowe biblioteki:** scikit-learn, skopt, pandas

---

## TL;DR

Po wyborze modelu trzeba dobrać jego hiperparametry. Lab pokazuje, jak zdefiniować **przestrzeń parametrów** i przeszukiwać ją metodami: **GridSearchCV**, **RandomizedSearchCV**, **BayesSearchCV** i **HalvingGridSearchCV**. W praktyce najczęściej zaczyna się od prostego baseline'u, potem robi szybki random search, a dopiero później dokładniejszy tuning. Wszystko powinno być spięte z cross-validation i właściwą metryką.

---

## Spis Treści

- [Koncept 1: Dobór estymatora](#koncept-1-dobór-estymatora)
- [Koncept 2: Hiperparametry i przestrzeń parametrów](#koncept-2-hiperparametry-i-przestrzeń-parametrów)
- [Koncept 3: GridSearchCV](#koncept-3-gridsearchcv)
- [Koncept 4: RandomizedSearchCV](#koncept-4-randomizedsearchcv)
- [Koncept 5: BayesSearchCV](#koncept-5-bayessearchcv)
- [Koncept 6: HalvingGridSearchCV](#koncept-6-halvinggridsearchcv)
- [Koncept 7: Pipeline i strojenie preprocessingu](#koncept-7-pipeline-i-strojenie-preprocessingu)

---

## Koncept 1: Dobór estymatora

### Co to jest?

Zanim stroisz parametry, musisz wybrać model pasujący do problemu: liniowy, drzewiasty, kernelowy, boostingowy.

### Kod

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model_lr = LogisticRegression(max_iter=5000)
model_lr.fit(X_train, y_train)
print(f1_score(y_test, model_lr.predict(X_test)))

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
print(f1_score(y_test, model_rf.predict(X_test)))
```

### Gotchas / Tips

- ⚠️ Metryka musi pasować do problemu, np. `F1` albo `ROC-AUC` dla danych niezbalansowanych.
- 💡 Zanim stroisz, porównaj kilka prostych modeli bazowych.
- 🔑 Niektóre modele wymagają skalowania, inne nie.

---

## Koncept 2: Hiperparametry i przestrzeń parametrów

### Co to jest?

Hiperparametry ustawia się przed treningiem i nie są uczone z danych. Trzeba zdefiniować sensowny zakres wartości do testów.

### Kod

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

### Gotchas / Tips

- ⚠️ Liczba kombinacji szybko rośnie wykładniczo.
- 💡 Przestrzeń parametrów powinna być sensowna domenowo.
- 🔑 Nie stroisz na zbiorze testowym.

---

## Koncept 3: GridSearchCV

### Co to jest?

Testuje wszystkie kombinacje wartości z siatki parametrów.

### Kiedy używać?

- mała przestrzeń parametrów,
- chcesz pełne, dokładne przeszukanie,
- liczba kombinacji jest rozsądna.

### Gotchas / Tips

- ⚠️ Przy dużej siatce robi się bardzo wolne.
- 💡 Dobry jako etap końcowy po zawężeniu przestrzeni.
- 🔑 `best_estimator_` jest gotowy do użycia po dopasowaniu.

---

## Koncept 4: RandomizedSearchCV

### Co to jest?

Losuje ograniczoną liczbę kombinacji z rozkładów parametrów zamiast testować wszystko.

### Kod

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 10),
    'class_weight': [None, 'balanced']
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

### Gotchas / Tips

- 💡 To zwykle najlepszy pierwszy krok dla dużych przestrzeni parametrów.
- 🔑 Dobrze działa, gdy nie wiesz jeszcze, gdzie są dobre regiony przestrzeni.

---

## Koncept 5: BayesSearchCV

### Co to jest?

Podejście bayesowskie uczy się na wcześniejszych próbach i kieruje kolejne testy w bardziej obiecujące obszary przestrzeni parametrów.

### Kiedy używać?

- Gdy przestrzeń parametrów jest duża i nie chcesz testować wszystkiego ręcznie.
- Gdy pojedyncze dopasowanie modelu jest kosztowne i każda iteracja ma znaczenie.
- Gdy chcesz lepszej jakości niż prosty random search przy podobnym budżecie iteracji.

### Kod

```python
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 15),
    'min_samples_split': Integer(2, 10),
    'class_weight': Categorical([None, 'balanced'])
}

opt = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=25,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
opt.fit(X_train, y_train)
```

### Gotchas / Tips

- ⚠️ Wymaga dodatkowej biblioteki `scikit-optimize`.
- 💡 Często znajduje dobre parametry szybciej niż random search.
- 🔑 Opłacalne przy droższych modelach i dużych przestrzeniach.

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `search_spaces` | definiuje zakresy i typy parametrów | zawsze ustawiasz jawnie |
| `n_iter` | liczba iteracji optymalizacji | zwiększ, gdy model jest tani lub przestrzeń szeroka |
| `cv` | schemat walidacji | daj `StratifiedKFold` dla klasyfikacji |
| `random_state` | reproducibility | ustawiaj zawsze |

### Praktyczna różnica względem random search

- RandomizedSearchCV losuje kombinacje niezależnie od wcześniejszych wyników.
- BayesSearchCV wykorzystuje poprzednie pomiary i eksploruje lepsze regiony przestrzeni.
- W praktyce BayesSearchCV częściej daje lepszy wynik przy małym budżecie iteracji.

---

## Koncept 6: HalvingGridSearchCV

### Co to jest?

Adaptacyjna wersja grid search: szybko odrzuca słabszych kandydatów i przeznacza więcej zasobów na lepszych.

### Kiedy używać?

- Gdy masz średnio dużą siatkę parametrów i chcesz przyspieszyć klasyczny grid search.
- Gdy dysponujesz większym zbiorem danych i możesz testować kandydatów na mniejszych podpróbkach.
- Gdy zależy Ci na kompromisie między dokładnością GridSearchCV a szybkością RandomizedSearchCV.

### Kod

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

halving = HalvingGridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    factor=2
)
halving.fit(X_train, y_train)
```

### Gotchas / Tips

- ⚠️ To API eksperymentalne w sklearn.
- 💡 Dobre, gdy chcesz przyspieszyć klasyczny grid search.
- 🔑 Wczesne odrzucanie kandydatów bywa bardzo opłacalne.

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `factor` | określa tempo odrzucania kandydatów | mniejszy factor = ostrożniejsze odrzucanie |
| `resource` | mówi, co rośnie w kolejnych rundach | zwykle liczba próbek jest OK |
| `min_resources` | minimalny zasób w pierwszej rundzie | podnieś, jeśli pierwsza runda jest zbyt szumna |
| `aggressive_elimination` | mocniejsze cięcie kandydatów | tylko gdy bardzo zależy Ci na czasie |

### Intuicja działania

1. Na początku wiele konfiguracji testowanych jest na małym zasobie.
2. Słabi kandydaci są odrzucani.
3. Lepsi kandydaci dostają więcej danych lub iteracji.
4. Na końcu zostaje mała liczba najmocniejszych konfiguracji.

---

## Koncept 7: Pipeline i strojenie preprocessingu

### Co to jest?

Pipeline pozwala stroić jednocześnie preprocessing i model, bez ryzyka leakage.

### Kiedy używać?

- Gdy model wymaga skalowania lub imputacji.
- Gdy chcesz stroić preprocessing i model w jednym przebiegu.
- Gdy zależy Ci na metodologicznie poprawnej walidacji bez przecieku informacji.

### Kod

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10]
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
```

### Gotchas / Tips

- ⚠️ Nazwy parametrów muszą mieć format `krok__parametr`.
- 💡 Pipeline jest najbezpieczniejszym sposobem łączenia preprocessingu z CV.
- 🔑 Jeśli model nie potrzebuje skalowania, pipeline nadal bywa dobry dla spójności eksperymentu.

### Najważniejsze korzyści

- Ten sam preprocessing jest stosowany w każdym foldzie poprawnie.
- Łatwo porówniać różne modele w jednej strukturze.
- Zmniejszasz ryzyko ręcznych błędów przy eksperymentach.

- ⚠️ Nazwy parametrów w pipeline wymagają formatu `krok__parametr`.
- 💡 To najlepszy sposób, żeby tuning preprocessingu był metodologicznie poprawny.
- 🔑 Dzięki pipeline preprocessing dopasowuje się osobno w każdym foldzie.

---

## 📊 Przydatne do Projektu

- Porównaj kilka modeli bazowych.
- Wybierz metrykę zgodną z problemem.
- Zrób tuning najlepszego modelu na CV.
- Raportuj najlepsze hiperparametry i wynik `średnia ± std`.
- Jeśli preprocessing jest potrzebny, strojenie rób przez `Pipeline`.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **GridSearchCV** i **RandomizedSearchCV** pozostają najbardziej praktycznymi narzędziami w klasycznym ML.
- **BayesSearchCV** i **Optuna** to dobry kolejny krok, gdy budżet obliczeniowy jest ograniczony.
- **Threshold tuning** po strojeniu modelu często daje dodatkowy zysk w klasyfikacji niezbalansowanej.

---

**Koniec notatki Lab 5: Strojenie Modeli**
