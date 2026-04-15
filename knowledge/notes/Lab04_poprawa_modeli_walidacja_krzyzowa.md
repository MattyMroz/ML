# Lab 4: Poprawa Modeli — Walidacja Krzyżowa

> **Tematyka:** Strategie oceny modeli przez właściwe podziały danych. Lab pokazuje, jak unikać overfittingu i przecieku informacji oraz jak dobrać poprawny schemat walidacji.
> **Notebooki:** `Poprawa modeli.ipynb`, `tools.ipynb`, `Zadanie.ipynb`
> **Kluczowe biblioteki:** scikit-learn, numpy, pandas, matplotlib

---

## TL;DR

Nie wolno trenować i testować na tych samych danych. Podstawą wiarygodnej oceny modelu jest odpowiedni podział danych: **hold-out** dla szybkich testów, **k-fold CV** dla stabilniejszej oceny, **stratyfikacja** przy niezbalansowanych klasach, **GroupKFold** dla danych grupowych i **TimeSeriesSplit** dla szeregów czasowych. Najważniejsze jest unikanie **data leakage** i raportowanie nie tylko średniej, ale też odchylenia standardowego wyniku.

---

## Spis Treści

- [Koncept 1: Overfitting i data leakage](#koncept-1-overfitting-i-data-leakage)
- [Koncept 2: Hold-Out](#koncept-2-hold-out)
- [Koncept 3: K-Fold Cross-Validation](#koncept-3-k-fold-cross-validation)
- [Koncept 4: Stratified K-Fold](#koncept-4-stratified-k-fold)
- [Koncept 5: ShuffleSplit](#koncept-5-shufflesplit)
- [Koncept 6: Podziały grupowe](#koncept-6-podziały-grupowe)
- [Koncept 7: LOO i TimeSeriesSplit](#koncept-7-loo-i-timeseriessplit)
- [Koncept 8: Jak wybrać metodę](#koncept-8-jak-wybrać-metodę)

---

## Koncept 1: Overfitting i data leakage

### Co to jest?

**Overfitting** oznacza, że model zapamiętuje dane treningowe zamiast uczyć się ogólnego wzorca. **Data leakage** pojawia się wtedy, gdy informacje ze zbioru testowego lub walidacyjnego przeciekają do procesu trenowania lub strojenia.

### Kod

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# zły schemat
model = RandomForestClassifier()
model.fit(X, y)
print(model.score(X, y))

# poprawny schemat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### Gotchas / Tips

- ⚠️ Nie skaluj i nie imputuj na całym zbiorze przed walidacją.
- 💡 Używaj `Pipeline`, żeby preprocessing działał osobno w każdym foldzie.
- 🔑 Zbiór testowy zostaw na sam koniec.

---

## Koncept 2: Hold-Out

### Co to jest?

Jednorazowy podział danych na train i test, czasem dodatkowo validation.

### Kod

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

### Gotchas / Tips

- ⚠️ Przy klasyfikacji często potrzebujesz `stratify=y`.
- 💡 Hold-out jest dobry do szybkiego baseline'u.
- 🔑 Przy małych zbiorach bywa zbyt niestabilny.

---

## Koncept 3: K-Fold Cross-Validation

### Co to jest?

Dane dzielisz na `k` foldów. Każdy fold raz jest testem, a pozostałe foldy są treningiem. Wyniki są uśredniane.

### Kod

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(scores.mean(), scores.std())
```

### Gotchas / Tips

- ⚠️ Bez `shuffle=True` możesz dostać zły podział na posortowanych danych.
- 💡 Małe odchylenie standardowe oznacza stabilniejszy model.
- 🔑 To dobry standard dla średnich zbiorów danych.

---

## Koncept 4: Stratified K-Fold

### Co to jest?

Wariant K-Fold, który utrzymuje podobne proporcje klas w każdym foldzie.

### Kod

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(scores.mean(), scores.std())
```

### Gotchas / Tips

- ⚠️ Dla klasyfikacji z niezbalansowanymi klasami to zwykle najlepszy wybór.
- 💡 Wymaga przekazania `y` przy splitowaniu.
- 🔑 To często złoty standard dla projektów klasyfikacyjnych.

---

## Koncept 5: ShuffleSplit

### Co to jest?

Losowe, wielokrotne podziały train/test. Foldy mogą się nakładać.

### Kod

```python
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=sss, scoring='accuracy')
print(scores.mean(), scores.std())
```

### Gotchas / Tips

- ⚠️ Podziały nie są rozłączne, więc interpretacja jest trochę inna niż w k-fold.
- 💡 Dobre dla dużych zbiorów i szybkich eksperymentów.
- 🔑 Przy klasyfikacji preferuj wersję stratyfikowaną.

---

## Koncept 6: Podziały grupowe

### Co to jest?

Gdy próbki należą do grup, np. kilku pomiarów jednego pacjenta, ta sama grupa nie może trafić jednocześnie do train i test.

### Kod

```python
from sklearn.model_selection import GroupKFold
import numpy as np

groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
gkf = GroupKFold(n_splits=3)

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    print(np.unique(groups[test_idx]))
```

### Gotchas / Tips

- ⚠️ Brak grupowej walidacji przy danych pacjentowych to klasyczny przeciek danych.
- 💡 Jeśli masz grupy i niezbalansowane klasy, rozważ `StratifiedGroupKFold`.
- 🔑 `n_splits` nie może być większe niż liczba grup.

---

## Koncept 7: LOO i TimeSeriesSplit

### Co to jest?

**Leave-One-Out** testuje każdą próbkę osobno. **TimeSeriesSplit** zachowuje porządek czasu i zawsze testuje na przyszłości.

### Kod

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression

model = LinearRegression()
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
print(scores.mean())

loo = LeaveOneOut()
```

### Gotchas / Tips

- ⚠️ LOO jest kosztowne obliczeniowo.
- ⚠️ Dla szeregów czasowych zwykły K-Fold jest błędem metodologicznym.
- 🔑 `TimeSeriesSplit` to obowiązek przy danych czasowych.

---

## Koncept 8: Jak wybrać metodę

### Tabela porównawcza metod

| Metoda | Losowość | Stratyfikacja | Grupy | Czas | Typowy use case |
|--------|----------|---------------|-------|------|-----------------|
| Hold-Out | tak | opcjonalnie | nie | nie | szybki baseline, duże zbiory |
| KFold | opcjonalnie | nie | nie | nie | średnie zbiory, regresja, dane zbalansowane |
| StratifiedKFold | opcjonalnie | tak | nie | nie | klasyfikacja, szczególnie przy niezbalansowanych klasach |
| ShuffleSplit | tak | nie | nie | nie | dużo danych, szybkie wielokrotne losowe podziały |
| StratifiedShuffleSplit | tak | tak | nie | nie | duże zbiory klasyfikacyjne z nierównymi klasami |
| GroupKFold | nie | nie | tak | nie | kilka próbek z tego samego pacjenta, urządzenia lub sesji |
| GroupShuffleSplit | tak | nie | tak | nie | grupy + losowość, gdy nie potrzebujesz ścisłych foldów |
| StratifiedGroupKFold | nie | tak | tak | nie | grupy i jednocześnie niezbalansowane klasy |
| Leave-One-Out | nie | nie | nie | nie | bardzo małe zbiory, gdy każda próbka jest cenna |
| TimeSeriesSplit | nie | nie | nie | tak | prognozowanie, dane uporządkowane w czasie |

### Mapa decyzji

1. Jeśli dane są szeregiem czasowym, wybierasz TimeSeriesSplit i nie tasujesz danych.
2. Jeśli próbki mają strukturę grupową, wybierasz GroupKFold albo StratifiedGroupKFold.
3. Jeśli to klasyfikacja, domyślnym wyborem jest StratifiedKFold.
4. Jeśli zbiór jest bardzo duży i liczy się szybkość, rozważ ShuffleSplit albo StratifiedShuffleSplit.
5. Jeśli zbiór jest bardzo mały, możesz rozważyć Leave-One-Out, ale świadomie akceptujesz koszt obliczeniowy.

### Jak czytać wybór w praktyce

- Hold-Out: dobry na start, ale jeden podział może dać mylący wynik.
- KFold: sensowny, gdy klasy są zbalansowane i nie ma grup ani czasu.
- StratifiedKFold: najbezpieczniejsza opcja dla większości klasyfikacji.
- GroupKFold: obowiązkowy, jeśli ten sam obiekt lub pacjent pojawia się wielokrotnie.
- TimeSeriesSplit: obowiązkowy, gdy przeszłość ma przewidywać przyszłość.

### Złote zasady

1. Nie dotykaj testu przy strojeniu hiperparametrów.
2. Klasy niezbalansowane oznaczają stratyfikację.
3. Grupy oznaczają walidację grupową.
4. Czas oznacza TimeSeriesSplit.
5. Raportuj średnią i odchylenie standardowe, nie pojedynczy wynik.
6. Jeśli preprocessing może przeciekać, zamknij go w pipeline.

---

## 📊 Przydatne do Projektu

- Do klasyfikacji medycznej: `StratifiedKFold(n_splits=5)`.
- Jeśli są próbki od tych samych pacjentów: `GroupKFold` albo `StratifiedGroupKFold`.
- Używaj `Pipeline`, żeby uniknąć leakage.
- W raporcie pokazuj metryki jako `średnia ± std`.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **Nested CV** to naturalny kolejny krok, jeśli chcesz stroić hiperparametry bardzo rzetelnie.
- **StratifiedGroupKFold** jest szczególnie ważny tam, gdzie są i grupy, i niezbalansowane klasy.
- **TimeSeriesSplit z gapem** bywa potrzebny w finansach i innych problemach z lookahead bias.

---

**Koniec notatki Lab 4: Poprawa Modeli — Walidacja Krzyżowa**
