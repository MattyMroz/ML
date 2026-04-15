# Lab 6: Dane Niezbalansowane

> **Tematyka:** Obsługa niezbalansowanych zbiorów danych w klasyfikacji: odpowiednie metryki, wagi klas, oversampling, undersampling i metody hybrydowe.
> **Notebooki:** `Dane niezbalansowane.ipynb`, `Zadanie.ipynb`
> **Kluczowe biblioteki:** sklearn, imbalanced-learn, pandas, numpy

---

## TL;DR

Gdy jedna klasa dominuje, klasyczna `accuracy` bywa myląca. Model może osiągać wysoki wynik i jednocześnie kompletnie ignorować klasę mniejszościową. Lab pokazuje trzy główne drogi: (1) zmiana metryki na `recall`, `F1`, `AUC`, `balanced accuracy`, (2) użycie `class_weight='balanced'`, (3) balansowanie danych przez **SMOTE**, oversampling, undersampling i metody hybrydowe. W praktyce warto zacząć od wag klas, a potem testować resampling.

---

## Spis Treści

- [Koncept 1: Problem danych niezbalansowanych](#koncept-1-problem-danych-niezbalansowanych)
- [Koncept 2: Metryki dla danych niezbalansowanych](#koncept-2-metryki-dla-danych-niezbalansowanych)
- [Koncept 3: Wagi klas](#koncept-3-wagi-klas)
- [Koncept 4: Oversampling i SMOTE](#koncept-4-oversampling-i-smote)
- [Koncept 5: Undersampling](#koncept-5-undersampling)
- [Koncept 6: Metody hybrydowe](#koncept-6-metody-hybrydowe)

---

## Koncept 1: Problem danych niezbalansowanych

### Co to jest?

Dane niezbalansowane to takie, gdzie liczba przykładów klas mocno się różni, np. 94% klasy 0 i 6% klasy 1. Model uczy się wtedy głównie klasy większościowej.

### Kod

```python
print(y.value_counts())

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### Gotchas / Tips

- ⚠️ Wysoka `accuracy` nie znaczy, że model jest użyteczny.
- 💡 Zawsze sprawdzaj rozkład klas na starcie.
- 🔑 Problem jest szczególnie ważny w medycynie, fraud detection i bezpieczeństwie.

---

## Koncept 2: Metryki dla danych niezbalansowanych

### Co to jest?

Lepsze od accuracy są metryki skupione na klasie mniejszościowej:
- `recall`,
- `precision`,
- `F1`,
- `AUC-ROC`,
- `balanced accuracy`,
- `G-mean`.

### Kod

```python
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    balanced_accuracy_score, roc_auc_score
)
from imblearn.metrics import geometric_mean_score

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(geometric_mean_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_score))
```

### Gotchas / Tips

- ⚠️ Sam recall też nie wystarczy, jeśli model zalewa wszystko pozytywami.
- 💡 `F1` to dobry skrótowy kompromis.
- 🔑 Dla raportu podawaj też odchylenie standardowe z CV.

---

## Koncept 3: Wagi klas

### Co to jest?

Niektóre algorytmy potrafią wewnętrznie zwiększyć wagę błędów dla klasy mniejszościowej przez parametr `class_weight`.

### Kod

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

weights = {0: 1, 1: 10}
rf = RandomForestClassifier(class_weight=weights, n_estimators=100)
rf.fit(X_train, y_train)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train, sample_weight=sample_weights)
```

### Gotchas / Tips

- 💡 To zwykle najlepszy pierwszy krok.
- ⚠️ Nie każdy model reaguje tak samo dobrze na `class_weight`.
- 🔑 Logistic Regression często zyskuje bardziej niż Random Forest.

---

## Koncept 4: Oversampling i SMOTE

### Co to jest?

Oversampling zwiększa liczbę przykładów klasy mniejszościowej:
- **Random Oversampling** powiela istniejące próbki,
- **SMOTE** tworzy syntetyczne przykłady między sąsiadami,
- istnieją też warianty jak Borderline-SMOTE i ADASYN.

### Kiedy używać?

- Gdy klasa mniejszościowa ma zbyt mało przykładów i model prawie jej nie wykrywa.
- Gdy nie chcesz tracić danych klasy większościowej.
- Gdy `class_weight='balanced'` poprawia wynik, ale nadal recall jest zbyt niski.

### Kod

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_train_smote)
```

### Kiedy wybrać którą metodę?

| Metoda | Mocna strona | Główne ryzyko |
|--------|--------------|---------------|
| Random Oversampling | bardzo prosta i szybka | łatwiejszy overfitting przez duplikację próbek |
| SMOTE | lepsza różnorodność danych niż ROS | może tworzyć sztuczne punkty w trudnych obszarach |
| Borderline-SMOTE | skupia się na granicy klas | bardziej wrażliwy na szum |
| ADASYN | generuje więcej próbek tam, gdzie problem jest trudny | może przesadzić w najbardziej zaszumionych regionach |

### Gotchas / Tips

- ⚠️ SMOTE może tworzyć nienaturalne punkty przy słabo rozdzielonych klasach.
- 💡 Random Oversampling jest prostszy, ale bardziej podatny na overfitting.
- 🔑 Resampling wykonuj tylko na zbiorze treningowym.

---

## Koncept 5: Undersampling

### Co to jest?

Undersampling usuwa część przykładów klasy większościowej. Może być losowy lub bardziej inteligentny, np. ENN, Tomek Links czy CNN.

### Kiedy używać?

- Gdy masz bardzo dużo danych i możesz pozwolić sobie na redukcję klasy większościowej.
- Gdy zależy Ci na szybkim eksperymencie albo skróceniu czasu treningu.
- Gdy podejrzewasz, że część większościowych próbek jest redundantna lub zaszumiona.

### Kod

```python
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

cnn = CondensedNearestNeighbour(random_state=42)
X_train_cnn, y_train_cnn = cnn.fit_resample(X_train, y_train)
```

### Porównanie podejść do undersamplingu

| Metoda | Zastosowanie | Ograniczenie |
|--------|--------------|--------------|
| Random Undersampling | bardzo szybki baseline | losowo wyrzuca potencjalnie cenne dane |
| Tomek Links | czyści granicę między klasami | nie zawsze wystarcza jako jedyna metoda |
| ENN | usuwa lokalny szum | może być zbyt agresywny |
| CNN | zostawia tylko niezbędne przykłady większości | potrafi drastycznie zmniejszyć zbiór |

### Gotchas / Tips

- ⚠️ Usuwasz realne dane, więc łatwo stracić ważną informację.
- 💡 Ma sens przy bardzo dużych zbiorach.
- 🔑 Czysty undersampling często przegrywa ze SMOTE lub metodami hybrydowymi.

---

## Koncept 6: Metody hybrydowe

### Co to jest?

Łączą oversampling i czyszczenie danych:
- **SMOTEENN**,
- **SMOTETomek**.

### Kiedy używać?

- Gdy chcesz poprawić recall, ale bez surowego duplikowania danych.
- Gdy czysty SMOTE daje zbyt dużo fałszywych alarmów.
- Gdy potrzebujesz kompromisu między lepszą reprezentacją klasy mniejszościowej a czyszczeniem granicy decyzyjnej.

### Kod

```python
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import LogisticRegression

smoteenn = SMOTEENN(random_state=42)
X_train_hybrid, y_train_hybrid = smoteenn.fit_resample(X_train, y_train)

smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_hybrid, y_train_hybrid)
```

### Jak wybrać hybrydę?

| Metoda | Co robi najlepiej | Kiedy po nią sięgać |
|--------|-------------------|---------------------|
| SMOTEENN | mocno czyści szum po wygenerowaniu syntetycznych próbek | gdy zależy Ci na recall i bardziej agresywnym czyszczeniu |
| SMOTETomek | delikatniej czyści granicę klas | gdy chcesz kompromis między stabilnością a agresywnością |

### Gotchas / Tips

- 💡 To często najbardziej praktyczny kompromis.
- 🔑 SMOTEENN bywa bardzo mocne, gdy zależy Ci na recall.
- ⚠️ Tak samo jak wcześniej: tylko na trainie, nigdy przed podziałem danych.

### Szybka rekomendacja praktyczna

1. Zacznij od `class_weight='balanced'`.
2. Jeśli wynik dalej jest słaby, porównaj SMOTE i SMOTEENN.
3. Jeśli zbiór jest ogromny, rozważ lekkie undersampling lub Tomek Links.
4. Na końcu porównaj metryki na tym samym schemacie CV.

---

## 📊 Przydatne do Projektu

- Na początku sprawdź `value_counts()` dla targetu.
- Jeśli klasy są niezbalansowane, nie raportuj samej accuracy.
- Zacznij od `class_weight='balanced'`.
- Potem porównaj to z podejściem SMOTE lub SMOTEENN.
- Walidację rób przez `StratifiedKFold`, a resampling tylko wewnątrz foldów treningowych.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **SMOTE** nadal jest standardem, ale metody hybrydowe często wypadają lepiej.
- **Cost-sensitive learning** i **threshold tuning** to bardzo sensowne alternatywy lub uzupełnienia.
- Największy błąd praktyczny to robienie resamplingu przed cross-validation.

---

**Koniec notatki Lab 6: Dane Niezbalansowane**
