# Lab 7: Detekcja Anomalii

> **Tematyka:** Wykrywanie anomalii i outlierów metodami nienadzorowanymi: Isolation Forest, Local Outlier Factor i One-Class SVM.
> **Notebooki:** `Detekcja anomalii.ipynb`
> **Kluczowe biblioteki:** scikit-learn, pandas, numpy, StandardScaler

---

## TL;DR

Anomalia to obserwacja wyraźnie odbiegająca od reszty danych. Gdy nie ma etykiet, używa się metod nienadzorowanych. W labie pojawiają się trzy główne algorytmy: **Isolation Forest** jako szybki i praktyczny baseline, **LOF** jako metoda oparta o lokalną gęstość oraz **One-Class SVM** jako model uczący granicy normalności. W praktyce najważniejsze są: poprawne skalowanie danych, świadomy dobór progu i zrozumienie kompromisu między `precision` i `recall`.

---

## Spis Treści

- [Koncept 1: Co to jest anomalia](#koncept-1-co-to-jest-anomalia)
- [Koncept 2: Isolation Forest](#koncept-2-isolation-forest)
- [Koncept 3: Local Outlier Factor](#koncept-3-local-outlier-factor)
- [Koncept 4: One-Class SVM](#koncept-4-one-class-svm)
- [Koncept 5: Fraud detection i creditcard.csv](#koncept-5-fraud-detection-i-creditcardcsv)
- [Koncept 6: Metryki i trade-off precision/recall](#koncept-6-metryki-i-trade-off-precisionrecall)

---

## Koncept 1: Co to jest anomalia

### Co to jest?

Anomalia to punkt danych, który znacząco odbiega od reszty. Może oznaczać błąd pomiaru, fraud, awarię albo rzadkie, ale istotne zdarzenie.

### Gotchas / Tips

- ⚠️ Nie każdy outlier należy usuwać.
- 💡 W medycynie i fraud detection anomalie często są właśnie tym, czego szukasz.
- 🔑 Zawsze rozróżniaj błąd danych od rzadkiego zjawiska.

---

## Koncept 2: Isolation Forest

### Co to jest?

Isolation Forest izoluje punkty przez losowe podziały danych. Anomalie zwykle izolują się szybciej niż punkty normalne.

### Kod

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination='auto',
    random_state=42
)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)  # -1 anomalia, 1 normalny

y_pred_binary = (y_pred == -1).astype(int)
```

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `n_estimators` | liczba drzew | zwiększ dla dokładności |
| `contamination` | oczekiwany udział anomalii | ustaw ręcznie, jeśli go znasz |
| `max_samples` | liczba próbek na drzewo | zwykle `auto` wystarcza |

### Gotchas / Tips

- ⚠️ `contamination='auto'` nie zawsze trafia dobrze.
- 💡 To zwykle najlepszy pierwszy model do testów.
- 🔑 Nadaje się do dużych zbiorów lepiej niż One-Class SVM.

---

## Koncept 3: Local Outlier Factor

### Co to jest?

LOF wykrywa obserwacje o istotnie niższej lokalnej gęstości niż ich sąsiedzi.

### Kod

```python
from sklearn.neighbors import LocalOutlierFactor

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination='auto',
    novelty=True
)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)
lof_scores = model.negative_outlier_factor_
```

### Gotchas / Tips

- ⚠️ LOF bywa wolny dla dużych zbiorów.
- 💡 Działa dobrze, gdy różne regiony danych mają różną gęstość.
- 🔑 `n_neighbors` mocno wpływa na wynik.

---

## Koncept 4: One-Class SVM

### Co to jest?

One-Class SVM uczy się granicy opisującej dane normalne. Punkty poza tą granicą traktowane są jako anomalie. To model trenowany wyłącznie na normalnych danych, więc jakość zbioru treningowego ma tu krytyczne znaczenie.

### Kiedy używać?

- Gdy masz względnie czysty zbiór normalnych obserwacji.
- Gdy anomalie mogą mieć różne formy i nie chcesz zakładać jednego prostego wzorca odchyleń.
- Gdy pracujesz na mniejszym lub średnim zbiorze i możesz zaakceptować wyższy koszt obliczeniowy.

### Kod

```python
from sklearn.svm import OneClassSVM

model = OneClassSVM(
    kernel='rbf',
    gamma='auto',
    nu=0.05
)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)
scores = model.decision_function(X_scaled)
```

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `kernel` | kształt granicy normalności | `rbf` dla złożonych granic, `linear` dla prostszych i większych danych |
| `gamma` | lokalność wpływu punktów | dostrajaj, gdy model jest zbyt sztywny albo zbyt czuły |
| `nu` | przybliżony górny limit udziału anomalii | ustaw bliżej oczekiwanego odsetka anomalii |

### Gotchas / Tips

- ⚠️ Wymaga względnie czystych danych treningowych.
- ⚠️ Skaluje się słabo dla dużych zbiorów.
- 💡 Może dawać bardzo wysoki recall kosztem tragicznej precision.
- 🔑 W fraud detection łatwo kończy z recall bliskim 100% i precision poniżej 1%, więc bez strojenia progu bywa niepraktyczny.

---

## Koncept 5: Fraud detection i creditcard.csv

### Co to jest?

Zbiór `creditcard.csv` to klasyczny benchmark do wykrywania fraudów. Ma około 284k transakcji, 31 kolumn i ekstremalnie niski odsetek fraudów: około 0.17%.

### Dlaczego ten zbiór jest ważny?

- Pokazuje realny problem skrajnie niezbalansowanych danych.
- Dobrze nadaje się do porównania Isolation Forest, LOF i One-Class SVM na tym samym scenariuszu.
- Uczy, że dobra detekcja anomalii to nie tylko wykrycie fraudu, ale też ograniczenie liczby fałszywych alarmów.

### Kod

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Data/creditcard.csv')
df = df.sample(frac=0.05, random_state=42)

X = df.drop('Class', axis=1)
y_true = df['Class']
X_scaled = StandardScaler().fit_transform(X)

print((y_true == 1).sum())
print((y_true == 0).sum())
```

### Co wynikało z labu?

| Model | Precision | Recall | Wniosek |
|-------|-----------|--------|---------|
| Isolation Forest | ~2.9% | ~76% | sensowny baseline, ale dużo false positives |
| LOF | ~2.8% | ~76% | podobny kompromis do IF |
| One-Class SVM | ~0.5% | ~100% | łapie prawie wszystko, ale oznacza niemal wszystko jako anomalię |

### Gotchas / Tips

- ⚠️ To ekstremalnie niezbalansowany zbiór.
- 💡 Accuracy praktycznie nic tu nie mówi.
- 🔑 Taki problem wymaga bardzo świadomego doboru metryk.
- 🔑 W praktyce trzeba z góry określić akceptowalny poziom precision i pod niego stroić `contamination`, `nu` albo próg anomaly score.

---

## Koncept 6: Metryki i trade-off precision/recall

### Co to jest?

W detekcji anomalii najważniejsze są:
- `precision`,
- `recall`,
- `F1`,
- `ROC-AUC`.

### Kod

```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

precision = precision_score(y_true, y_pred_binary, zero_division=0)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
accuracy = accuracy_score(y_true, y_pred_binary)
roc_auc = roc_auc_score(y_true, y_pred_binary)

print(precision, recall, f1, accuracy, roc_auc)
```

### Gotchas / Tips

- ⚠️ Wysoki recall może oznaczać bardzo dużo fałszywych alarmów.
- ⚠️ Wysoka accuracy może być całkowicie myląca.
- 💡 Zamiast sztywnej klasyfikacji warto pracować na anomaly score i samodzielnie dobrać próg.
- 🔑 Wybór precision vs recall zależy od kosztu biznesowego błędów.

### Szybka mapa decyzji

| Sytuacja | Lepszy priorytet |
|----------|------------------|
| każdy przeoczony fraud jest bardzo kosztowny | wyższy recall |
| manualna weryfikacja jest droga i ograniczona | wyższa precision |
| chcesz jeden kompromisowy wskaźnik | F1 |
| stroisz próg i porównujesz wiele ustawień | anomaly score + ROC-AUC |

---

## 📊 Przydatne do Projektu

- Jeśli masz rzadkie przypadki, nie raportuj samej accuracy.
- Możesz użyć detekcji anomalii jako preprocessingu lub osobnego modułu ostrzegającego.
- Do raportu pokaż confusion matrix i uzasadnij dobór metryki.
- Jeśli stroisz próg, opisz kompromis precision/recall.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **Isolation Forest** to zwykle najbardziej praktyczny punkt startowy.
- **LOF** bywa lepszy przy danych o różnej lokalnej gęstości.
- **One-Class SVM** ma sens głównie przy czystym zbiorze normalnych danych i mniejszej skali.
- W praktyce dobrze działa podejście hybrydowe: anomaly score + reguły heurystyczne + model nadzorowany.

---

**Koniec notatki Lab 7: Detekcja Anomalii**
