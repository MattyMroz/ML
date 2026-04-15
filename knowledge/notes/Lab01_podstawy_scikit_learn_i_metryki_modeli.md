# Lab 1: Podstawy Scikit-Learn i Metryki Modeli

> **Tematyka:** Fundamenty biblioteki scikit-learn, architektura Estimator API, supervised learning (klasyfikacja/regresja) oraz unsupervised learning (PCA, K-Means), a także metryki ewaluacji dla klasyfikacji i regresji. To fundamentalny lab do każdego projektu ML.
> **Notebooki:** `Podstawowe zasady (Scikit-learn).ipynb`, `Metryki_klasyfikacja.ipynb`, `Metryki_regresja.ipynb`
> **Kluczowe biblioteki:** sklearn, numpy, pandas, matplotlib, plotly

---

## TL;DR

Scikit-learn opiera się na trzech kluczowych ideach: (1) **Estimator API** — każdy algorytm ma podobny interfejs (`fit`, `predict`, `transform`); (2) **supervised learning** — klasyfikacja i regresja; (3) **unsupervised learning** — PCA do redukcji wymiarów i K-Means do klasteryzacji. Do każdego modelu trzeba dobrać właściwą metrykę: dla klasyfikacji `accuracy`, `precision`, `recall`, `F1`, `AUC`, a dla regresji `MAE`, `RMSE`, `R²`. Zawsze rób podział train/test i cross-validation.

---

## Spis Treści

- [Koncept 1: Estimator API Scikit-learn](#koncept-1-estimator-api-scikit-learn)
- [Koncept 2: Supervised Learning — Klasyfikacja](#koncept-2-supervised-learning--klasyfikacja)
- [Koncept 3: Supervised Learning — Regresja](#koncept-3-supervised-learning--regresja)
- [Koncept 4: Unsupervised Learning — PCA](#koncept-4-unsupervised-learning--pca)
- [Koncept 5: Unsupervised Learning — K-Means](#koncept-5-unsupervised-learning--k-means)
- [Koncept 6: Walidacja Modelu](#koncept-6-walidacja-modelu)
- [Koncept 7: Metryki Klasyfikacji](#koncept-7-metryki-klasyfikacji)
- [Koncept 8: Metryki Regresji](#koncept-8-metryki-regresji)

---

## Koncept 1: Estimator API Scikit-learn

### Co to jest?

Scikit-learn standaryzuje algorytmy wokół koncepcji **Estimator** — obiektu ze wspólnym interfejsem niezależnie od algorytmu. Parametry ustawiasz przy tworzeniu obiektu, a parametry dopasowane po treningu kończą się znakiem `_`.

### Kod

```python
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

model = LinearRegression(fit_intercept=True)
X_train = [[1], [2], [3]]
y_train = [2, 4, 6]
model.fit(X_train, y_train)
y_pred = model.predict([[4]])
print(model.coef_, model.intercept_)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
proba = clf.predict_proba(X_train)
```

### Kluczowe parametry

| Parametr | Rodzaj | Co robi | Kiedy zmieniać |
|----------|--------|---------|----------------|
| `fit_intercept` | LinearRegression | czy liczyć wyraz wolny | zwykle `True` |
| `n_neighbors` | KNeighborsClassifier | liczba sąsiadów | zwykle 3-10 |
| `n_clusters` | KMeans | liczba klastrów | trzeba dobrać |
| `n_components` | PCA | liczba lub procent komponentów | np. `0.95` |

### Gotchas / Tips

- ⚠️ `X` zwykle musi być 2D.
- 💡 `fit_transform()` bywa wydajniejsze niż `fit()` + `transform()`.
- 🔑 Interfejs estimatorów jest spójny i to duża przewaga sklearn.

---

## Koncept 2: Supervised Learning — Klasyfikacja

### Co to jest?

Klasyfikacja to przewidywanie **dyskretnej etykiety** na podstawie cech. Przykłady: diagnoza tak/nie, gatunek irysa, spam/nie spam.

### Kod

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
result = knn.predict([[3, 5, 4, 2]])
print(iris.target_names[result])
proba = knn.predict_proba([[3, 5, 4, 2]])
```

### Gotchas / Tips

- ⚠️ Dla niezbalansowanych klas sama `accuracy` jest pułapką.
- 💡 Dla małych datasetów SVC często działa bardzo dobrze.
- 🔑 Zawsze oceniaj model na danych niewidzianych podczas treningu.

---

## Koncept 3: Supervised Learning — Regresja

### Co to jest?

Regresja przewiduje **ciągłą wartość liczbową**, np. cenę, temperaturę, wiek, poziom biomarkera.

### Kod

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(20) / 5

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)
y_pred = rf.predict(X)
```

### Gotchas / Tips

- ⚠️ Regresja liniowa jest wrażliwa na outliery.
- 💡 Random Forest lepiej łapie nieliniowości.
- 🔑 Drzewa nie wymagają skalowania tak bardzo jak modele liniowe.

---

## Koncept 4: Unsupervised Learning — PCA

### Co to jest?

**PCA** redukuje wymiar danych, znajdując główne składowe wyjaśniające największą wariancję.

### Kod

```python
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data

pca = PCA(n_components=0.95)
pca.fit(X)
X_reduced = pca.transform(X)

print(X.shape)
print(X_reduced.shape)
print(pca.explained_variance_ratio_)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=iris.target, cmap='tab10')
plt.show()
```

### Gotchas / Tips

- ⚠️ PCA jest wrażliwa na skalę cech.
- 💡 Często używa się `StandardScaler` przed PCA.
- 🔑 PCA pomaga w wizualizacji i kompresji, ale pogarsza interpretowalność.

---

## Koncept 5: Unsupervised Learning — K-Means

### Co to jest?

**K-Means** dzieli dane na `K` klastrów poprzez iteracyjne przypisywanie punktów do najbliższych centroidów.

### Kod

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = [[0, 0], [1, 1], [2, 2], [5, 5], [6, 6], [7, 7]]

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
labels = kmeans.predict(X)
print(labels)
print(kmeans.cluster_centers_)

plt.scatter([x[0] for x in X], [x[1] for x in X], c=labels, cmap='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=200)
plt.show()
```

### Gotchas / Tips

- ⚠️ K-Means jest czuły na skalę danych i inicjalizację.
- 💡 Dobór liczby klastrów zwykle robi się metodą elbow.
- 🔑 `k-means++` to sensowna domyślna inicjalizacja.

---

## Koncept 6: Walidacja Modelu

### Co to jest?

Walidacja sprawdza, **jak dobrze model generalizuje** na nowe dane. Standard to train/test split i cross-validation.

### Kod

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
y_pred_cv = cross_val_predict(clf, X, y, cv=cv)
cm = confusion_matrix(y, y_pred_cv)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot()
plt.show()
```

### Gotchas / Tips

- ⚠️ Nie wolno mieszać danych train i test.
- 💡 `StratifiedKFold` zwykle jest lepsze od zwykłego `KFold` dla klasyfikacji.
- 🔑 Cross-validation jest wiarygodniejsza niż pojedynczy split.

---

## Koncept 7: Metryki Klasyfikacji

### Co to jest?

Metryki klasyfikacji oceniają jakość modelu klasyfikacyjnego. Najważniejsze to `accuracy`, `precision`, `recall`, `F1` i `AUC`.

### Kod

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt

y_true = [0, 0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0]
y_scores = [0.1, 0.4, 0.8, 0.9, 0.2, 0.7]

print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

fpr, tpr, _ = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.show()
```

### Tabela metryk

| Metryka | Interpretacja | Kiedy używać |
|---------|---------------|-------------|
| `Accuracy` | procent poprawnych predykcji | dane zbalansowane |
| `Precision` | ile przewidzianych pozytywów było poprawnych | gdy koszt FP jest wysoki |
| `Recall` | ile prawdziwych pozytywów wykryto | gdy koszt FN jest wysoki |
| `F1` | kompromis precision/recall | dane niezbalansowane |
| `AUC` | jakość rankingu prawdopodobieństw | klasyfikacja binarna i niezbalansowana |

### Gotchas / Tips

- ⚠️ Sama `accuracy` nie wystarcza przy niezbalansowanych klasach.
- 💡 `Weighted avg` jest przydatne dla klasyfikacji wieloklasowej.
- 🔑 Do raportu zwykle pokazujesz kilka metryk równocześnie.

---

## Koncept 8: Metryki Regresji

### Co to jest?

Metryki regresji oceniają, jak dobrze model przewiduje wartości ciągłe. Najczęściej używa się `MAE`, `RMSE` i `R²`.

### Kod

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
max_err = max_error(y_true, y_pred)
```

### Tabela metryk

| Metryka | Interpretacja | Kiedy używać |
|---------|---------------|-------------|
| `MAE` | średni błąd w jednostkach celu | łatwa interpretacja |
| `RMSE` | silniej karze duże błędy | gdy outliery są ważne |
| `R²` | część wariancji wyjaśniona przez model | ogólna jakość dopasowania |
| `Max Error` | największy pojedynczy błąd | analiza worst-case |

### Gotchas / Tips

- ⚠️ `RMSE` jest bardziej czułe na outliery niż `MAE`.
- 💡 Ujemne `R²` oznacza model gorszy niż przewidywanie średniej.
- 🔑 Do raportu warto podawać średnią i odchylenie z CV.

---

## 📊 Przydatne do Projektu

- Klasyfikacja: `accuracy`, `precision`, `recall`, `F1`, `AUC`.
- Regresja: `MAE`, `RMSE`, `R²`.
- Walidacja: minimum 5-fold cross-validation.
- Raport: confusion matrix, classification report, porównanie kilku modeli, średnia i odchylenie standardowe metryk.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **Scikit-learn** pozostaje złotym standardem dla klasycznego ML na danych tabelarycznych.
- **Random Forest, SVM, Logistic Regression, KNN** to nadal bardzo sensowne baseline'y.
- **AutoML, XGBoost, LightGBM, Optuna** są naturalnym kolejnym krokiem po opanowaniu podstaw.

---

**Koniec notatki Lab 1: Podstawy Scikit-Learn i Metryki Modeli**
