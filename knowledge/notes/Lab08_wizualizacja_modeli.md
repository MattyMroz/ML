# Lab 8: Wizualizacja Modeli i Wyników Klasyfikacji/Regresji

> **Tematyka:** Interpretacja graficzna jakości modeli ML — krzywe uczenia, ROC, Precision-Recall, residuals, confusion matrix, kalibracja. Techniki diagnostyczne do oceny i poprawy modeli.
> **Notebooki:** `01 Wykresy - modele.ipynb`, `02 Wykresy - regresja.ipynb`, `03 Wykresy - klasyfikacja.ipynb`, `Dodatki.ipynb`, `Zadanie.ipynb`
> **Kluczowe biblioteki:** sklearn, matplotlib, seaborn

---

## TL;DR

Lab skupia się na **graficznej diagnostyce modeli** — technikach wizualizacji, które ujawniają problemy niewidoczne w samych metrykach liczbowych. Nauczysz się:

- Dobierać próg decyzyjny dla klasyfikacji binarnej (threshold tuning)
- Interpretować krzywe uczenia (underfitting vs overfitting)
- Obserwować wpływ hiperparametrów (validation curves)
- Analizować błędy regresji (residuals, rozkłady)
- Pracować z ROC, Precision-Recall, DET curves
- Kalibrować predykcje probabilistyczne

To **fundamentalne narzędzia dla projektów zaliczeniowych** — pozwalają wykazać dogłębne zrozumienie działania modeli.

---

## Spis Treści

- [Koncept 1: Próg Decyzyjny (Threshold Tuning)](#koncept-1-próg-decyzyjny-threshold-tuning)
- [Koncept 2: Krzywe Uczenia (Learning Curves)](#koncept-2-krzywe-uczenia-learning-curves)
- [Koncept 3: Krzywe Walidacyjne (Validation Curves)](#koncept-3-krzywe-walidacyjne-validation-curves)
- [Koncept 4: Metryki Regresji i Wykresy Residuów](#koncept-4-metryki-regresji-i-wykresy-residuów)
- [Koncept 5: Krzywa ROC i AUC](#koncept-5-krzywa-roc-i-auc)
- [Koncept 6: Krzywa Precision-Recall (PR)](#koncept-6-krzywa-precision-recall-pr)
- [Koncept 7: DET Curve i Detection Error Tradeoff](#koncept-7-det-curve-i-detection-error-tradeoff)
- [Koncept 8: Macierz Pomyłek (Confusion Matrix)](#koncept-8-macierz-pomyłek-confusion-matrix)

---

## Koncept 1: Próg Decyzyjny (Threshold Tuning)

### Co to jest?

Domyślnie klasyfikatory binarne przypisują obserwację do klasy pozytywnej, jeśli przewidywane prawdopodobieństwo ≥ 0.5. **Threshold tuning** zmienia ten próg — zamiast 0.5 możesz użyć 0.3, 0.7 itd. Zmiana progu wpływa na relację między recall a precision: niższy próg → więcej wykryć, ale mniej trafnych; wyższy próg → mniej alarmów, ale bardziej wiarygodne.

### Kiedy używać?

- Niezrównoważone klasy (fraud: 1% pozytywnych) — wtedy 0.5 jest suboptimalny
- Asymetryczne koszty błędów (FP kosztuje 100x więcej niż FN) — wybierasz próg aby minimalizować stratę
- Chcesz maksymalizować konkretną metrykę (F1, precision, recall)

### Kod

```python
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import TunedThresholdClassifierCV

scorer = make_scorer(f1_score, pos_label=1)
tuned_model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(), 
    scoring=scorer
)
tuned_model.fit(X_train, y_train)
best_threshold = tuned_model.best_threshold_
y_pred = (y_proba >= best_threshold).astype(int)
```

### Kluczowe parametry

| Parametr | Zastosowanie | Wnioski |
|----------|--------------|---------|
| Próg domyślny (0.5) | Zbalansowane klasy | Często suboptimalny dla nierównoważu |
| Próg poniżej 0.5 | Maksymalizacja recall | Więcej wykryć, mniej precyzji |
| Próg powyżej 0.5 | Maksymalizacja precision | Mniej alarmów, bardziej pewne |
| `TunedThresholdClassifierCV` | Automat. dobór progu | Bierze hiperparametr (`scoring`) |

### Gotchas / Tips

- ⚠️ Nigdy nie dobieraj progu na zbiorze testowym! Zawsze używaj zestawu walidacyjnego lub CV
- 💡 Wykreslisz wiele progów (np. 0-1 w 101 punktów) i zobaczysz, jak precision/recall/F1 się zmieniają
- 🔑 Optimal próg zależy od **kosztu błędu** — odwołaj się do wymagań biznesowych, nie tylko metryk

---

## Koncept 2: Krzywe Uczenia (Learning Curves)

### Co to jest?

Wykres pokazujący, jak zmienia się wydajność modelu na zbiorze treningowym i walidacyjnym w zależności od ilości danych. Pozwala:

- Wykryć **underfitting** — obie krzywe nisko, blisko siebie
- Wykryć **overfitting** — trening wysoki, walidacja niska, duża luka
- Odpowiedzieć: "Czy zwiększanie danych pomoże?"

### Kiedy używać?

- Po wstępnym treningu modelu, pytanie: "Czy to niedouczenie czy przeuczenie?"
- Przed zbieraniem więcej danych — czy to ma sens?
- Porównanie różnych modeli (np. drzewo vs SVM)

### Kod

```python
from sklearn.model_selection import LearningCurveDisplay, learning_curve
import matplotlib.pyplot as plt

train_sizes = np.linspace(0.1, 1.0, 10)
cv = StratifiedKFold(n_splits=5)

LearningCurveDisplay.from_estimator(
    estimator=LogisticRegression(max_iter=1000),
    X=X, y=y,
    train_sizes=train_sizes,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
plt.show()
```

### Interpretacja Przebiegów

| Scenariusz | Krzywa treningowa | Krzywa walidacyjna | Wniosek |
|-----------|------|------|---------|
| **Dobre dopasowanie** | Wysoka (~0.9) | Wysoka (~0.88), blisko treningu | Model działa dobrze |
| **Underfitting** | Niska (~0.6) | Niska (~0.58), blisko treningu | Model zbyt prosty |
| **Overfitting** | Bardzo wysoka (0.99) | Znacznie niższa (0.75) | Model zapamiętuje, nie uczy się |
| **Widać plateau** | Obie płaskie | Dalsze dane nie pomagają | Osiągnięto limit modelu |

### Gotchas / Tips

- ⚠️ Luka między krzywymi ≠ zawsze zły model; sam poziom walidacji je ważniejszy
- 💡 Regresja logistyczna zwykle ma małą lukę (liniowy model); DT bez ograniczeń — ogromną
- 🔑 Patrz na **rozrzut** (pas cieniowania) — szeroki = model niestabilny, wąski = stabilny

---

## Koncept 3: Krzywe Walidacyjne (Validation Curves)

### Co to jest?

Pokazuje, jak wydajność modelu zmienia się w zależności od **jednego hiperparametru** (np. `max_depth` drzewa, `C` w regresji logistycznej). Pomaga znaleźć optymalną wartość bez wyczerpującego gridu.

### Kiedy używać?

- Szybka diagnostyka: "Co się dzieje, jak zmieniam głębokość drzewa?"
- Przed pełnym GridSearchCV — żeby wiedzieć, w jakim zakresie szukać
- Pokazanie w raporcie: "Wybieram max_depth=5, bo tu się osiąga najlepszą walidację"

### Kod

```python
from sklearn.model_selection import ValidationCurveDisplay
import numpy as np

param_range = np.logspace(-4, 2, 7)  # range dla C: [0.0001, ..., 100]

ValidationCurveDisplay.from_estimator(
    estimator=LogisticRegression(max_iter=1000),
    X=X, y=y,
    param_name="C",
    param_range=param_range,
    cv=StratifiedKFold(n_splits=5),
    scoring="accuracy",
    n_jobs=-1
)
plt.xscale("log")
plt.show()
```

### Interpretacja

| Hiperparametr | Efekt (np. DT max_depth) | Trening | Walidacja | Wniosek |
|---------|---------|---------|---------|---------|
| Zbyt niska (1-2) | Model zbyt prosty | Niska | Niska | Underfitting |
| Optymalna (~5) | Model zbalansowany | Średnia-wysoka | Wysoka, blisko treningu | **Wybierz to** |
| Zbyt wysoka (50+) | Model zbyt złożony | Bardzo wysoka | Znacznie niższa | Overfitting |

### Gotchas / Tips

- ⚠️ Optymalna wartość != punkt, gdzie trening najwyższy; patrzysz na walidację
- 💡 Pas cieniowania pokazuje rozrzut — jeśli szeroki, hiperparametr jest wrażliwy
- 🔑 Najlepiej porównać kilka hiperparametrów jednocześnie (4 wykresy: LR-C, DT-depth, NB-var, SVM-gamma)

---

## Koncept 4: Metryki Regresji i Wykresy Residuów

### Co to jest?

**Metryki liczbowe**: MAE (średni błąd bezwzględny), MSE/RMSE (błąd kwadratowy), R² (proporcja wariancji wyjaśniona).

**Wykresy diagnostyczne**:
- **Rzeczywista vs Przewidywana** — czy punkty przy y=x?
- **Residuals vs Predicted** — czy residua losowo wokół 0?
- **Histogram residuów** — czy normalny rozkład?
- **Q-Q Plot** — czy residua normalne (ogony)?

### Kiedy używać?

- Chcesz zrozumieć, **gdzie model się myli**
- Liczby (MAE=5 tys.) mogą być mylące; wykres ujawnia systematyczne błędy
- Sprawdzenie założeń regresji liniowej: normalność, homoskedastyczność

### Kod

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats

residuals = y_test - y_pred
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Real vs Pred
axes[0, 0].scatter(y_pred, y_test, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--')

# Residuals vs Pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
axes[0, 1].axhline(0, color='red', linestyle='--')

# Histogram residuals
axes[1, 0].hist(residuals, bins=30, edgecolor='black')

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])

plt.tight_layout()
plt.show()
```

### Metryki Definicje

| Metrika | Wzór | Interpretacja |
|---------|------|---|
| **MAE** | $\frac{1}{n}\sum\|\text{błąd}_i\|$ | Średni błąd bezwzględny; jednostki pierwotne |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(\text{błąd}_i)^2}$ | Pierwiastek MSE; bardziej wrażliwy na outliers |
| **R²** | $1 - \frac{\sum(\text{błąd})^2}{\sum(y - \bar{y})^2}$ | Proporcja wyjaśnionej wariancji; 0-1 |
| **MedAE** | mediana($\|\text{błędy}\|$) | Mediana błędów; odporna na outliers |

### Gotchas / Tips

- ⚠️ R² może być mylny przy małych datasach; zawsze patrzysz na wykres
- 💡 Residuals vs Predicted pokazuje heteroskedastyczność (wachlarz) — metoda LS może być niestosowna
- 🔑 Jeśli Q-Q Plot jest daleko od linii przy ogonach → dane nie są normalne → modelo nie idealny

---

## Koncept 5: Krzywa ROC i AUC

### Co to jest?

**ROC** (Receiver Operating Characteristic) pokazuje zależność między:
- **TPR (True Positive Rate / Recall)** — ile % pozytywnych przypadków model wykrył
- **FPR (False Positive Rate)** — ile % negatywnych niepoprawnie zaklasyfikował

Dla każdego progu decyzyjnego (0.0 do 1.0) powstaje punkt (FPR, TPR). **AUC** = pole pod krzywą (0.5 = losowy, 1.0 = idealny).

### Kiedy używać?

- **Zawsze** w klasyfikacji binarnej
- Gdy klasy mniej-więcej zbalansowane
- Chcesz pokazać: "Sprawdzałem różne progi, model konsekwentnie dobry"

### Kod

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.show()
```

### Interpretacja

| AUC | Ocena | Czemu? |
|-----|-------|--------|
| 0.90-1.00 | Doskonały | Model doskonale rozróżnia klasy |
| 0.80-0.90 | Bardzo dobry | Praktycznie użyteczny |
| 0.70-0.80 | Dobry | Wystarczający, ale są błędy |
| 0.60-0.70 | Słaby | Model ledwie lepszy niż losowy |
| 0.50 | Losowy | Model bezużyteczny |
| < 0.50 | Gorszy niż losowy | Coś nie tak (np. labels odwrócone) |

### Gotchas / Tips

- ⚠️ ROC może być mylna przy **muy niezrównoważonych klasach** (1% pozytywnych) — użyj PR Curve
- 💡 Prógi (`thresholds`) możesz dodać do wykresu co kilka punktów
- 🔑 "Optimal próg" to punkt, gdzie FPR i TPR najbardziej Cię satysfakcjonują (zależy od kosztu)

---

## Koncept 6: Krzywa Precision-Recall (PR)

### Co to jest?

Alternatywa dla ROC, **lepsza dla niezrównoważonych klas**. Pokazuje zależność:
- **Recall (X)** — ile % pozytywnych wykrył
- **Precision (Y)** — ile % przewidzianych jako pozytywne rzeczywiście było pozytywne

Ideał: (1, 1) = 100% recall, 100% precision. **AP (Average Precision)** = pole pod krzywą PR.

### Kiedy używać?

- **Niezrównoważone klasy** (fraud 0.1%, churn 5%) — ROC może być mylna
- Ważna ci pewność (precision) lub czułość (recall)
- Raport dla biznesu: "Wykrywamy 80% oszustw, ale mamy 40% fałszywych alarmów"

### Kod

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
```

### Interpretacja

| Punkt na krzywej | Recall | Precision | Bedeut |
|--------|--------|----------|---------|
| Lewy koniec | Niska (~0.1) | Wysoka (~0.9) | Konserwatywny model, mało alarmów |
| Środek | Średnia (~0.5) | Średnia (~0.5) | Kompromis |
| Prawy koniec | Wysoka (~0.95) | Niska (~0.2) | Liberalny model, dużo alarmów |

### Gotchas / Tips

- ⚠️ PR Curve może być bardziej "zaszumiona" niż ROC — to normalne przy małych klasach
- 💡 AP jest lepszy niż AUC dla niezrównoważonych zbiorów
- 🔑 Baseline PR = udział klasy pozytywnej; jeśli precision byłaby ok 0.05 dla niezrównoważonego zbioru, Twój model powinien być znacznie wyżej

---

## Koncept 7: DET Curve i Detection Error Tradeoff

### Co to jest?

Podobnie jak ROC, ale pokazuje:
- **Oś X**: FPR (False Positive Rate)
- **Oś Y**: FNR (False Negative Rate = 1 - Recall)

Obie osie są **w skali probability** (inverse normal CDF) — robi z logarytmicznego jak liniowy. Punkt przecięcia (FPR = FNR) to **EER (Equal Error Rate)**.

### Kiedy używać?

- Gdy **niskie błędy są krytyczne** (biometryka, bezpieczeństwo, medyka)
- Chcesz pokazać: "Przy EER = 2.5%, model wykonuje dobrze w obu kierunkach"
- Lepiej rozróżnia modele przy bardzo małych błędach (< 5%)

### Kod

```python
from sklearn.metrics import DetCurveDisplay

DetCurveDisplay.from_predictions(y_test, y_proba).plot()
plt.title("DET Curve")
plt.show()
```

### Interpretacja

| Element | Znaczenie |
|---------|-----------|
| Lewy dolny róg (0, 0) | Ideał: brak błędów |
| EER (Equal Error Rate) | Punkt, gdzie FPR = FNR; ufny wynik dla obu klas |
| Bardziej zakrzywiona krzywa | Lepszy model |
| Blisko przekątnej | Model gorszej jakości |

### Gotchas / Tips

- ⚠️ DET jest mniej intuicyjna niż ROC — wymaga tłumaczenia dla niatechnicznych odbiorcy
- 💡 Używaj DET, gdy chcesz pokazać: "Model zawisze równomiernie oba błędy"
- 🔑 EER jest jedną liczbą syntetyczną, przydatną do porównania artykułów naukowych

---

## Koncept 8: Macierz Pomyłek (Confusion Matrix)

### Co to jest?

Tabelka 2×2 (dla klasyfikacji binarnej) lub N×N (dla wieloklasowej) pokazująca:

```
                Predicted Negative    Predicted Positive
Actual Negative        TN                  FP
Actual Positive        FN                  TP
```

Stąd wyliczają się wszystkie metryki: precision, recall, F1, specificity.

### Kiedy używać?

- Zawsze jako first visualization w klasyfikacji
- Wizualizuj `ConfusionMatrixDisplay` — heatmap jest czytelniejszy niż liczby
- Dla wieloklasowych: pokaż, które klasy się myją

### Kod

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg", "Pos"])
disp.plot(cmap='Blues')
plt.show()
```

### Metryki z Macierzy

| Metrika | Wzór | Co mówi |
|---------|------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Procent trafnych; nie zawsze najlepsze |
| **Precision** | TP/(TP+FP) | Ile % alarmu to prawdziwy alarm |
| **Recall** | TP/(TP+FN) | Ile % rzeczywistych pozytywnych wykrył |
| **Specificity** | TN/(TN+FP) | Ile % przypadków negatywnych prawidłowo zidentyfikowano |
| **F1** | 2×Precision×Recall / (Precision+Recall) | Harmoniczna średnia; balans |

### Gotchas / Tips

- ⚠️ Accuracy przy niezrównoważonych klasach (fraud 0.1%) może być 99% — bezużyteczne
- 💡 Zawsze patrzysz na Recall dla klasy ważnej (rare) — czy model ją wykrywa?
- 🔑 Heatmap z wartościami % zamiast surowych liczb jest bardziej readable

---

## 📊 Przydatne do Projektu

Z tego laba przyda się do projektu na zaliczenie:

**Metryki obwieszczane w raporcie**:
- ✅ **Dokładność (Accuracy)** — zawsze, nawet jeśli niezrównoważone
- ✅ **Precision & Recall** — dla każdej klasy
- ✅ **F1-score** — zwłaszcza dla niezrównoważu
- ✅ **AUC-ROC** — (+ PR-AUC, jeśli niezrównoważone)
- ✅ **Confusion Matrix** — wizualizacja w Markdown lub PNG

**Wykresy do raportu**:
- ROC Curve (zrób go, pokaż interpretację)
- Learning Curves (pokaż: czy underfitting czy overfitting?)
- Confusion Matrix jako heatmap
- Dla regresji: Real vs Predicted scatter

**K-fold Walidacja**:
- Nie wystarczy train/test split!
- Użyj `StratifiedKFold` (dla klasyfikacji) lub zwykłego `KFold` (dla regresji)
- Raportuj średnią i std dla każdej metryki
- Pokaż: `Accuracy: 0.82 ± 0.03` (5-fold CV)

**Porównanie z literaturą**:
- Jeśli używasz istniejącego datasetu (Breast Cancer, Iris, ...) — znajdź artykuł/GitHub z wynikami
- Pokaż: "Baseline z literatury: AUC=0.78, moja implementacja: AUC=0.75" — to pokazuje, że Twój model robi sens
- Jeśli gorsza: analizuj dlaczego (preprocessing, hyperparametry, regularyzacja)

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze** — traktuj jako dodatkowe źródło wiedzy, NIE jako materiał do raportu dla prowadzącego.

**🔄 Aktualność:**

Techniki z laba (ROC, PR, Learning Curves) to **fundamenty, nie mody**. Są w każdym ML pipeline'ie od 2000 roku i będą za 20 lat. Nie idą w zapomnienie.

Nowsze trendy (2024-2026):
- **Kalibrated predictions** — coraz ważniejsze (szczęśliwe dla medicinał AI)
- **Cumulative Gains / Lift charts** — powrót do biznesu (marketing, finance)
- **CAP curves** — alternatywa do ROC w sklepach (e-commerce)

**📚 Zasoby:**

1. [scikit-learn Visualizations Guide](https://scikit-learn.org/stable/visualizations.html) — kompletny reference dla `Display` klas
2. [Tamogu Schramkamp — ROC vs PR](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_%28information_retrieval%29) — czysty wytłumaczenie
3. [Hands-On ML with Scikit-Learn](https://github.com/ageron/handson-ml3) — dużo diagramów
4. Dla regresji: [RMS Normalized Bias (RMSB)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) — nowsza metrika niż stare MAE/MSE combo

**⚡ Nowsze podejścia:**

- **Conformal Prediction** — zamiast point estimates, dostajesz prediction sets (np. "wartość między 5 a 8 z 95% confidence")
- **Brier Score** — kompletnie zapomniana, ale przydatna: $\text{Brier} = \frac{1}{n}\sum(y_i - \hat{p}_i)^2$
- **Youden's Index** — `J = TPR + Specificity - 1` — jedna liczba do wyboru threshold'u (zamiast patrzcenia na ROC)

**🆚 Porównania:**

| Narzędzie | Zaleta | Wada | Kiedy |
|-----------|--------|------|-------|
| **ROC** | Intuicyjny | Mylny przy niezrównoważu | Domyślnie, klasy w miarę zbilansowane |
| **PR** | Lepszy dla niezrównoważu | Mniej intuicyjny | Fraud, Churn, Rare disease |
| **DET** | Najlepszy przy małych błędach | Trudna do wytłumaczenia | Biometryka, Security |
| **Confusion Matrix** | Pełna informacja | Nie normalizowana | Zawsze razem z ROC/PR |
| **Calibration** | Prawdziwe prawdopodobieństwa | Wymaga dużo danych | Binned forecaster (weather, prob) |

---

To powinno dać Ci pełny obraz wizualizacji. Lab jest **praktycznie ważny** — prawie każdy projekt wymaga przynajmniej ROC + CM. W swoim projekcie wykaż wszystkie 3-4 typy wykresów + 5-fold CV + porównanie z baseline'em z artykułu. To= automatycznie "bardzo dobrze" w ocenie.
