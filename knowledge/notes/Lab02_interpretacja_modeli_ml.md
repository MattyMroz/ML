# Lab 2: Interpretacja i Wyjaśnialność Modeli ML

> **Tematyka:** Zrozumienie i wyjaśnianie decyzji algorytmów ML: feature importance, współczynniki regresji, SHAP i LIME.
> **Notebooki:** `Interpretacja modeli.ipynb`
> **Kluczowe biblioteki:** sklearn, shap, lime, statsmodels, xgboost

---

## TL;DR

Lab skupia się na trzech poziomach interpretacji modeli: (1) **tradycyjne metody** — feature importance i współczynniki regresji, (2) **SHAP** — wartości Shapleya do globalnej i lokalnej interpretacji, (3) **LIME** — lokalne wyjaśnienia dla pojedynczych obserwacji. Do raportów projektowych: normalizuj cechy przed porównywaniem współczynników, sprawdzaj istotność statystyczną i używaj SHAP dla modeli blackbox.

---

## Spis Treści

- [Koncept 1: Feature Importance](#koncept-1-feature-importance)
- [Koncept 2: Współczynniki Regresji Liniowej](#koncept-2-współczynniki-regresji-liniowej)
- [Koncept 3: Regresja Logistyczna i Odds Ratios](#koncept-3-regresja-logistyczna-i-odds-ratios)
- [Koncept 4: SHAP](#koncept-4-shap)
- [Koncept 5: LIME](#koncept-5-lime)

---

## Koncept 1: Feature Importance

### Co to jest?

Feature importance w modelach drzewiastych mierzy, jak mocno dana cecha wpływa na podziały w drzewach i końcową predykcję. To szybka metoda na uzyskanie globalnego rankingu istotności cech.

### Kiedy używać?

- Gdy pracujesz z modelami drzewiastymi i chcesz szybkiego globalnego rankingu cech.
- Gdy potrzebujesz pierwszej diagnozy przed głębszą analizą SHAP lub manualną walidacją.
- Gdy chcesz sprawdzić, czy model podkreśla cechy zgodne z intuicją domenową.

### Kod

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.show()
```

### Gotchas / Tips

- ⚠️ Feature importance nie oznacza kauzalności.
- ⚠️ Cechy o dużej liczbie unikalnych wartości mogą być sztucznie premiowane.
- 💡 To dobra pierwsza diagnoza przed głębszą analizą.
- 🔑 Jeśli ranking cech mocno kłóci się z wiedzą domenową, traktuj to jako sygnał do dalszej kontroli danych i modelu.

---

## Koncept 2: Współczynniki Regresji Liniowej

### Co to jest?

Współczynniki regresji liniowej pokazują, o ile zmieni się zmienna docelowa przy zmianie danej cechy o 1 jednostkę, przy pozostałych cechach stałych.

### Kod

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', ascending=False)
```

### Gotchas / Tips

- ⚠️ Bez standaryzacji porównywanie współczynników bywa bez sensu.
- ⚠️ Współliniowość destabilizuje interpretację.
- 💡 Do p-value i testów istotności użyj `statsmodels`, nie samego sklearn.

---

## Koncept 3: Regresja Logistyczna i Odds Ratios

### Co to jest?

Regresja logistyczna modeluje log-szanse zajścia zdarzenia. Współczynniki można przekształcić do ilorazu szans przez `exp(β)`.

### Kod

```python
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

X_train_sm = sm.add_constant(X_train)
logit = sm.Logit(y_train, X_train_sm).fit()

results_df = pd.DataFrame({
    'Feature': logit.params.index,
    'Coefficient': logit.params.values,
    'Odds_Ratio': np.exp(logit.params.values),
    'p_value': logit.pvalues.values
}).sort_values('p_value')
```

### Gotchas / Tips

- ⚠️ OR zależy od jednostki cechy.
- 💡 W medycynie warto raportować także przedziały ufności OR.
- 🔑 Obok interpretacji sprawdzaj też AUC i F1.

---

## Koncept 4: SHAP

### Co to jest?

SHAP wykorzystuje wartości Shapleya z teorii gier do oszacowania wkładu każdej cechy do predykcji modelu. Działa zarówno globalnie, jak i lokalnie.

### Kiedy używać?

- Gdy chcesz jednocześnie globalnej i lokalnej interpretacji.
- Gdy model jest blackboxem, ale nadal chcesz wyjaśnić konkretne predykcje.
- Gdy przygotowujesz raport dla projektu i potrzebujesz czytelnych wykresów dla interesariuszy.

### Kod

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type='bar')
shap.summary_plot(shap_values, X_test, plot_type='dot')
```

### Gotchas / Tips

- ⚠️ `KernelExplainer` jest wolny dla dużych danych.
- 💡 SHAP potrafi pokazać zarówno siłę, jak i kierunek wpływu cechy.
- 🔑 Dla modeli drzewiastych `TreeExplainer` jest naturalnym wyborem.

### Najważniejsze wykresy

- `summary_plot(..., plot_type='bar')` daje globalny ranking średniego wpływu cech.
- `summary_plot(..., plot_type='dot')` pokazuje zarówno siłę, jak i kierunek wpływu.
- `dependence_plot(...)` pomaga zobaczyć relację między wartością cechy a jej wpływem na predykcję.
- `force_plot` lub `waterfall` najlepiej nadają się do analizy pojedynczego przypadku.

---

## Koncept 5: LIME

### Co to jest?

LIME buduje lokalny, prosty model wokół jednej obserwacji i na tej podstawie wyjaśnia decyzję złożonego modelu.

### Kiedy używać?

- Gdy chcesz wyjaśnić jedną konkretną decyzję modelu.
- Gdy potrzebujesz szybkiej, lekkiej interpretacji lokalnej dla pojedynczego case'u.
- Gdy model nie jest drzewem, a pełny SHAP byłby zbyt ciężki obliczeniowo.

### Kod

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=list(X_train.columns),
    class_names=['target'],
    mode='regression'
)

exp = explainer.explain_instance(
    X_test.iloc[0],
    model.predict,
    num_features=5
)
print(exp.as_list())
```

### Gotchas / Tips

- ⚠️ LIME jest lokalny, więc nie daje pełnego obrazu modelu.
- ⚠️ Wynik może się nieco zmieniać między uruchomieniami.
- 💡 Jest szybszy niż SHAP dla pojedynczego przypadku.

### SHAP vs LIME w praktyce

- SHAP jest lepszy, gdy chcesz spójną analizę dla wielu obserwacji i ranking globalny.
- LIME jest wygodny, gdy analizujesz pojedynczy przypadek i zależy Ci na szybkości.
- W raportach projektowych najczęściej wygrywa SHAP, a LIME jest sensownym dodatkiem do lokalnych przykładów.

---

## Jak dobrać metodę interpretacji?

| Metoda | Najlepsza do | Mocna strona | Ograniczenie |
|--------|--------------|--------------|--------------|
| Feature Importance | szybki globalny screening | bardzo szybka i prosta | tylko wybrane modele, podatna na bias |
| Współczynniki regresji | modele liniowe | czytelna interpretacja znaku i siły wpływu | wymaga standaryzacji i kontroli współliniowości |
| SHAP | globalna + lokalna interpretacja blackboxa | najbardziej uniwersalne i bogate wizualnie | bywa kosztowny obliczeniowo |
| LIME | pojedyncze case studies | szybka lokalna eksplanacja | brak perspektywy globalnej, pewna niestabilność |

### Prosta reguła wyboru

1. Zacznij od feature importance albo współczynników, jeśli model to umożliwia.
2. Jeśli potrzebujesz pełniejszego obrazu, przejdź do SHAP.
3. Jeśli analizujesz pojedynczą nietypową obserwację, dołóż LIME albo lokalny SHAP.

---

## 📊 Przydatne do Projektu

- Pokaż ranking cech przez feature importance lub SHAP.
- Dla modeli liniowych raportuj współczynniki po standaryzacji.
- Dla modeli binarnych możesz dodać odds ratios.
- Dla 2-3 wybranych obserwacji przygotuj lokalne wyjaśnienia SHAP lub LIME.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **SHAP** pozostaje bardzo mocnym standardem interpretowalności w 2025.
- **LIME** nadal ma sens dla lokalnych wyjaśnień, ale częściej przegrywa z SHAP przy pełniejszych analizach.
- **Counterfactual explanations** i **Integrated Gradients** to naturalne dalsze kroki po tym labie.

---

**Koniec notatki Lab 2: Interpretacja i Wyjaśnialność Modeli ML**
