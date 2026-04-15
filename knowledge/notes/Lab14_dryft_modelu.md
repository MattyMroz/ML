# Lab 14: Dryft Modelu

> **Tematyka:** Model drift (concept drift, data drift) — przyczyny spadku jakości modelu w produkcji, metody detekcji i monitorowania, strategie naprawy. Lab uczy, dlaczego modele „starzeją się” i jak na to reagować.
> **Notebooki:** `Dryft modelu.ipynb`
> **Kluczowe biblioteki:** scikit-learn, pandas, scipy.stats, matplotlib

---

## TL;DR

Model drift to spadek wydajności (AUC, F1, RMSE) mimo niezmienionej architektury — przyczyna: zmiany w rozkładzie danych wejściowych (data drift), rozkładzie etykiet (label drift) lub relacji wejście-wyjście (concept drift). Wykrywa się go porównując metryki modelu w czasie i testami na rozkładach cech (KS, PSI, dywergencja KL). Reakcja: retraining na nowych danych, zmiana progu decyzyjnego albo przebudowa modelu. W systemach z regularnym monitoringiem dryf daje się kontrolować, ale wymaga procesów MLOps.

---

## Spis Treści

- [Koncept 1: Data Drift vs Concept Drift](#koncept-1-data-drift-vs-concept-drift)
- [Koncept 2: Szybkość i Typy Dryftu](#koncept-2-szybkość-i-typy-dryftu)
- [Koncept 3: Metody Detekcji](#koncept-3-metody-detekcji)
- [Koncept 4: Test Kołmogorowa-Smirnowa (KS)](#koncept-4-test-kołmogorowa-smirnowa-ks)
- [Koncept 5: Naprawa Modelu po Dryfie](#koncept-5-naprawa-modelu-po-dryfie)

---

## Koncept 1: Data Drift vs Concept Drift

### Co to jest?

**Dryft modelu** to obserwowalny spadek jakości modelu (metryki: AUC, F1, RMSE, accuracy) w produkcji przez wiele tygodni lub miesięcy, mimo że wagi modelu się nie zmieniają. Przyczyna: zmiana w statystycznych własnościach danych.

Rozróżniamy trzy główne typy:

1. **Data drift** (covariate shift): zmienia się rozkład cech wejściowych $P(X)$, ale relacja $P(Y|X)$ zostaje ta sama.
2. **Label drift**: zmienia się rozkład wartości wyjściowych, czyli $P(Y)$.
3. **Concept drift**: zmienia się sama relacja wejście-wyjście $P(Y|X)$, nawet jeśli rozkład cech wygląda podobnie.

### Kiedy używać?

Monitoring dryftu jest obowiązkowy dla każdego modelu w produkcji, który:
- procesuje nowe dane w regularnych cyklach,
- ma stałą architekturę i wagi,
- przewiduje zjawiska podlegające zmianom sezonowym lub strukturalnym.

### Kod

```python
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
auc_baseline = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# nowe dane
y_new_pred = model.predict_proba(X_new)[:, 1]
auc_new = roc_auc_score(y_new, y_new_pred)

if auc_baseline - auc_new > 0.05:
    print("Model drift detected")

ks_stat, ks_p = ks_2samp(X_train["feature1"], X_new["feature1"])
if ks_p < 0.05 and ks_stat > 0.1:
    print("Data drift in feature1")
```

### Kluczowe parametry

| Parametr | Co to | Wartość alarmu |
|----------|-------|----------------|
| spadek AUC/F1 | spadek jakości | >5% poniżej baseline |
| KS statystyka | różnica rozkładów cechy | >0.1 przy p<0.05 |
| PSI | dryf dla cech kategorycznych | >0.1-0.25 |
| okres monitoringu | jak często sprawdzać | tydzień/miesiąc |

### Gotchas / Tips

- ⚠️ Bez etykiet dla nowych danych możesz wykryć tylko potencjalny drift danych, nie pełny concept drift.
- 💡 Skup się na najważniejszych cechach z feature importance.
- 🔑 Przy dużych próbkach patrz nie tylko na p-value, ale też na wielkość efektu.

---

## Koncept 2: Szybkość i Typy Dryftu

### Co to jest?

Dryft może pojawiać się w różny sposób:

1. **Nagły dryft** — zmiana z dnia na dzień.
2. **Dryft stopniowy** — małe zmiany kumulują się w czasie.
3. **Dryft impulsowy** — jednorazowy skok.
4. **Sezonowość** — nawracający wzorzec, który może wyglądać jak dryft.

### Kiedy używać?

- Nagły dryft → szybka reakcja.
- Stopniowy dryft → monitoring miesięczny.
- Impulsowy → sprawdzenie pipeline'u danych.
- Sezonowość → uwzględnij w modelu lub feature engineering.

### Kod

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

auc_by_week = []
for week in df['week'].unique():
    mask = df['week'] == week
    if mask.sum() < 100:
        continue
    y_pred = model.predict_proba(X[mask])[:, 1]
    auc_w = roc_auc_score(y[mask], y_pred)
    auc_by_week.append((week, auc_w))

weeks, aucs = zip(*auc_by_week)
plt.plot(weeks, aucs, marker='o')
plt.axhline(y=baseline_auc, color='r', linestyle='--', label='Baseline')
plt.legend()
plt.show()
```

### Gotchas / Tips

- ⚠️ Zbyt małe okna czasowe dają hałaśliwe metryki.
- 💡 Sezonowość czasem rozwiążesz cechami cyklicznymi zamiast retrainingiem.
- 🔑 Retraining co tydzień bywa zbyt kosztowny.

---

## Koncept 3: Metody Detekcji

### Co to jest?

Dryft detektujemy na dwa sposoby:

**Z ground truth:**
- monitorowanie AUC, F1, precision, recall, kosztu biznesowego,
- metody sekwencyjne jak CUSUM, ADWIN, SPC.

**Bez ground truth:**
- testy na rozkładach cech: KS, PSI, dywergencja KL/Jensen-Shannon.

### Kiedy używać?

- Testowanie rozkładów to early warning system.
- Gdy etykiety pojawiają się z opóźnieniem, działasz na rozkładach.
- Gdy etykiety są szybkie, monitoruj bezpośrednio metryki.

### Kod

```python
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import numpy as np

x_train_feature = df_train['age']
x_new_feature = df_new['age']

ks_stat, p_value = ks_2samp(x_train_feature, x_new_feature)
print(f"KS = {ks_stat:.3f}, p = {p_value:.1e}")

def psi(baseline_dist, new_dist, epsilon=1e-10):
    return sum((new_dist - baseline_dist) * np.log((new_dist + epsilon) / (baseline_dist + epsilon)))

js = jensenshannon(baseline_dist, new_dist)
```

### Gotchas / Tips

- ⚠️ KS jest czuły przy dużych próbkach.
- 💡 Łącz kilka metod, nie jedną.
- 🔑 Dla kategorii używaj PSI, chi-kwadrat lub KL, nie samego KS.

---

## Koncept 4: Test Kołmogorowa-Smirnowa (KS)

### Co to jest?

Test KS porównuje dystrybuanty dwóch rozkładów i sprawdza, czy pochodzą z tego samego rozkładu. Zwraca statystykę KS i p-value.

### Kiedy używać?

- dla zmiennych ciągłych,
- gdy chcesz szybko wykryć data drift,
- gdy nie chcesz zakładać konkretnego rozkładu.

### Kod

```python
from scipy.stats import ks_2samp
import numpy as np
import matplotlib.pyplot as plt

x_train = np.random.normal(0, 1, 5000)
x_new = np.random.normal(1.5, 1, 5000)

ks_stat, p_value = ks_2samp(x_train, x_new)
print(f"KS stat = {ks_stat:.3f}, p = {p_value:.1e}")

plt.hist(x_train, bins=50, alpha=0.5, label='train', density=True)
plt.hist(x_new, bins=50, alpha=0.5, label='new', density=True)
plt.title(f"Data Drift — KS = {ks_stat:.3f}, p = {p_value:.1e}")
plt.legend()
plt.show()
```

### Kluczowe parametry

| Parametr | Znaczenie |
|----------|-----------|
| `ks_stat` | maksymalna różnica dystrybuant, >0.1 sugeruje dryft |
| `p_value` | <0.05 sugeruje istotną różnicę |

### Gotchas / Tips

- ⚠️ Przy dużych próbkach p-value prawie zawsze będzie bardzo małe.
- 💡 Jeśli interesują Cię ogony rozkładu, rozważ Anderson-Darlinga lub Wasserstein distance.
- 🔑 Przed porównaniem warto obsłużyć outliery.

---

## Koncept 5: Naprawa Modelu po Dryfie

### Co to jest?

Po wykryciu dryftu masz kilka opcji:

1. naprawić pipeline ETL,
2. zrobić retraining na nowszych danych,
3. zmienić próg decyzyjny,
4. zbudować adaptacyjną architekturę,
5. dodać nowe cechy.

### Kiedy używać?

- Retraining to standard.
- Zmiana progu to szybka łata.
- Architektura adaptacyjna sprawdza się przy sezonowości.
- Feature engineering pomaga, gdy znasz przyczynę driftu.

### Kod

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Retraining
df_retrain = df[df['date'] >= '2024-01-01']
X_rt = df_retrain[features]
y_rt = df_retrain['target']

X_rt_train, X_rt_val, y_rt_train, y_rt_val = train_test_split(
    X_rt, y_rt, test_size=0.3, random_state=42, stratify=y_rt
)

clf_new = LogisticRegression(max_iter=1000)
clf_new.fit(X_rt_train, y_rt_train)
auc_new = roc_auc_score(y_rt_val, clf_new.predict_proba(X_rt_val)[:, 1])
print(f"AUC nowego modelu: {auc_new:.3f}")

# Zmiana progu
y_scores = clf.predict_proba(X_new)[:, 1]
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_scores >= thr).astype(int)
    prec = precision_score(y_new, y_pred)
    rec = recall_score(y_new, y_pred)
    f1 = f1_score(y_new, y_pred)
    print(f"thr={thr}: P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")
```

### Gotchas / Tips

- ⚠️ Zbyt częsty retraining jest kosztowny.
- 💡 Mieszaj stare i nowe dane, żeby nie overfitować do świeżego, małego wycinka.
- 🔑 W dojrzałym MLOps dryft łączy się z testami modeli, bezpiecznym wdrożeniem i monitoringiem wyjątków.

---

## 📊 Przydatne do Projektu

Lab daje praktyczny fundament monitoringu modelu w produkcji:

1. walidacja w czasie,
2. tracking AUC/F1 na segmentach czasowych,
3. time-based split zamiast zwykłego shuffle k-fold,
4. testy driftu przed i po preprocessingu,
5. sekcja stabilności modelu w raporcie.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu dla prowadzącego.

- **Aktualność:** koncepty driftu są nadal bardzo aktualne. Popularne narzędzia to Evidently AI, NannyML, WhyLabs.
- **Trend 2025:** większy nacisk na root cause analysis, automated retraining pipelines i monitoring w systemach federacyjnych.
- **Porównanie metod:** KS to dobry baseline, ale w praktyce łączy się go z PSI, KL divergence i Wasserstein distance.

---

**Koniec notatki Lab 14: Dryft Modelu**
