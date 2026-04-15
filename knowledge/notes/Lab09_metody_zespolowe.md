# Lab 9: Metody Zespołowe (Ensemble Methods)

> **Tematyka:** Metody zespołowe to kluczowa grupa algorytmów, które łączą wiele modeli bazowych w celu poprawy dokładności predykcji i redukcji nadmiernego dopasowania. Lab obejmuje Bagging, Boosting, Stacking oraz ich praktyczne zastosowania w klasyfikacji i regresji.
> **Notebooki:** `Metody zespołowe.ipynb`, `Zadanie.ipynb`
> **Kluczowe biblioteki:** sklearn.ensemble, pandas, xgboost, scikit-learn

---

## TL;DR

Zamiast polegać na jednym modelu, metody zespołowe tworzą "zespół" różnych klasyfikatorów/regresorów, których zbiorowa decyzja jest zwykle bardziej dokładna i odporna na overfitting. Lab pokrywa trzy główne podejścia: **Bagging** redukuje wariancję poprzez losowe próbkowanie, **Boosting** redukuje bias poprzez sekwencyjne uczenie na błędach, a **Stacking** łączy predykcje różnych modeli za pomocą meta-modelu. Każda metoda ma inne zastosowania zależnie od problemu i wymaganego kompromisu bias–wariancja.

---

## Spis Treści

- [Koncept 1: Bagging (Bootstrap Aggregating)](#koncept-1-bagging-bootstrap-aggregating)
- [Koncept 2: Boosting](#koncept-2-boosting)
- [Koncept 3: Stacking (Stacked Generalization)](#koncept-3-stacking-stacked-generalization)
- [Koncept 4: Bias–Wariancja Trade-off](#koncept-4-bias--wariancja-trade-off)
- [📊 Przydatne do Projektu](#-przydatne-do-projektu)
- [💡 Dodatkowe — Komentarze Agenta](#-dodatkowe--komentarze-agenta)

---

## Koncept 1: Bagging (Bootstrap Aggregating)

### Co to jest?

Bagging to technika polegająca na treningowaniu wielu modeli tego samego typu na różnych losowych podzbiorach danych (próbkach wygenerowanych z powtórzeniami — bootstrap). Każdy model jest trenowany niezależnie na innym zestawie danych, a następnie predykcje wszystkich modeli są agregowane (średnia dla regresji, głosowanie większościowe dla klasyfikacji). Główna idea: różne modele uczą się na nieco innych danych, stąd mają różne błędy — ich uśrednienie zmniejsza losowe fluktuacje (wariancję).

### Kiedy używać?

Bagging sprawdza się najlepiej gdy:
- Model bazowy ma **wysoką wariancję** (np. niestabilne drzewo decyzyjne),
- Chcesz **zmniejszyć overfitting** bez zmiany bias,
- Masz problemy gdzie **losowość** w danych jest źródłem problemów (brakujące wartości, szum),
- Chcesz **szybko** poprawić jakość bez głębokich zmian architekturalnych.

### Kod

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Klasyfikacja
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,          # liczba modeli bazowych
    max_samples=0.8,          # % próbek do każdego modelu
    max_features=0.8,         # % cech do każdego modelu
    bootstrap=True,           # czy losować z powtórzeniami
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Dla regresji: BaggingRegressor – identyczna API
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `n_estimators` | 10 | liczba modeli bazowych w zespole | zwiększ do 50-100 dla większej stabilności |
| `estimator` | DecisionTreeClassifier | model bazowy | zmień na inny klasyfikator (SVM, KNN) |
| `max_samples` | 1.0 | % lub liczba próbek do każdego modelu | zmniejsz (0.5-0.8) by jeszcze bardziej zdywersyfikować |
| `max_features` | 1.0 | % lub liczba cech do każdego modelu | zmniejsz by limitować liczę cech |
| `bootstrap` | True | czy losować z powtórzeniami | zmień na False dla losowania bez powtórzeń |

### Gotchas / Tips

- ⚠️ Bagging zwiększa **czas treningu** – musisz wytrenować N modeli zamiast 1. Kompensuje to równoległość – większość implementacji ma `n_jobs=-1`.
- 💡 **Random Forest** to specjalny przypadek Baggingu dla drzew decyzyjnych, ale dodatkowo losuje cechy na każdym podziale (split) – jeszcze bardziej redukuje wariancję.
- 🔑 Bagging działa najlepiej gdy model bazowy ma **wysoką wariancję i niski bias** (np. niestabilne, głębokie drzewo). Na modelu już stabilnym (np. regresja liniowa) efekt jest marginalny.
- 💡 Use `max_samples < 1.0` i `max_features < 1.0` aby jeszcze bardziej zdywersyfikować modele.

### Random Forest i Extra Trees (Extreme Randomized Trees)

**Random Forest**:
- Bagging specjalnie dla drzew decyzyjnych.
- Dodatkowo losuje cechy na każdym podziale (`max_features='sqrt'` lub `'log2'`).
- Bardzo popularne, łatwe do tuningu, dobre wyniki na większości problemów.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,           # pełna głębokość
    max_features='sqrt',      # losuj sqrt(n_features) cech
    random_state=42,
    n_jobs=-1                 # użyj wszystkich rdzeni
)
```

**Extra Trees**:
- Jak Random Forest, ale losuje też **progi podziału** (thresholdy).
- Szybsze na dużych danych, ale mniej stabilne – wymaga więcej `n_estimators`.
- Lepsze gdy masz dużo cech lub szukasz szybszych wyników.

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

---

## Koncept 2: Boosting

### Co to jest?

Boosting to rodzina metod, gdzie modele są trenowane **sekwencyjnie** – każdy kolejny model uczy się poprawiać błędy poprzednich. W przeciwieństwie do Baggingu, gdzie wszystkie modele są niezależne, w Boostingu każdy model zwraca uwagę na próby, które poprzedni model źle sklasyfikował. Uzyskuje się to poprzez:
- przypisanie większych wag (weights) błędnie sklasyfikowanym przykładom,
- trenowanie kolejnego modelu z naciskiem na te trudne przypadki,
- ważone agregowanie wyników (zwykle suma ważona zamiast średnia).

Główna idea: sekwencyjnie poprawiamy słabe punkty modelu → systematyczne zmniejszenie bias (i czasem wariancji).

### Kiedy używać?

Boosting sprawdza się gdy:
- Chcesz **zmniejszyć bias** – model bazowy jest zbyt prosty (underfitting),
- Masz problemy z **trudnymi do nauczenia przypadkami** – model regularnie myli się w ten sam sposób,
- Jesteś gotowy na **dłuższy czas treningu** (szeregowe trenowanie modeli),
- Chcesz **maksymalną dokładność** – boosting zwykle daje najlepsze wyniki, ale jest bardziej podatny na overfitting.

### Kod

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# AdaBoost – adaptacyjny boosting
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # model słaby
    n_estimators=50,
    learning_rate=1.0,        # wpływ każdego modelu
    random_state=42
)

# Gradient Boosting – optymalizuje funkcję straty
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,        # szybkość uczenia
    max_depth=3,              # głębokość drzewa
    subsample=0.8,            # % próbek do każdego kroku
    random_state=42
)

ada.fit(X_train, y_train)
gb.fit(X_train, y_train)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `n_estimators` | 50 | liczba modeli w sekwencji | zwiększ jeśli underfitting, zmniejsz jeśli overfitting |
| `learning_rate` | 1.0 (Ada) / 0.1 (GB) | siła wpływu każdego modelu | zmniejsz (0.01-0.05) dla więzszej stabilności, zwiększ (0.1-1.0) dla szybszej zbieżności |
| `estimator` (Ada) | DecisionTree | model bazowy | zwykle `max_depth=1` (stumpy) by uniknąć overfittingu |
| `max_depth` (GB) | 3 | głębokość drzewa | zwiększ dla bardziej złożonych zależności, uważaj na overfitting |
| `subsample` (GB) | 1.0 | % próbek do każdego kroku | zmniejsz (0.8) by zmniejszyć overfitting |

#### Inne implementacje Boost
- **HistGradientBoostingClassifier** – szybsza wersja GB (binuje cechy na histogramy).
- **LightGBM**, **CatBoost**, **XGBoost** – popularne w prakyce, bardzo wydajne, obsługują GPU, early stopping.

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# XGBoost – najpopularniejsze
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,    # % cech do każdego drzewa
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
```

### Gotchas / Tips

- ⚠️ **Overfitting** – boosting jest podatny na nadmierne dopasowanie, szczególnie z powolnym learning_rate i dużą liczbą iteracji. Monitoruj wyniki na zbiorze walidacyjnym.
- 💡 **Learning rate** – niski learning_rate (0.01-0.05) wymaga więcej iteracji, ale daje bardziej stabilne wyniki. To klucz do dobrych wyników.
- 🔑 W **AdaBoost** model bazowy powinien być słaby (zwykle `max_depth=1` – drzewo jedno-poziomowe). W GB zwykle `max_depth=3-5`.
- 💡 **Early stopping** w XGBoost/LightGBM: trenuj na zbiorze walidacyjnym, zatrzymaj gdy metryka przestanie się poprawiać.
- ⚠️ Boosting to **sekwencyjne** trenowanie – nie można zrównoleglić różnych modeli (mogą być równoległy obliczenia w jednym drzewie, ale nie między modelami).

---

## Koncept 3: Stacking (Stacked Generalization)

### Co to jest?

Stacking to technika łączenia **wielu różnych modeli** (nie tylko oryginały tego samego typu jak Bagging/Boosting). Idea jest dwuwarstwowa:
1. **Warstwa 1 (Base Layer):** Trenujesz kilka różnych klasyfikatorów (np. drzewo, KNN, regresja logistyczna, SVM) na oryginalnych danych.
2. **Warstwa 2 (Meta Layer):** Predykcje modeli bazowych z warunkowej danych treningowych stają się **nowymi cechami** dla metamodelu (zwykle regresja logistyczna). Metamodel uczy się jak najlepiej **ważyć** i **kombinować** predykcje modeli bazowych.

To pozwala metamodelowi nauczyć się, które modele bazowe są dobre w jakich sytuacjach.

### Kiedy używać?

Stacking sprawdza się gdy:
- Masz **różnorodne modele** z **komplementarnymi mocnymi stronami** – każdy uchwytuje inne aspekty danych,
- Chcesz **maksymalizować dokładność** – meta-model widzi wszystko i decyduje optymalnie,
- Masz **wystarczająco dużo danych** – Stacking wymaga osobnego zestawu walidacyjnego do trenowania meta-modelu,
- Możesz sobie pozwolić na **wyższy czas treningu i złożoność** – musisz wytrenować wiele modeli bazowych + meta-model.

### Kod

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Definiuj modele bazowe
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))  # probability=True wymagane!
]

# Metamodel
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# StackingClassifier
stacker = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,                      # k-fold do trenowania meta-modelu
    passthrough=False          # czy dodać oryginalne cechy?
)

stacker.fit(X_train, y_train)
y_pred = stacker.predict(X_test)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `estimators` | – | lista (nazwa, model) dla warstwy bazowej | wybierz modele, które są dobre i różne |
| `final_estimator` | – | metamodel (warstwa 2) | zwykle regresja logistyczna, ale możesz spróbować innego |
| `cv` | 5 | liczba foldów do k-fold walidacji meta-modelu | zwiększ dla lepszych oszacowań meta-modelu |
| `passthrough` | False | czy dodać oryginalne cechy dla meta-modelu? | ustaw True jeśli oryginalne cechy zawierają dodatkową informację |

### Gotchas / Tips

- ⚠️ **passthrough=True** – jeśli ustawisz na True, metamodel otrzyma zarówno oryginalne cechy, jak i predykcje modeli bazowych. Może to poprawić wynik, ale zwiększa liczbę cech i ryzyko overfittingu.
- 💡 **Różnorodność modeli bazowych** – Stacking działa najlepiej gdy modele bazowe są **zróżnicowane** (np. drzewo, KNN, SVM). Jeśli wszystkie są podobne, meta-model ma mało nowych informacji.
- ⚠️ **probability=True** – niektóre klasyfikatory (SVM, LightGBM) wymagają `probability=True` aby wygenerować wyjście probabilistyczne dla meta-modelu.
- 🔑 **cv** parametr – sklearn automatycznie używa k-fold do trenowania meta-modelu (aby uniknąć data leakage). Nie trzeba ręcznie tworzyć foldów.

---

## Koncept 4: Bias–Wariancja Trade-off

### Co to jest?

Każdy model ML ma dwa źródła błędu:
1. **Bias (obciążenie)** – systematyczne błędy, gdy model jest zbyt prosty i ignoruje złożoność danych. Model o wysokim bias podejmuje złe decyzje regularnie **w ten sam sposób**.
2. **Wariancja** – losowe fluktuacje, gdy model jest zbyt elastyczny i dopasowuje się do szumu. Model o wysokiej wariancji podejmuje różne decyzje na słabo różniących się danych.

### Wizualizacja:

- **High Bias, Low Variance** (underfitting): Model przewiduje prostą linię dla nieliniowego problemu. Dobre wyniki na treningu? Nie – konsekwentnie źle.
- **Low Bias, High Variance** (overfitting): Model wiggle'uje się wokół każdego punktu. Świetnie na treningu, słabo na testach.
- **Low Bias, Low Variance** (ideal): Model uchwyca rzeczywistą strukturę bez nadmiernego dopasowania.

### Jak metody zespołowe pomagają?

- **Bagging** → zmniejsza **wariancję** (uśrednianie zmniejsza losowe fluktuacje bez zwiększania bias).
- **Boosting** → zmniejsza **bias** (sekwencyjne uczenie na trudnych przypadkach ulepszy model systematycznie).
- **Stacking** → zmniejsza zarówno **bias** (różne modele) jak i **wariancję** (meta-model optymizuje kombinację).

### Kod – Ilustracja

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Dane – funkcja sin() z szumem
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_test_true = np.sin(X_test).ravel()

# Model 1: Regresja liniowa (wysoki bias, niska wariancja)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X_test)

# Model 2: Głębokie drzewo (niski bias, wysoka wariancja)
deep_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
deep_tree.fit(X, y)
y_pred_tree = deep_tree.predict(X_test)

# Model 3: Gradient Boosting (niski bias, umiarkowana wariancja)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb.fit(X, y)
y_pred_gb = gb.predict(X_test)

# Porównanie
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='train data')
plt.plot(X_test, y_test_true, 'k--', label='true function')
plt.plot(X_test, y_pred_lin, label='Linear (high bias)')
plt.plot(X_test, y_pred_tree, label='Deep Tree (high var)')
plt.plot(X_test, y_pred_gb, label='Gradient Boosting')
plt.legend()
plt.show()
```

### Gotchas / Tips

- 💡 **Złota zasada:** Jeśli model osiąga wysoką dokładność na treningu, ale słabą na testach → **wysoka wariancja** (overfitting). Jeśli niska na obu → **wysoki bias** (underfitting).
- 🔑 **Regularyzacja** (dropout, L1/L2, max_depth) zmniejsza wariancję kosztem bias. To zawsze kompromis.
- ⚠️ Duża liczba modeli w boostingu/baggingu nie zawsze pomaga – może zwiększyć wariancję przy zbyt małym learning_rate.

---

## 📊 Przydatne do Projektu

### Wymagane metryki
Projekt u dr. Smółki wymaga raportowania **do każdego modelu**:
- **Klasyfikacja:** Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix
- **Regresja:** R², MAE, RMSE, Mean Absolute Percentage Error (MAPE)

### Metody zespołowe w kontekście projektu

1. **Random Forest i Gradient Boosting** – najpopularniejsze metody na dane rzeczywiste (medyczne, finansowe).
   - Random Forest: szybki, łatwy do tuningu, dobrze interpretowalne feature importance.
   - Gradient Boosting (XGBoost/LightGBM): wymagają tuningu, ale zwykle lepsze wyniki.

2. **Preprocessing przed ensemble**
   - Skaluj cechy (`StandardScaler`, `MinMaxScaler`) jeśli używasz KNN, SVM w stackingu.
   - Dla Random Forest nie trzeba skalować.
   - Obsługuj brakujące wartości **przed** treningiem zespołu.

3. **K-fold walidacja w teamach**
   ```python
   from sklearn.model_selection import cross_validate
   
   cv_results = cross_validate(
       RandomForestClassifier(),
       X_train, y_train,
       cv=5,
       scoring=['accuracy', 'f1', 'roc_auc']
   )
   # Raportuj średnią i std dla każdej metryki
   ```

4. **Porównanie modeli na testach**
   - Trenuj wiele modeli (prosty klasyfikator, Bagging, Boosting, Stacking).
   - Raportuj metryki dla każdego – pokaż, że ensemble jest lepszy.
   - Opisz w raporcie dlaczego dany model wypadł lepiej.

5. **Feature importance**
   ```python
   # Dla Random Forest
   importances = model.feature_importances_
   # Wizualizuj top N cech
   ```

6. **Hiperparametr tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [3, 5, 10],
       'learning_rate': [0.01, 0.1, 1.0]
   }
   gs = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
   gs.fit(X_train, y_train)
   best_model = gs.best_estimator_
   ```

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy na przyszłość, NIE jako materiał do raportu dla prowadzącego.

### 🔄 Aktualność (2025-2026)

- **Gradient Boosting (XGBoost, LightGBM, CatBoost)** – wciąż dominują w konkursach ML i przemyśle. Stan-of-the-art dla tabelarycznych danych.
- **Random Forest** – "work horse" ML, czasem przegrywa z XGBoost na dokładności, ale jest bardziej stabilny i łatwiejszy do interpretacji.
- **Deep Learning (Neural Networks)** – dla danych strukturyzowanych zwykle poniżej ensemble methods, ALE jeśli masz obrazy lub tekst, neural networks wygrywają.
- **AutoML (H2O, AutoGluon)** – coraz popularniejsze; automatycznie tunują hiperparametry i łączą modele. Wart spróbować zamiast ręcznego tuningu.

### 📚 Zasoby & Komentarze

1. **XGBoost/LightGBM Guide**
   - XGBoost lepsze dla małych danych, bardziej stabilne.
   - LightGBM szybsze dla dużych danych (będzie szukaj parametrów `num_leaves`, `max_depth`, `feature_fraction`).
   - Zawsze używaj `early_stopping_rounds` + `eval_set` aby zatrzymać trenowanie gdy walidacja prestaje się poprawiać.

2. **Stacking w prakyce**
   - Rzadko lepsze niż dobrze wytunowany XGBoost.
   - Przydatne gdy masz kilka różnych typów danych (np. tabela + tekst) i chcesz połączyć modele na każdym typie.

3. **Feature importance**
   - XGBoost: `plot_importance()`, ale nie musisz usuwać cech o niskiej wichtości – mogą być interakcją.
   - Random Forest: prostsze do interpretacji, ale pamiętaj że feature importance to średnia redukcji Gini – nie mówi o przyczynowości.

### ⚡ Nowsze podejścia (2025-2026)

1. **Neural Network Ensembles** – trenowanie wielu różnych architektur neuronowych i średniowanie predykcji. Działa dobrze, ale drogo obliczeniowo.
2. **Tabular Neural Networks** (NODE, SAINT, TabR) – sieci neuronowe specjalnie dla danych tabelarycznych, pierwsze wyniki obiecujące, ale XGBoost nadal dominuje.
3. **Explainability** – SHAP (`import shap`) jest nowością: pokazuje indywidualny wkład każdej cechy do każdej predykcji. Lepsze niż feature importance.
   ```python
   import shap
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```

### 🆚 Porównania

| Metoda | Pros | Cons | Kiedy używać |
|--------|------|------|----------|
| Random Forest | Szybko, stabilnie, interpretowalne | Gorsze wyniki niż boosting | Quick baseline, duże dane |
| XGBoost | Najlepsze wyniki, elastyczne, early stopping | Wymaga tuningu, wolniejszy | Konkursy, produkcja, kiedy liczy się dokładność |
| Gradient Boosting | Bardzo elastyczne | Łatwo się przeuczyć, wymaga walidacji | Mały dataset, chcesz najlepszych wyników |
| Stacking | Kombinuje różne modele | Drogi obliczeniowo, rzadko lepszy niż XGBoost | Multiple different data types |
| CatBoost | Świetnie obsługuje kategorie | Powolny, retro | Dane z dużo kategoriami |

### 🎓 Praktyczne rady do projektu

1. **Start z Random Forest** – szybki baseline, zobaczysz czy zadanie w ogóle jest sensowne.
2. **Przejdź na XGBoost** – prawie zawsze lepsze wyniki z domyślnymi parametrami.
3. **Tune hiperparametry** – GridSearchCV lub Optuna dla XGBoost (`learning_rate`, `max_depth`, `subsample`).
4. **Raportuj ablation study** – trening kilka modeli (prosty, bagging, boosting, stacking) i pokaż, że ensemble wygrywają.
5. **Nie zapomnij o preprocessing** – brakujące wartości, outliers, skalowanie. To 50% pracy.
