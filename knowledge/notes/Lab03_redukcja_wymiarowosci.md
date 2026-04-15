# Lab 3: Redukcja Wymiarowości

> **Tematyka:** Techniki redukcji wymiarowości danych: PCA, testy istotności przed PCA, interpretacja ładunków czynnikowych oraz krótkie porównanie z t-SNE i UMAP.
> **Notebooki:** `Redukcja wielowymiarowości.ipynb`, `PCA_functions_min.ipynb`, `Zadanie PCA.ipynb`
> **Kluczowe biblioteki:** sklearn, pandas, numpy, scipy, matplotlib, plotly

---

## TL;DR

Dane wysokowymiarowe utrudniają trenowanie modeli i wizualizację. **PCA** to podstawowa liniowa metoda redukcji wymiarowości: szuka kierunków maksymalnej wariancji i pozwala zmniejszyć liczbę cech przy relatywnie małej utracie informacji. Przed PCA warto sprawdzić sens jego użycia testami **Bartletta** i **KMO**, a po dopasowaniu dobrać liczbę komponentów na podstawie wyjaśnionej wariancji. Do wizualizacji nieliniowej można porównać PCA z t-SNE i UMAP.

---

## Spis Treści

- [Koncept 1: Przekleństwo Wymiarowości](#koncept-1-przekleństwo-wymiarowości)
- [Koncept 2: PCA](#koncept-2-pca)
- [Koncept 3: Testy przed PCA](#koncept-3-testy-przed-pca)
- [Koncept 4: Interpretacja PCA](#koncept-4-interpretacja-pca)
- [Koncept 5: Dobór liczby komponentów](#koncept-5-dobór-liczby-komponentów)
- [Koncept 6: Inne metody redukcji wymiarowości](#koncept-6-inne-metody-redukcji-wymiarowości)

---

## Koncept 1: Przekleństwo Wymiarowości

### Co to jest?

Im więcej cech, tym trudniej efektywnie pokryć przestrzeń danych. Dane stają się rzadkie, rośnie koszt obliczeń, a część zmiennych jest redundantna lub szumowa.

### Gotchas / Tips

- ⚠️ Więcej cech nie zawsze oznacza lepszy model.
- 💡 Redukcja wymiarowości to kompromis między prostotą a utratą informacji.
- 🔑 Standaryzacja jest krytyczna przed większością metod liniowych.

---

## Koncept 2: PCA

### Co to jest?

**Principal Component Analysis** znajduje nowe osie danych, które maksymalizują wariancję i są wzajemnie prostopadłe. Każda składowa to liniowa kombinacja oryginalnych zmiennych.

### Kod

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

X = df.drop(columns=['target'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))
print(pca.n_components_)
```

### Kluczowe parametry

| Parametr | Co robi | Kiedy zmieniać |
|----------|---------|----------------|
| `n_components` | liczba komponentów lub procent wariancji | np. `0.95` dla 95% wariancji |
| `svd_solver` | algorytm dekompozycji | `randomized` dla dużych danych |
| `random_state` | reproducibility | gdy solver używa losowości |

### Gotchas / Tips

- ⚠️ PCA bez standaryzacji bywa mylące.
- ⚠️ Outliery potrafią mocno zniekształcić wyniki.
- 💡 `fit()` tylko na zbiorze treningowym, `transform()` na testowym.

---

## Koncept 3: Testy przed PCA

### Test Bartletta

Sprawdza, czy macierz korelacji jest różna od jednostkowej. Jeśli `p-value < 0.05`, PCA ma sens.

```python
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

chi_sq, p_val = calculate_bartlett_sphericity(X)
print(chi_sq, p_val)
```

### Test KMO

Mierzy adekwatność danych do PCA w skali 0-1.

```python
from factor_analyzer.factor_analyzer import calculate_kmo

kmo_all, kmo_model = calculate_kmo(X)
print(kmo_model)
```

### Interpretacja KMO

| KMO | Interpretacja |
|-----|---------------|
| > 0.9 | doskonałe |
| 0.8-0.9 | bardzo dobre |
| 0.7-0.8 | dobre |
| 0.6-0.7 | umiarkowane |
| < 0.5 | raczej nie rób PCA |

### Gotchas / Tips

- 💡 Bartlett + KMO to dobry health check przed PCA.
- 🔑 Jeśli KMO jest niskie, rozważ selekcję cech zamiast PCA.

---

## Koncept 4: Interpretacja PCA

### Ładunki czynnikowe

Ładunki mówią, jak mocno dana oryginalna zmienna wpływa na konkretną składową główną.

```python
import pandas as pd

loadings = pca.components_.T
loadings_df = pd.DataFrame(
    loadings,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
print(loadings_df)
```

### Gotchas / Tips

- ⚠️ Interpretacja komponentów nie jest tak intuicyjna jak oryginalnych cech.
- 💡 Długi wektor na wykresie ładunków oznacza dobrą reprezentację zmiennej przez wybrane komponenty.
- 🔑 Factor loadings i biplot mają największy sens, gdy pierwsze komponenty wyjaśniają dużą część wariancji.

---

## Koncept 5: Dobór liczby komponentów

### Metody

1. procent wyjaśnionej wariancji,
2. kryterium Kaisera (`eigenvalue > 1`),
3. scree plot i punkt łokcia,
4. pragmatyka zadania: 2-3 komponenty do wizualizacji, więcej do modelowania.

### Kod

```python
import matplotlib.pyplot as plt
import numpy as np

pca_full = PCA()
pca_full.fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o')
plt.axhline(0.80, color='r', linestyle='--')
plt.axhline(0.95, color='g', linestyle='--')
plt.show()
```

### Gotchas / Tips

- 💡 Nie ma jednej magicznej liczby komponentów.
- 🔑 Zawsze porównaj wyniki modelu na danych oryginalnych i po PCA.

---

## Koncept 6: Inne metody redukcji wymiarowości

### t-SNE

Nieliniowa metoda do wizualizacji lokalnej struktury danych.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

### UMAP

Szybsza od t-SNE, zwykle lepiej zachowuje też strukturę globalną.

```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
```

### Gotchas / Tips

- ⚠️ t-SNE i UMAP są głównie do wizualizacji, nie zawsze do cech wejściowych modelu.
- 💡 UMAP bywa lepszym nowoczesnym wyborem do eksploracji.
- 🔑 Autoenkodery są kolejnym krokiem, gdy potrzebujesz nieliniowej kompresji na większą skalę.

---

## 📊 Przydatne do Projektu

- Standaryzuj dane przed PCA.
- Sprawdź Bartlett i KMO.
- Porównaj model na oryginalnych cechach i po PCA.
- Do raportu pokaż wykres skumulowanej wariancji, ewentualnie biplot i tabelę ładunków.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy, nie jako materiał do raportu.

- **PCA** nadal jest podstawą dla danych tabelarycznych i szybkich eksperymentów.
- **UMAP** jest dziś częstym wyborem do wizualizacji zamiast t-SNE.
- **Feature selection** bywa lepsze od PCA, jeśli zależy Ci na interpretowalności cech.

---

**Koniec notatki Lab 3: Redukcja Wymiarowości**
