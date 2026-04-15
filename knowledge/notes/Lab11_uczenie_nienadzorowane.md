# Lab 11: Uczenie Nienadzorowane (Unsupervised Learning)

> **Tematyka:** Klasteryzacja danych (K-means, DBSCAN, hierarchiczna) oraz uczenie reguł asocjacyjnych (Apriori, support, confidence, lift). Metody odkrywania struktury i wzorców w danych bez etykiet.
> **Notebooki:** `Uczenie nienadwzorowane.ipynb`, `Zadanie.ipynb`
> **Kluczowe biblioteki:** `sklearn.cluster`, `scipy.cluster.hierarchy`, `mlxtend` (Apriori), `pandas`

---

## TL;DR

Uczenie nienadzorowane to odkrywanie ukrytych struktur danych bez dostępu do etykiet. Trzy główne techniki: (1) **K-means** — podział na sfery klastrów wokół centroidów, mierz jakość metodą łokcia és silhouette score; (2) **DBSCAN** — wykrywanie klastrów na bazie gęstości, obsługuje szum i nieregularne kształty, dobierz `eps` z k-distance plot; (3) **Hierarchiczna klasteryzacja** — buduje dendrogram bez konieczności podawania k z góry. Dla reguł asocjacyjnych (koszykami): **support** (jak często zestawu), **confidence** (P(Y|X)), **lift** (niezależność) — lift > 1 oznacza pozytywny związek, perfect do segmentacji klientów i analizy zachowań zakupowych.

---

## Spis Treści

- [Koncept 1: Klasteryzacja i K-means](#koncept-1-klasteryzacja-i-k-means)
- [Koncept 2: DBSCAN — klasteryzacja oparta na gęstości](#koncept-2-dbscan--klasteryzacja-oparta-na-gęstości)
- [Koncept 3: Hierarchiczna klasteryzacja](#koncept-3-hierarchiczna-klasteryzacja)
- [Koncept 4: Miary oceny klasteryzacji](#koncept-4-miary-oceny-klasteryzacji)
- [Koncept 5: Uczenie reguł asocjacyjnych](#koncept-5-uczenie-reguł-asocjacyjnych)
- [Koncept 6: Support, Confidence, Lift](#koncept-6-support-confidence-lift)

---

## Koncept 1: Klasteryzacja i K-means

### Co to jest?

**Klasteryzacja** to podział zbioru danych na grupy (klastry) w taki sposób, aby obiekty w obrębie jednej grupy były do siebie bardziej podobne niż do grupy innej. W odróżnieniu od klasyfikacji, w klasteryzacji nie mamy z góry zdefiniowanych etykiet — algorytm samodzielnie identyfikuje struktury w danych.

**K-means** to najpopularniejszy algorytm klasteryzacji, który dzieli dane na `k` sferycznych klastrów, minimalizując sumę kwadratów odległości punktów od przypisanych centroidów (within-cluster sum of squares — WCSS).

### Kiedy używać?

- Segmentacja klientów w marketingu
- Analiza eksploracyjna dużych zbiorów danych
- Grupy tematów w analizie tekstu
- Wstępne zrozumienie struktury danych przed dalszym modelowaniem
- **Gdy podatny na zakładanie kulistych klastrów o podobnym rozmiarze**

### Kod

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pandas as pd

# Dane
iris = load_iris()
X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])

# Klasteryzacja
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X)

# Centroidy
centroids = kmeans.cluster_centers_  # środki klastrów
wcss = kmeans.inertia_  # suma kwadratów odległości w klastrach
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `n_clusters` | — | liczba klastrów k | zawsze trzeba podać |
| `random_state` | None | seed inicjalizacji | dla reprodukowalności ustaw wartość |
| `n_init` | 10 | ile razy uruchamia algorytm z różnymi inicjalizacjami | zwiększ (15-20) dla większych zbiorów |
| `max_iter` | 300 | maksymalna liczba iteracji | zwykle domyślnie ok, zwiększ jeśli nie konwerguje |

### Gotchas / Tips

- ⚠️ **K-means jest wrażliwy na inicjalizację** — zawsze uruchamiaj kilka razy lub zwiększ `n_init`
- ⚠️ **Wynik zależy od skali danych** — zawsze standaryzuj (`StandardScaler`) przed klasteryzacją
- ⚠️ **Nie potrafi obsługiwać klastrów niesferycznych** — dla irregular shapes użyj DBSCAN
- 💡 **Dobierz k metodą łokcia (elbow method)** — oblicz WCSS dla różnych k i szukaj punktu załamania
- 💡 **Użyj silhouette score** do oceny jakości — wartości bliskie 1 oznaczają dobre separacja
- 🔑 **K-means jest szybki**, nawet dla bardzo dużych zbiorów danych

---

## Koncept 2: DBSCAN — klasteryzacja oparta na gęstości

### Co to jest?

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to algorytm, który grupuje punkty na podstawie spójności przestrzennej i gęstości. W przeciwieństwie do K-means:
- **Nie wymaga podawania liczby klastrów z góry**
- **Potrafi wykrywać klastry o nieregularnych kształtach**
- **Automatycznie identyfikuje szum (outliers)**

Algorytm definiuje trzy rodzaje punktów:
- **Punkt rdzeniowy (core point)** — ma co najmniej `min_samples` punktów w promieniu `eps`
- **Punkt graniczny (border point)** — ma mniej niż `min_samples`, ale leży w promieniu `eps` jakiegoś punktu rdzeniowego
- **Punkt odstający (noise point)** — nie spełnia żadnego z powyższych; przypisywany do klastra `-1`

### Kiedy używać?

- Dane z klastrami o nieregularnych kształtach (nie-sferycznych)
- Dane zawierające szum i outliers
- Gdy liczba klastrów nie jest znana z góry
- Analiza danych geograficznych (np. skupiska miast)
- **Nie: Dane o zbyt dużych różnicach w gęstości klastrów**

### Kod

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standaryzacja (obowiązkowa!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN z dobrze dobranym eps
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# -1 oznacza szum
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `eps` | 0.5 | promień sąsiedztwa | **krytyczne** — dobieraj z k-distance plot |
| `min_samples` | 5 | minimalna liczba punktów w `eps` dla core point | zwiększ dla dużych zbiorów, zmniejsz dla małych |
| `metric` | 'euclidean' | miara odległości | dla high-dimensional spróbuj 'cosine' |

### Gotchas / Tips

- ⚠️ **`eps` to najważniejszy parametr** — zła wartość zepsuł całą klasteryzację
- ⚠️ **DBSCAN również wymaga standaryzacji** — dane na różnych skalach mogą się źle grupować
- 💡 **Dobierz `eps` z wykresu k-distance plot** — szukaj "łokcia" na wykresie sortowanych odległości do k-tego sąsiada (gdzie k = min_samples - 1)
- 💡 **Zwiększ `min_samples` dla większych zbiorów** — reguła praktyczna: min_samples = 2 × liczba cech
- 🔑 **DBSCAN zwróci punkty szumu jako `-1`** — zawsze sprawdź, ile masz outlierów

---

## Koncept 3: Hierarchiczna klasteryzacja

### Co to jest?

**Hierarchiczna klasteryzacja** buduje strukturę drzewa (dendrogram), pokazującą hierarchię grupowania. Nie wymaga z góry podawania liczby klastrów — zamiast tego wybierasz poziom „przecięcia" dendrogramu, aby uzyskać konkretną liczbę klastrów.

Dwa podejścia:
- **Agregacyjne (bottom-up)**: Każdy punkt zaczyna jako osobny klaster, następnie są stopniowo łączone
- **Dzielące (top-down)**: Wszystko w jednym klastrze, które są dzielone na mniejsze

### Kiedy używać?

- Gdy chcesz zobaczyć hierarchię grupowania (nie tylko końcowy rezultat)
- Gdy liczba klastrów nie jest znana z góry
- Dane biologiczne (taksonomia, analiza genów)
- Analiza eksploracyjna struktur hierarchicznych
- **Nie: Bardzo duże zbiory danych** (złożoność O(n²) w pamięci)

### Kod

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Obliczenie macierzy łączeń
linked = linkage(X, method='ward')  # 'ward' = minimalizacja wariancji w klastrach

# Rysowanie dendrogramu
plt.figure(figsize=(12, 6))
dendrogram(linked, leaf_color_threshold=0)  # leaf_color_threshold = próg koloru
plt.show()

# Cięcie na konkretną liczbę klastrów
clusters = fcluster(linked, t=3, criterion='maxclust')  # 3 klastry
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `method` | — | jak liczyć odległość między klastrami | `'ward'` (wariancja), `'complete'` (daleki sąsiad), `'average'` (wszystkie) |
| `metric` | 'euclidean' | odległość między punktami | rzadko zmieniane |
| `t` (fcluster) | — | poziom cięcia (liczba klastrów lub odległość) | ustaw, aby uzyskać żądaną liczbę klastrów |

### Gotchas / Tips

- ⚠️ **Hierarchiczna klasteryzacja zużywa dużo pamięci** — dla > 10k próbek może być wolna
- ⚠️ **`method='ward'` jest najczęściej używany**, ale do danych skalowych spróbuj `'complete'` lub `'average'`
- 💡 **Dendrogram pokazuje historię łączeń** — im wyżej połączenie, tym bardziej różne klastry
- 💡 **Szukaj na dendrogramie największych "skoków" w wysokości** — to sugerowane poziomy cięcia
- 🔑 **Hierarchiczna klasteryzacja jest deterministyczna** — zawsze da ten sam wynik

---

## Koncept 4: Miary oceny klasteryzacji

### Co to jest?

Ponieważ w uczeniu nienadzorowanym nie mamy etykiet, ocena jakości klasteryzacji jest **subiektywna**. Istnieją jednak wskaźniki statystyczne:

#### **Metoda łokcia (Elbow Method)** — do K-means
Obliczasz WCSS (within-cluster sum of squares) dla różnych `k` i szukasz punktu, gdzie WCSS przestaje gwałtownie maleć.

```python
from sklearn.cluster import KMeans
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Wykres
plt.plot(range(1, 11), wcss)
plt.xlabel('k')
plt.ylabel('WCSS')
plt.show()  # szukaj punktu załamania
```

#### **Silhouette Score** — uniwersalny, dla wszystkich metod
Wskaźnik od -1 do 1, mierzący, jak dobrze punkt "pasuje" do swojego klastra:
- $> 1$: dobrze oddzielone klastry
- $≈ 0$: punkty na granicy klastrów
- $< 0$: punkt przypisany do niewłaściwego klastra

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

# Wybierz k z najwyższym score
best_k = 2 + scores.index(max(scores))
```

#### **k-distance plot** — do DBSCAN
Dla każdego punktu obliczasz odległość do jego k-tego sąsiada (k = `min_samples - 1`). Szukasz "łokcia" — to sugerowana wartość `eps`.

```python
from sklearn.neighbors import NearestNeighbors
k = 4  # min_samples - 1
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X)
distances, _ = nbrs.kneighbors(X)

k_distances = distances[:, k-1]
k_distances_sorted = np.sort(k_distances)
plt.plot(k_distances_sorted)
plt.ylabel('Distance to 4th NN')
plt.show()  # szukaj łokcia
```

### Gotchas / Tips

- ⚠️ **Żadna miara nie jest uniwersalna** — zawsze kombinuj kilka (elbow + silhouette)
- 💡 **Silhouette score jest lepszy od WCSS**, ponieważ uwzględnia względną odległość
- 💡 **k-distance plot jest najbardziej praktycznym narzędziem do DBSCAN**
- 🔑 **Zawsze wizualizuj wyniki** — obrazek jest warte więcej niż liczby

---

## Koncept 5: Uczenie reguł asocjacyjnych

### Co to jest?

**Uczenie reguł asocjacyjnych** to odkrywanie zależności i wzorców współwystępowania w danych transakcyjnych. Celem jest znalezienie reguł postaci:

> **Jeśli klient kupił X, to z dużym prawdopodobieństwem kupił też Y**
> `{chleb, masło} ⇒ {mleko}`

Algorytm **Apriori** znajduje **częste zestawy produktów** (itemsets), a następnie buduje reguły z minimalnym wsparciem (support).

### Kiedy używać?

- Analiza koszykowa (market basket analysis)
- Rekomendacje produktów
- Odkrywanie wzorców w transakcjach zakupowych
- Optymalizacja układu produktów w sklepie
- Tworzenie promocji wiązanych

### Kod

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Transakcje (każdy element to lista produktów)
transactions = [
    ['mleko', 'chleb', 'masło'],
    ['mleko', 'pieluchy', 'jajka'],
    ['piwo', 'cola'],
    ['chleb', 'masło'],
]

# 1. Kodowanie do postaci binarnej
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 2. Częste zestawy (min_support=0.3)
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# 3. Reguły asocjacyjne (min_confidence=0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 4. Wynik
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### Kluczowe parametry

| Parametr | Gdzie | Domyślnie | Co robi | Kiedy zmienić |
|----------|-------|-----------|---------|---------------|
| `min_support` | `apriori()` | — | szukaj zestawów powyżej tej częstości (0-1) | zwiększ dla dużych zbiorów (aby nie było zbyt wiele reguł) |
| `metric` | `association_rules()` | — | miara do filtrowania (`'confidence'`, `'lift'`, etc.) | zależy od celu |
| `min_threshold` | `association_rules()` | — | próg dla wybranej metryki | zwykle 0.5-0.7 dla confidence, >1 dla lift |
| `use_colnames` | `apriori()` | False | jeśli True, reguły zawierają nazwy produktów zamiast indeksów | zawsze True |

### Gotchas / Tips

- ⚠️ **Apriori generuje dużo reguł** — zawsze ustaw rozsądny `min_support` (0.1-2% dla średnich zbiorów)
- ⚠️ **Support jest obliczany dla całego zestawu** — jeśli X→Y ma support=0.1, to zarówno X i Y muszą być w 10% transakcji
- 💡 **Użyj `lift` do filtrowania** — lift > 1 oznacza pozytywny związek niezależny od samych częstości
- 💡 **Posortuj wyniki wg `lift`**, aby znaleźć najinteresujące reguły
- 🔑 **Zawsze interpretuj wyniki biznesowo** — wysoki lift ale niski support = niszowa reguła

---

## Koncept 6: Support, Confidence, Lift

### Support — Jak często zestawu

**Support** określa, jak często dany zestaw produktów występuje w całym zbiorze transakcji:

$$\text{support}(X \cup Y) = \frac{\text{liczba transakcji zawierających X i Y}}{\text{liczba wszystkich transakcji}}$$

Przykład:
- 1000 transakcji, z których 150 zawiera {chleb, mleko}
- support({chleb, mleko}) = 150 / 1000 = **0.15** (w 15% transakcji)

**Interpretacja:** Wsparcie mówi, jak popularna jest reguła w całej populacji. Niska support = reguła rzadka, wysoki support = reguła częsta (ogólna).

### Confidence — Jak pewna reguła

**Confidence** (zaufanie) opisuje, jak często reguła się sprawdza — czyli ile procent transakcji zawierających X zawiera też Y:

$$\text{confidence}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)}$$

czyli: $P(Y | X)$

Przykład:
- Spośród 200 osób kupujących chleb, 120 kupiło też mleko
- confidence(chleb ⇒ mleko) = 120 / 200 = **0.6** (w 60% przypadków gdy kupią chleb, kupią też mleko)

**Interpretacja:** Im wyższa confidence, tym bardziej "pewna" reguła. Confidence=1.0 oznacza, że zawsze gdy X, to Y.

### Lift — Jak niezależne są X i Y

**Lift** mówi, **ile razy częściej** X i Y występują razem niż gdyby były niezależne statystycznie:

$$\text{lift}(X \Rightarrow Y) = \frac{\text{confidence}(X \Rightarrow Y)}{\text{support}(Y)} = \frac{\text{support}(X \cup Y)}{\text{support}(X) \cdot \text{support}(Y)}$$

| Wartość lift | Znaczenie |
|--------------|-----------|
| **> 1** | X i Y występują razem **częściej niż oczekiwano** (reguła istotna, pozytywny związek) |
| **= 1** | X i Y są **statystycznie niezależne** (brak związku) |
| **< 1** | X i Y występują razem **rzadziej niż oczekiwano** (negatywny związek) |

Przykład:
- support(chleb) = 0.2, support(mleko) = 0.3
- support({chleb, mleko}) = 0.1
- confidence(chleb ⇒ mleko) = 0.1 / 0.2 = 0.5
- **lift** = 0.5 / 0.3 = **1.67**

To oznacza: mleko występuje z chlebem **1.67 razy częściej** niż byśmy się spodziewali, gdyby były niezależne.

### Gotchas / Tips

- ⚠️ **Wysoka confidence ≠ wysoki lift** — reguła może być pewna, ale przypadkowa (np. jeśli Y jest bardzo popularne)
- ⚠️ **Niski support reguły ≠ zła reguła** — niszowe powiązania mogą być biznesowo cennym insights
- 💡 **Zawsze szukaj reguł z lift > 1.5** — to sugeruje prawdziwy związek
- 💡 **Support filtruje rzadkie reguły, confidence filtruje słabe reguły, lift filtruje przypadkowe reguły**
- 🔑 **Ranking: lift > confidence > support** — jeśli muszą wybrać, priorytetyzuj lift

---

## 📊 Przydatne do Projektu

Jeśli projekt dotyczy zagadnienia ML na danych medycznych (klasyfikacja/regresja):
- Ten lab **nie będzie bezpośrednio wykorzystany** w ramach klasyfikacji nadzorowanej
- Jednak techniki tutaj mogą przydać się do:
  - **Wstępnej eksploracji struktury danych** — K-means do sprawdzenia naturalnych klastrów w danych
  - **Analizy interakcji między zmiennymi** — reguły asocjacyjne dla zmiennych kategorowych
  - **Segmentacji pacjentów** — clusteryzacja mogła by się przydać, jeśli projekt byłby na temat prognozowania dla grup

- Z projektu zapamiętaj: metryki (accuracy, F1, AUC, RMSE, MAE), k-fold walidacja, preprocessing, porównanie z literaturą

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy na przyszłość, NIE jako materiał do raportu dla prowadzącego.

- 🔄 **Aktualność:**
  - K-means i DBSCAN to wciąż fundamentalne algorytmy (2025).
  - Hierarchiczna klasteryzacja ubywa na popularności w big data (za wolna dla milionów próbek), ale jest teaching standard.
  - Apriori to praktycznie classic, ale dla big data (miliardy transakcji) używa się technik takie FP-Growth zamiast.

- ⚡ **Nowsze podejścia:**
  - **UMAP / t-SNE** zamiast PCA do wizualizacji — lepiej zachowują strukturę lokalną
  - **Gaussian Mixture Models (GMM)** zamiast K-means — bardziej pełne, mogą modelować klastry o różnych wariancjach
  - **Spectral Clustering** — lepiej dla klastrów niesferycznych niż K-means
  - **FP-Growth** zamiast Apriori — znacznie szybsze dla dużych zbiorów

- 🆚 **Porównania:**
  - K-means vs DBSCAN: K-means szybszy, ale założy kuliste klastry; DBSCAN wolniejszy, ale bardziej elastyczny
  - Apriori vs Random Forest feature importance: Apriori mówi o relacjach między zmiennymi, RF mówi o ich mocy predykcyjnej
  - Hierarchiczna klasteryzacja vs. K-means: Hierarchiczna pokazuje pełną historię łączeń, K-means daje szybki wynik na jednym poziomie

---

**Koniec notatki Lab 11: Uczenie Nienadzorowane**
