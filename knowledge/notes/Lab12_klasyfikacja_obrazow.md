# Lab 12: Klasyfikacja Obrazów

> **Tematyka:** Konwolucyjne sieci neuronowe (CNN) do rozpoznawania obrazów. Transfer learning z wstępnie wytrenowanymi modelami. Feature extraction i klasyfikacja z użyciem embeddingów. Dataset Fashion MNIST. TensorFlow/Keras + scikit-learn.
> **Notebooki:** `00 Klasyfikacja obrazów.ipynb`, `01 Animals.ipynb`, `02 FasionMNIST.ipynb`
> **Kluczowe biblioteki:** TensorFlow/Keras, scikit-learn, numpy, pandas, PIL, matplotlib

---

## TL;DR

Lab 12 uczy fundamentów klasyfikacji obrazów poprzez CNN. Poznajemy: architekturę CNN (filtry konwolucyjne, pooling, pooling), transfer learning (Inception v3 do ekstrakcji cech), budowanie sieci od zera (Fashion MNIST), preprocessing obrazów (normalizacja, reshaping, one-hot encoding). Kluczowa idea: CNN wyodrębniają cechy obrazu automatycznie poprzez hierarchię filtrów. Wstępnie wytrenowane modele (np. Inception) oszczędzają czas — używamy ich embeddingów zamiast budować sieć od zera. Fashion MNIST to benchmark dla weryfikacji, czy model generalizuje.

---

## Spis Treści

- [Koncept 1: CNN — Fundamenty](#koncept-1-cnn--fundamenty)
- [Koncept 2: Transfer Learning z Inception v3](#koncept-2-transfer-learning-z-inception-v3)
- [Koncept 3: Fashion MNIST — Budowanie CNN od Zera](#koncept-3-fashion-mnist--budowanie-cnn-od-zera)
- [Koncept 4: Preprocessing i Augmentacja Obrazów](#koncept-4-preprocessing-i-augmentacja-obrazów)
- [Koncept 5: Embeddings vs. Końcowa Klasyfikacja](#koncept-5-embeddings-vs-końcowa-klasyfikacja)
- [Koncept 6: Random Forest na Embeddingach](#koncept-6-random-forest-na-embeddingach)

---

## Koncept 1: CNN — Fundamenty

### Co to jest?

Sieć konwolucyjna (CNN) to typ sieci neuronowej, która modeluje pracę ludzkiego widzenia. Zamiast przetwarzać każdy piksel osobno, CNN używa małych filtrów (kernelów) przesuwanych po obrazie. Filtr 3×3 jest nakładany na kolejne fragmenty obrazu, a z każdego fragmentu powstaje jedna wartość w mapie cech. Obliczenie to: suma iloczynów odpowiadających sobie elementów filtru i fragmentu obrazu. Pierwsza warstwa CNN wyodrębnia cechy proste (krawędzie, kolory), kolejne warstwy — cechy bardziej zaawansowane (kształty, tekstury, obiekty).

Kluczowe warstwy:
- **Conv2D** — ekstrakcja cech poprzez filtry
- **MaxPooling2D** — redukcja wymiarów (bierze max z fragmentu 2×2)
- **Flatten** — zmiana macierzy w wektor
- **Dense** — klasyfikacja (warstwy w pełni połączone)

### Kiedy używać?

CNN jest standardem dla:
- Klasyfikacji obrazów (kot/pies, znaki drogowe, choroby na RTG)
- Detekcji obiektów
- Segmentacji obrazów
- Rozpoznawania twarzy

Klasyczne podejście (KNN, SVM) wymaga ręcznej ekstrakcji cech (SIFT, HOG) — CNN robi to automatycznie, oszczędzając czas.

### Kod

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 klas
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_valid, y_valid))
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `filters` | brak | Liczba filtrów w warstwie Conv2D | Zwiększ dla bardziej złożonych danych, zmniejsz dla szybszego trenowania |
| `kernel_size` | `(3,3)` | Rozmiar filtru (3×3 to standard) | (5,5) dla większych struktur, (1,1) dla refinementu |
| `pool_size` | `(2,2)` | Rozmiar okna MaxPooling | 2×2 to standard, staraj się nie zmieniać |
| `activation` | brak | Funkcja aktywacji | ReLU w hidden, softmax w output, sigmoid dla binarnej |
| `batch_size` | 32 | Ile obrazów na raz | 128-256 dla mniejszych danych, 32-64 dla dużych |
| `epochs` | brak | Liczba iteracji | Zacznij od 10, następnie dostrojenie elastyczne |

### Gotchas / Tips

- ⚠️ **Uwaga na przetrenowanie** — jeśli validation accuracy spada, a train rośnie, model się przetrenowuje. Zmniejsz liczbę warstw lub dodaj Dropout.
- ⚠️ **Normalizacja pikseli** — dzielenie przez 255 jest OBOWIĄZKOWE, bez tego trenowanie będzie bardzo powolne.
- 💡 **Zacznij od małych modeli** — 2-3 warstwy konwolucyjne to zwykle wystarczy. Głębokie sieci (VGG, ResNet) to overkill dla prostych zadań.
- 🔑 **MaxPooling zmniejsza wymiar** — (28,28) → maxpool → (14,14). Zbyt wiele poolingu = strata informacji.

---

## Koncept 2: Transfer Learning z Inception v3

### Co to jest?

Transfer learning to technika, w której używamy modelu wytrenowanego na ogromnym zbiorze danych (np. ImageNet — 1,2M obrazów, 1000 klas) do naszego zadania. Zamiast trenować sieć od zera, bierzemy Inception v3, usuwamy ostatnią warstwę klasyfikacyjną (1000 neuronów) i wyciągamy **embeddingi** — wektory o wymiarze 2048, które reprezentują cechy obrazu rozpoznane przez model.

Inception v3 już wie, co to są oczy, uszy, tekstury, kształty etc. My tylko używamy tej wiedzy!

### Kiedy używać?

- Mało danych treningowych (< 1000 obrazów)
- Klasy zbliżone do ImageNet (zwierzęta, przedmioty codzienne)
- Szybkość jest ważna — ekstrakcja cech trwa minuty zamiast godzin
- Transfer learning skraca training o 10–100×

**Kiedy NIE**: dane zupełnie inne (np. zdjęcia mikroskopowe), mogą być problemy — wtedy finetune top warstw lub trenuj od zera.

### Kod

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Model bez ostatniej warstwy klasyfikacyjnej
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Ekstrakcja cech z jednego obrazu
def extract_features(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Inception wymaga 299x299
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalizacja ImageNet
    features = model.predict(img_array, verbose=0)
    return features.flatten()  # 2048 wymiarów

# Przetworzenie całego folderu
def process_folder(root_dir):
    data, labels = [], []
    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        for fname in tqdm(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    vec = extract_features(os.path.join(class_dir, fname))
                    data.append(vec)
                    labels.append(label)
                except:
                    pass
    return np.array(data), np.array(labels)

X_train, y_train = process_folder("./Data/Train")
# Zapisz do CSV
df = pd.DataFrame(X_train)
df['label'] = y_train
df.to_csv("embeddings.csv", index=False)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `weights` | `'imagenet'` | Wstępne wagi z ImageNet | Zostawiaj `'imagenet'` — to sens transfer learning |
| `include_top` | `True` | Czy włączyć ostatnią warstwę (1000 neuronów) | Zawsze `False` — chcemy embeddingi |
| `pooling` | `'avg'` | Globalne pooling (avg/max) | 'avg' to standard, zmienia rozmiar embedingu |
| `target_size` | `(299, 299)` | Rozmiar wejścia dla Inception | **MUSI** być (299, 299) — zmiana złamie model |

### Gotchas / Tips

- ⚠️ **NIGDY nie zmieniaj `include_top=True`** — chcesz embeddingi, nie klasyfikację na 1000 klas ImageNet.
- ⚠️ **Preprocess_input jest KLUCZOWY** — Inception wymaga normalizacji [-1, 1] (nie [0, 1]). Bez niego wyniki będą śmieci.
- ⚠️ **Rozmiar (299, 299) jest OBOWIĄZKOWY** — jeśli zmienisz, model zacrashuje.
- 💡 **Embeddingi są stabilne** — możesz je wyciągnąć raz, zapisać do CSV i wielokrotnie ich używać. Oszczędzasz czas.
- 💡 **Skalowanie** — jedno image_processing trwa ~100ms. Na 10k obrazów = kilka godzin. Ale robisz to raz!

---

## Koncept 3: Fashion MNIST — Budowanie CNN od Zera

### Co to jest?

Fashion MNIST to zbiór 70k obrazów (60k train, 10k test) odzieży z Zalando w rozdzielczości 28×28 pikseli, skala szarości. Każdy obraz ma etykietę: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

To **benchmark** do nauki CNN — prostszy niż ImageNet, bardziej realistyczny niż MNIST (cyfry). Idealna piaskownica do eksperymentów.

### Kiedy używać?

- Aby oswoić się z Keras/Sequential API
- Do testowania architektur CNN (ile warstw? ile filtrów?)
- Do nauki preprocessing (normalizacja, reshape, one-hot encoding)
- Aby zweryfikować, czy Twoja implementacja trenuje w ogóle

### Kod

```python
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# 1. Załaduj zbiór
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# Shapes: (60000, 28, 28), (10000, 28, 28)

# 2. Podziel train na train + validation (80/20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 3. Reshape dla CNN (brak kanałów RGB, czarno-białe)
X_train = X_train.reshape(-1, 28, 28, 1)
X_valid = X_valid.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 4. Normalizacja [0, 255] → [0, 1]
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

# 5. One-hot encoding labels (10 klas)
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

# 6. Zbuduj model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 7. Trenuj
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_valid, y_valid),
    verbose=1
)

# 8. Ewaluuj
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `reshape` | brak | Input shape dla CNN musi być 4D: (samples, height, width, channels) | Fashion MNIST: (-1, 28, 28, 1), ImageNet: (-1, 224, 224, 3) |
| `test_size` | brak | Procent danych do validation | 0.2 (20%) to standard |
| `batch_size` | 32 | Liczba obrazów na iterację | 128 dla fashion_mnist, zmniejsz dla szybszego tesztu |
| `epochs` | brak | Ile razy przejść przez całe dane | 10-20 dla Fashion MNIST, 50+ dla bardziej złożonych |

### Gotchas / Tips

- ⚠️ **Reshape jest OBOWIĄZKOWY** — jeśli nie dodasz wymiaru kanału (1 dla czarno-białych, 3 dla RGB), Keras się złamie.
- ⚠️ **One-hot encoding dla kategorycznego loss** — `categorical_crossentropy` wymaga wektorów [1, 0, 0, ...], a nie liczb [0, 1, 2, ...].
- 💡 **Fashion MNIST trenuje szybko** — 10 epok ≈ 2-3 minuty na CPU. Idealny do eksperymentów.
- 💡 **Validation accuracy > 91%** to bardzo dobry wynik. Jeśli jest < 85%, model nie konwerguje — zwiększ epochs lub zmień architekturę.

---

## Koncept 4: Preprocessing i Augmentacja Obrazów

### Co to jest?

Preprocessing to przygotowanie surowych pikseli do trenowania:
- **Reshape** — zmiana wymiarów dla CNN (dodanie kanału)
- **Normalizacja** — dzielenie przez 255 (wspólna skala [0, 1])
- **Konwersja koloru** — RGB → szarość (np. dla Fashion MNIST)
- **Zmiana rozmiaru** — resize do docelowego rozmiaru modelu

Augmentacja = tworzenie wariacji obrazów (rotacja, przesunięcie, zoom) aby:
1. Zbyt małe zbiory danych rozszerzyć
2. Model nauczył się być invariantny na transformacje

### Kiedy używać?

- Zawsze normalizuj piksele
- Reshape zawsze dla CNN
- Augmentacja gdy < 5000 obrazów treningowych
- Augmentacja dla real-world robustności (zdjęcia pod różnymi kątami)

### Kod

```python
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Normalizacja
X_train = X_train / 255.0

# Augmentacja (opcjonalna, ale rekomendowana dla małych zbiorów)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Trenowanie z augmentacją
model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=20,
    validation_data=(X_valid, y_valid)
)

# Ładowanie własnego zdjęcia z Internetu
from PIL import Image
import requests
from io import BytesIO

url = "https://example.com/image.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('L')  # Konwersja na szarość
img = img.resize((28, 28))  # Resize do rozmiaru modelu
X = np.array(img) / 255.0
X = X.reshape(1, 28, 28, 1)

# Predykcja
y_pred = model.predict(X)
```

### Gotchas / Tips

- ⚠️ **ImageDataGenerator to legacy** — w nowszych wersji Keras/TensorFlow użyj `image_dataset_from_directory()` (szybsze).
- ⚠️ **Augmentacja dodaje czasu trenowania** — ~2× dłużej. Stosuj tylko jeśli mało danych.
- 💡 **Normalizacja [0, 1] vs. [-1, 1]** — większość modeli trenuje na [0, 1]. Transfer learning (Inception) wymaga ImageNet preprocessing.
- 💡 **Konwersja .convert('L')** — konwertuj RGB → szarość dla Fashion MNIST. PNG bywa paskudny, JPG lepszy.

---

## Koncept 5: Embeddings vs. Końcowa Klasyfikacja

### Co to jest?

**Embeddingi** = wektory reprezentujące obraz w ukrytej przestrzeni. Dla Inception v3: **2048 wymiarów**.
**Klasyfikacja** = końcowa warstwa (Dense + softmax) przyporządkowująca klasy.

Dwie strategie:

1. **End-to-end CNN** — trenujemy całą sieć, ostatnia warstwa klasyfikuje bezpośrednio (Fashion MNIST, mały zbiór)
2. **Transfer Learning + Meta-classifier** — wyciągamy embeddingi wstępnie wytrenowaną siecią, trenujemy Random Forest (Animals)

### Kiedy używać co?

| Strategia | Kiedy | Plusy | Minusy |
|-----------|-------|-------|--------|
| End-to-end CNN | Duży zbiór (>50k), konkretne dane | Pełna optymalizacja, wysoka acc | Długie trenowanie, ryzyko overfittingu |
| Transfer Learning | Mały zbiór (<5k), ogólne klasy | Szybko, mało RAM, stabilne | Zależy od wstępnego treningu |

### Kod

```python
# === STRATEGIA 1: END-TO-END CNN ===
# Używaj w Fashion MNIST, małych zbiorach
model = Sequential([Conv2D(...), ..., Dense(num_classes, activation='softmax')])
model.fit(X_train, y_train, epochs=10)

# === STRATEGIA 2: TRANSFER LEARNING ===
# Używaj dla Animals, medialne zbiory
from sklearn.ensemble import RandomForestClassifier

# Krok 1: Wyciągnij embeddingi (Inception bez ostatniej warstwy)
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
# ... ekstrakcja cech ...
X_train_embeddings = np.array([extract_features(img) for img in images])

# Krok 2: Trenuj Random Forest na embeddingach
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_embeddings, y_train_labels)

# Predykcja na testowych embeddingach
y_pred = clf.predict(X_test_embeddings)
```

### Gotchas / Tips

- ⚠️ **Transfer learning ≠ finetune** — finetune = trenować na warstwach CNN (zaawansowane). Tu tylko meta-classifier.
- 💡 **Embeddingi zapisuj do CSV** — wyciągasz raz, używasz wielokrotnie (RandomForest, SVM, KNN, itp).
- 💡 **Random Forest ≠ jedyna opcja** — możesz też trenować Dense layers na embeddingach prostszych niż CNN.

---

## Koncept 6: Random Forest na Embeddingach

### Co to jest?

Random Forest to ensemble algorytmu — 100 drzew decyzyjnych głosując na wynik. Dla embeddingów (2048 cech):
1. Każde drzewo tnie cechy niezależnie
2. Głosowanie większościowe daje predykcję
3. OOB (Out-of-Bag) validation na darmo

Dlaczego Random Forest, a nie Dense layer?
- Bardziej interpretowalne (feature importance)
- Szybsze trenowanie (~sekundy vs. minuty)
- Brak hyperparametryzacji (domyślnie działa dobrze)
- Naturalnie obsługuje multi-class bez softmax

### Kiedy używać?

- Po embeddingach (transfer learning)
- Szybki baseline do porównania
- Małe zbiory treningowe (RF nie overfittuje jak sieci)
- Kiedy chcesz wiedzieć, które cechy są ważne

### Kod

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# Załaduj embeddingi z CSV
df_train = pd.read_csv("inception_embeddings_train.csv")
df_test = pd.read_csv("inception_embeddings_test.csv")

# Przygotowanie danych
X_train = df_train.drop(columns=["class", "file"]).values
y_train = df_train["class"].values
X_test = df_test.drop(columns=["class", "file"]).values
y_test = df_test["class"].values

# Trenowanie Random Forest
clf = RandomForestClassifier(
    n_estimators=100,        # Liczba drzew
    max_depth=None,          # Brak ograniczenia głębokości
    min_samples_split=2,     # Min próbek do podziału
    random_state=42,
    n_jobs=-1                # Parallelizacja
)
clf.fit(X_train, y_train)

# Predykcja
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)  # Prawdopodobieństwa

# Ewaluacja
print(classification_report(y_test, y_pred))

# Macierz pomyłek
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Macierz pomyłek")
plt.tight_layout()
plt.show()

# Analiza błędów
df_test['predicted'] = y_pred
df_errors = df_test[df_test['class'] != df_test['predicted']]
print(f"Błędnie sklasyfikowane: {len(df_errors)} / {len(df_test)}")

# Feature importance (które embeddingi są ważne?)
feature_importance = clf.feature_importances_
top_features = np.argsort(feature_importance)[-10:]  # Top 10
print("Top 10 wymiarów embeddingu:", top_features)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `n_estimators` | 100 | Liczba drzew | 100 to standard, zmiana do 50-200 mało zmienia |
| `max_depth` | None | Maks. głębokość drzewa | Zostawiaj None, zamiast tego `min_samples_split` |
| `min_samples_split` | 2 | Min próbek do podziału węzła | 2 to OK dla embeddingów (mało overfittingu) |
| `n_jobs` | 1 | Parallelizacja (-1 = wszystkie CPU) | -1 dla szybkości |
| `random_state` | None | Seed do reprodukowalności | Zawsze ustaw (42), bez tego wyniki mogą się zmieniać |

### Gotchas / Tips

- ⚠️ **RF dominuje na embeddingach** — oszczędzisz 10× na trenowaniu vs Dense networks.
- ⚠️ **Feature importance to narzędzie diagnostyczne** — możesz sprawdzić, które wymiary embedingu są ważne.
- 💡 **OOB score bez walidacji** — `clf.oob_score_` daje wbudowaną ocenę bez potrzeby osobnego validation setu.
- 💡 **Predykcje probabilistyczne** — `predict_proba()` daje wiarygodność — przydatne do filtrowania niskich confidence.

---

## 📊 Przydatne do Projektu

### Metryki (bezpośrednio do raportu)

Z `classification_report` łatwo wyciągnąć wymagane metryki:
- **Accuracy** — (TP+TN) / All — całkowita poprawność
- **Precision** — TP / (TP+FP) — jak wiele pozytywnych predykcji było poprawnych
- **Recall (Sensitivity)** — TP / (TP+FN) — jak wiele prawdziwych pozytywów złapaliśmy
- **F1-score** — 2 × (Precision × Recall) / (Precision + Recall) — harmoniczna średnia
- **AUC-ROC** — (dla binary classification lub one-vs-rest)

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Raport per klasa
print(classification_report(y_test, y_pred, target_names=class_names))

# AUC dla multi-class
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"AUC: {auc:.4f}")
```

### K-fold Walidacja

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"Fold scores: {scores}")
print(f"Mean ± Std: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Preprocessing do Projektu

Jeśli dane są czarno-białe (jak medyczne obrazy):
```python
X = X / 255.0
X = X.reshape(-1, height, width, 1)  # Dodaj kanał
```

Jeśli RGB:
```python
X = X / 255.0
X = X.reshape(-1, height, width, 3)
```

### Struktura Raportu

Dla dr. Smółki wymagane elementy:
1. **Zadanie** — klasyfikacja obrazów, klasy, rozmiar zbioru
2. **Preprocessing** — resize, normalizacja, augmentacja (jeśli)
3. **Model** — transfer learning (Inception) czy CNN od zera?
4. **Metryki** — tabela z accuracy, precision, recall, F1 per klasa
5. **Macierz pomyłek** — confusion matrix heatmap
6. **Wnioski** — które klasy były trudne, co poprawiło wyniki
7. **K-fold** — średnia ± std z 5-fold CV

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy na przyszłość, NIE jako materiał do raportu dla prowadzącego.

### Aktualność Technik

- **Inception v3 (2015)** — nadal solid na edge devices i szybkie CPU. Ale dla nowych projektów: **MobileNetV3** (2019) — 2× szybciej, podobna acc. **ResNet50** — benchmark standard. **EfficientNet** — best bang for buck.
- **Transfer learning** — best practice 2024. Każdy zaczyna od wstępnie wytrenowanego modelu (ImageNet, CLIP). Trenowanie od zera to dla wielkich korporacji.
- **Fashion MNIST** — praktycznie opuszczony. Dla nauki lepiej: **CIFAR-10** (32×32, 10 klas, bardziej trudne), **STL-10** (większe, ponad 100k), **ImageNet-mini** (1000 klas).

### Nowsze Podejścia (2025)

- **Vision Transformers (ViT)** — zamiast CNN filtry, całe obrazy dzielą się na patche i transformers je przetwarzają. State-of-art, ale powolne na CPU.
- **Foundation Models (CLIP, DINOv2)** — wstępnie wytrenowane na 400M+ obrazów z Internetu. Embeddingi lepsze niż Inception na praktycznie wszystko. Free w HuggingFace.
- **Data augmentation beyond flip/rotate** — mixup (mieszanie obrazów), cutmix (naklejanie fragmentów), RandAugment (losowe transformacje). Zwiększają acc o 2-5%.

### Porównanie: Transfer Learning vs. CNN od Zera

| Aspekt | Transfer Learning | CNN od Zera |
|--------|-------------------|------------|
| Czas trenowania | Sekundy (Random Forest) | Minuty–godziny |
| Wymagane dane | 100–1000 obrazów | 10k+ |
| Dokładność | 85–96% (zależy od zbliżenia klas do ImageNet) | 90–99% (jeśli wystarczająco danych) |
| Hardware | CPU OK | GPU polecana |
| Debug/interpretacja | Feature importance (RF) | Grad-CAM, Saliency Maps (trudne) |

**Rekomendacja**: Zaczynaj od transfer learning (szybko, stabilnie). Jeśli dokładność < 80%, przejdź na CNN z augmentacją.

### Anti-patterns

- ❌ **Trenowanie CNN od zera na <5k obrazów** — 99% overfitting
- ❌ **Brak normalizacji pikseli** — network collapse
- ❌ **Inception v3 bez `preprocess_input`** — śmiecie na wejściu
- ❌ **Random Forest bez validation** — no idea czy model generalny
- ❌ **Fashion MNIST jako "real test"** — za łatwy dataset, nie mówi nic o rzeczywistej wydajności

---

**Koniec notatki Lab 12: Klasyfikacja Obrazów**
