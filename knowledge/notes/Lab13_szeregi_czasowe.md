# Lab 13: Szeregi czasowe

> **Tematyka:** Lab dotyczy prognozowania szeregów czasowych — technik analizy danych uporządkowanych w czasie. Obejmuje stacjonarność, sezonowość, trendy, modele ARIMA/SARIMA, średnie ruchome, wygładzanie wykładnicze i dekompozycję. Kluczowy skill do projektów forecasting oraz analizy trendów.
>
> **Notebooki:** `Szeregi czasowe - przykłady.ipynb`, `Zadanie - rozwiązanie.ipynb`
>
> **Kluczowe biblioteki:** pandas, numpy, statsmodels (`holt-winters`, `SARIMAX`, `seasonal_decompose`, `auto_arima`), pmdarima, sklearn

---

## TL;DR

Szereg czasowy to ciąg obserwacji pomierzanych w regularnych interwałach czasowych. Lab uczy trzech głównych zadań: (1) preprocessing (datetime konwersja, indeksowanie), (2) analiza struktury (sezonowość, trend, stacjonarność) oraz (3) prognozowanie. Klucze to: średnie ruchome (SMA/EWMA) dla wygładzania, Holt-Winters dla prostych danych z trendem/sezonowością, SARIMA dla statystycznie zaawansowanej analizy z automatycznym dopasowaniem parametrów. Auto-ARIMA i `seasonal_decompose` to narzędzia diagnostyczne — zmniejszają ręczne eksperymenty.

---

## Spis Treści

- [Koncept 1: Preprocessing szeregów czasowych](#koncept-1-preprocessing-szeregów-czasowych)
- [Koncept 2: Dekompozycja sezonowa](#koncept-2-dekompozycja-sezonowa)
- [Koncept 3: Średnie ruchome (SMA, EWMA)](#koncept-3-średnie-ruchome-sma-ewma)
- [Koncept 4: Wygładzanie wykładnicze (Holt-Winters)](#koncept-4-wygładzanie-wykładnicze-holt-winters)
- [Koncept 5: SARIMA (Seasonal ARIMA)](#koncept-5-sarima-seasonal-arima)
- [Koncept 6: Auto-ARIMA — automatyczne dopasowanie parametrów](#koncept-6-auto-arima--automatyczne-dopasowanie-parametrów)
- [Koncept 7: Diagnostyka — ACF/PACF](#koncept-7-diagnostyka--acfpacf)

---

## Koncept 1: Preprocessing szeregów czasowych

### Co to jest?

Szereg czasowy to realizacja procesu stochastycznego, którego dziedziną jest czas — ciąg informacji uporządkowanych w czasie. Preprocessing to przygotowanie surowych danych: konwersja dat na format datetime, ustawienie daty jako indeksu, określenie częstotliwości (MS = miesiąc, D = dzień). Bez tego pandas nie "rozumie" struktury czasowej i nie możemy używać operacji przesunięcia, resample czy szeregowych dekompozycji.

### Kiedy używać?

**Zawsze na początku**. Każdy szereg czasowy wymaga:
- Konwersji kolumny daty na `datetime64`
- Ustawienia daty jako indeksu (umożliwia `.resample()`, `.shift()`, prognozowanie)
- Upewnienia się co do regularności kroku czasowego
- (Opcjonalnie) resample na różne granularności (np. dzień → miesiąc)

### Kod

```python
import pandas as pd

# Wczytaj dane
df = pd.read_csv('./dane/Monthly_milk_production.csv')

# Konwersja daty
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # lub parse_dates=['Date']

# Ustawianie indeksu
df.set_index('Date', inplace=True)

# Przypisz częstotliwość (ważne dla ARIMA/SARIMA)
df.index.freq = 'MS'  # MS = miesiąc (Month Start), D = dzień, H = godzina

# Resample na wyższy poziom agregacji
monthly = df.resample('MS').sum()  # lub .mean(), .count()
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `parse_dates` | – | Automatycznie konwertuje kolumny na datetime | Zawsze używaj dla kolumn dat |
| `format` | Domyślny parser | Specyfikuje format daty (np. `'%Y-%m-%d'`) | Jeśli parser nie rozpozna formatu |
| `index_col` | – | Ustawia kolumnę jako indeks przy wczytaniu | Oszczędza linijkę kodu w `.set_index()` |
| `df.index.freq` | `None` | Przypisuje częstotliwość do indeksu | **Obowiązkowe** dla ARIMA/SARIMA |

### Gotchas / Tips

- ⚠️ **Bez `.index.freq`** — SARIMA wyrzuci błąd. Zawsze przypisz `'MS'`, `'D'`, `'H'` itp.
- ⚠️ **Brakujące daty** — jeśli seria ma przerwy, `.index.freq` zwróci ostrzeżenie. Użyj `.asfreq()` do interpolacji.
- 💡 **Resample przed modelowaniem** — jeśli masz dane na niskim poziomie granularności (dni), resample do wyższego (miesiące, kwartały) dla lepszej sezonowości.
- 🔑 **DataFrame vs Series** — dla ARIMA/SARIMA często potrzebujesz `df['Column']` (Series), nie całego DataFrame.

---

## Koncept 2: Dekompozycja sezonowa

### Co to jest?

Dekompozycja sezonowa (`seasonal_decompose`) rozkłada szereg czasowy na cztery komponenty:
- **Trend** — długoterminowa trajektoria (rosnąca, malejąca, płaska)
- **Sezonowość** — regularne wzory powtarzające się w cyklu (rok, miesiąc, dzień tygodnia)
- **Residua (szum)** — przypadkowe fluktuacje, które model nie wyjaśnia
- **Obserwowane** — oryginalna seria

Dwa modele:
- **Addytywny** (`model='add'`): Obserwacja = Trend + Sezonowość + Residua
- **Multiplikatywny** (`model='mul'`): Obserwacja = Trend × Sezonowość × Residua

### Kiedy używać?

- **Diagnostyka przed ARIMA** — sprawdzenie, czy są trendy i sezonowość
- **Wizualizacja struktury** — pokazanie, ile zmienności wyjaśnia każdy komponent
- **Preprocessing** — jeśli chcesz detrendować szereg (modelować tylko residua)
- **Nie używaj do prognozowania bezpośrednio** — tylko do analizy i decyzji o parametrach ARIMA

### Kod

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Dekompozycja addytywna (dane bez silnych fluktuacji amplitudy)
result = seasonal_decompose(df['Employees'], model='add', period=12)  # period=12 dla danych miesięcznych

# Wizualizacja
result.plot()  # 4 wykresy: observed, trend, seasonal, residual

# Dostęp do komponentów
trend = result.trend
seasonal = result.seasonal
residuals = result.resid
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `model` | `'add'` | Typ dekompozycji (addytywna vs multiplikatywna) | Addytywna dla małych amplitud, multiplikatywna dla rosnących |
| `period` | – | Długość sezonowego cyklu (12 dla miesięcznych/rocznych, 7 dla dziennych/tygodniowych) | **Wymagane** — dostosuj do danych |
| `extrapolate_trend` | `'freq'` | Czy ekstrapolować trend na brzegach | `'freq'` lub `'zero'` |

### Gotchas / Tips

- ⚠️ **Brakuje parametru `period`** — domyślnie dekompozycja użyje 12. Jeśli masz dane dzienne, ustaw `period=7` (tydzień).
- 💡 **Sprawdzaj wizualnie** — jeśli sezonowość jest "płaska", może oznaczać, że `period` jest źle wybrany.
- 🔑 **Multiplikatywny dla rosnących amplitud** — jeśli szczyty sezonowości rosną z czasem, użyj `model='mul'`.

---

## Koncept 3: Średnie ruchome (SMA, EWMA)

### Co to jest?

**SMA (Simple Moving Average)** — prosta średnia arytmetyczna z ostatnich N obserwacji, przesuwająca się w przód w szeregu.

**EWMA (Exponentially Weighted Moving Average)** — ważona średnia ruchoma, gdzie obserwacje bliższe w czasie mają większą wagę.

Oba wygładzają szum, ujawniając trend.

### Kiedy używać?

- **SMA** — szybkie wygładzanie, interpretacja: średnia z ostatnich X dni
- **EWMA** — płynniejsze przejścia, lepsze dla danych z trendem
- **Nie dla prognozowania** — średnie ruchome to narzędzia wygładzania, nie modelowania przyszłości

### Kod

```python
# SMA — prosta średnia z ostatnich 12 obserwacji
df['SMA12'] = df['EnergyIndex'].rolling(window=12).mean()

# EWMA — ważona średnia
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
df['EWMA12'] = SimpleExpSmoothing(df['EnergyIndex']).fit(smoothing_level=2/(12+1), optimized=False).fittedvalues

# Wizualizacja
df[['EnergyIndex', 'SMA12', 'EWMA12']].plot(figsize=(12, 6))
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `window` (SMA) | – | Rozmiar okna średniej ruchomej | Dłuższe okno = mocniejsze wygładzanie, krótsze = mniej wygładzania |
| `smoothing_level` (EWMA) | Auto | Waga (0–1) dla bieżącej obserwacji | Wyższe = więcej wagi na bieżące dane |

### Gotchas / Tips

- ⚠️ **Sztuczne opóźnienie** — SMA opóźnia trend o około N/2 obserwacji. EWMA mniej, ale też.
- 💡 **Kombinacja: SMA dla wizualizacji, EWMA dla baseline forecast**.
- 🔑 **Nie mieszaj z ARIMA** — średnie ruchome są alternatywą do ARIMA, nie wejściem do niego.

---

## Koncept 4: Wygładzanie wykładnicze (Holt-Winters)

### Co to jest?

**Holt-Winters (ETS — Error, Trend, Seasonal)** to metoda wygładzania, która obsługuje trend i sezonowość jednocześnie:
- **Holt** — trend + poziom
- **Holt-Winters** — trend + poziom + sezonowość

Ma dwa warianty:
- **Addytywny** (`seasonal='add'`) — sezonowość ma stały dodatek co okres
- **Multiplikatywny** (`seasonal='mul'`) — sezonowość jest proporcjonalna

### Kiedy używać?

- **Proste dane z wyraźnym trendem i/lub sezonowością**
- **Szybka prognoza bez głębokich analiz**
- **Krótkie szeregi** (50–200 obserwacji)
- **Nie dla złożonych struktur** — SARIMA jest bardziej elastyczny

### Kod

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Trend + sezonowość (addytywna)
model = ExponentialSmoothing(
    monthly['Accidents'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

# Prognoza na 12 miesięcy
forecast = model.forecast(steps=12)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `trend` | `None` | Typ trendu: `'add'`, `'mul'`, `None` | Zwykle `'add'` |
| `seasonal` | `None` | Typ sezonowości: `'add'`, `'mul'`, `None` | `'add'` dla stałych amplitud, `'mul'` dla rosnących |
| `seasonal_periods` | – | Długość cyklu sezonowego | **Wymagane** jeśli `seasonal != None` |
| `initialization_method` | `'estimated'` | Jak inicjalizować parametry | `'estimated'` jest standardem |

### Gotchas / Tips

- ⚠️ **Wymaga co najmniej 2 pełnych cykli sezonowości**.
- 💡 **Addytywny vs Multiplikatywny** — wybieraj na podstawie amplitudy sezonowości.
- 🔑 **Dobry szybki baseline** przed SARIMA.

---

## Koncept 5: SARIMA (Seasonal ARIMA)

### Co to jest?

**SARIMA(p, d, q)(P, D, Q, m)** to zaawansowany model statystyczny do szeregów czasowych z sezonowością:
- **(p, d, q)** — części ARIMA:
  - **p** — autoregresja
  - **d** — różnicowanie dla stacjonarności
  - **q** — średnia ruchoma błędów
- **(P, D, Q, m)** — części sezonowe
  - **m** — okres sezonowy, np. 12 dla miesięcznych danych

**Stacjonarność** oznacza, że średnia, wariancja i autokorelacja są stałe w czasie. SARIMA wymaga stacjonarności.

### Kiedy używać?

- **Szeregi z trendem i sezonowością**
- **Potrzebujesz statystycznego modelu** z AIC/BIC, confidence intervals i diagnostyką
- **Złożone zależności** — gdy Holt-Winters nie wystarcza

### Kod

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

train = df.iloc[:-12]
test = df.iloc[-12:]

model = SARIMAX(
    train['Employees'],
    order=(0, 1, 0),
    seasonal_order=(2, 0, 0, 12)
)
results = model.fit()

forecast = results.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `order` | – | (p, d, q) | Znajduj via `auto_arima()` |
| `seasonal_order` | – | (P, D, Q, m) | Dopasuj do sezonowości danych |
| `typ` | `'levels'` | Skala predykcji | Używaj `'levels'` dla oryginalnej skali |

### Gotchas / Tips

- ⚠️ **Bez `d` lub `D`** — niestacjonarny szereg będzie źle modelowany.
- 💡 **Auto-ARIMA jest punktem startowym**.
- 🔑 **`typ='levels'`** zwraca prognozy w oryginalnej skali.

---

## Koncept 6: Auto-ARIMA — automatyczne dopasowanie parametrów

### Co to jest?

`auto_arima()` z biblioteki `pmdarima` automatycznie testuje kombinacje parametrów i zwraca najlepszą konfigurację na podstawie AIC/BIC.

### Kiedy używać?

- **Zawsze na początku** jako baseline
- **Gdy nie wiesz, gdzie zacząć**
- **Nie jako ostateczna decyzja** — wynik trzeba zinterpretować

### Kod

```python
from pmdarima import auto_arima

best_model = auto_arima(
    df['Employees'],
    seasonal=True,
    m=12,
    stepwise=True,
    trace=True
)

print(best_model.summary())
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `seasonal` | `False` | Czy szukać parametrów sezonowych | `True` dla danych z sezonowością |
| `m` | 1 | Okres sezonowy | 12 dla danych miesięcznych |
| `stepwise` | `True` | Heurystyczne vs pełne przeszukiwanie | `False` dla małych zbiorów |
| `max_order` | 5 | Maksymalna złożoność modelu | Zwiększ, jeśli standard nie wystarcza |

### Gotchas / Tips

- ⚠️ **Czasochłonne dla dużych zbiorów** — `stepwise=True` oszczędza czas.
- 💡 **Patrz na wynik krytycznie** — auto_arima może zwrócić zbyt złożony model.
- 🔑 **Niższy AIC = lepsze dopasowanie**, ale nie wszystko.

---

## Koncept 7: Diagnostyka — ACF/PACF

### Co to jest?

Narzędzia diagnostyczne do sprawdzenia autokorelacji:
- **ACF (Autocorrelation Function)** — korelacja szeregu z jego opóźnieniami
- **PACF (Partial Autocorrelation Function)** — korelacja po odjęciu wpływu pośrednich opóźnień

### Kiedy używać?

- **Przed ARIMA** — wskazówki dla p i q
- **Po modelowaniu** — ocena, czy residua są white noise
- **Po dekompozycji** — czy została struktura do modelowania

### Kod

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Employees'], lags=40, ax=axes[0], title='ACF')
plot_pacf(df['Employees'], lags=40, ax=axes[1], title='PACF')
plt.tight_layout()
plt.show()
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `lags` | 40 | Ile opóźnień narysować | Dla danych miesięcznych: 40–60 |

### Gotchas / Tips

- ⚠️ **Niebieska linia = 95% confidence interval** — słupki poza nią sugerują istotną autokorelację.
- 💡 **Szczyty co 12 obserwacji** potwierdzają sezonowość miesięczną.
- 🔑 **Po dobrym modelu** residua powinny przypominać biały szum.

---

## 📊 Przydatne do Projektu

Jeśli projekt ma komponent czasowy:
- preprocessing dat,
- wykrywanie sezonowości,
- prognozy Holt-Winters lub SARIMA,
- metryki: RMSE, MAE, MAPE,
- podział train/test na końcowe okno czasu.

Do raportu przydadzą się:
- wykres trendu i sezonowości,
- ACF/PACF jako uzasadnienie parametrów,
- summary modelu z AIC/BIC,
- prognoza z confidence intervals.

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie agenta** — dobry materiał do uzupełniania wiedzy, ale nie materiał do raportu dla prowadzącego.

### Aktualność

- **ARIMA/SARIMA** to nadal mocna podstawa analityczna.
- **Holt-Winters** pozostaje świetnym szybkim baseline'em.
- **Auto-ARIMA** przyspiesza start, ale nie zastępuje myślenia.

### Nowsze podejścia (2024–2026)

- **LSTM / GRU** dla bardziej złożonych zależności.
- **Transformery czasowe** dla state-of-the-art forecastingu.
- **Gradient Boosting z lag features** jako praktyczna alternatywa dla ARIMA.
- **Probabilistic forecasting** dla przedziałów ufności i ryzyka.

### ARIMA vs Holt-Winters

| Aspekt | ARIMA | Holt-Winters |
|--------|-------|--------------|
| Złożoność | Wyższa | Niższa |
| Szybkość treningu | Średnia | Bardzo szybka |
| Interpretacja | Trudniejsza | Prostsza |
| Krótkie szeregi | Czasem gorszy | Często lepszy |
| Diagnostyka | Bardzo dobra | Prostsza |

**Praktyka:** startuj od Holt-Winters, jeśli nie wystarcza przechodź do SARIMA.

---

**Koniec notatki Lab 13: Szeregi czasowe**
