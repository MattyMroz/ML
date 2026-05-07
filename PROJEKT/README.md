# PROJEKT — Choroby skóry (WUML 2025/2026)

Temat 6: **Diagnostyka różnicowa chorób skóry** metodami uczenia maszynowego.

## Skład zespołu

| # | Imię i nazwisko | Indeks | Zbiór danych | Notebook |
|---|----------------|--------|--------------|----------|
| 1 | Mateusz Mróz | 251190 | UCI Dermatology | `notebooks/mateusz_mroz_uci_dermatology.ipynb` |
| 2 | Dawid Kośka | XXXXXX | DermNet (Kaggle) | `notebooks/dawid_koska_dermnet.ipynb` *(stub)* |
| 3 | Wiktor Grzyb | XXXXXX | Fitzpatrick17k | `notebooks/wiktor_grzyb_fitzpatrick17k.ipynb` *(stub)* |

## Struktura projektu

```
PROJEKT/
├── raport.md                 # główny raport (eksport do PDF przed oddaniem)
├── README.md                 # ten plik
├── data/                     # dane źródłowe (patrz README w każdym podfolderze)
│   ├── uci_dermatology/      # ✅ gotowe (366 próbek, tabularne)
│   ├── dermnet/              # ⏳ do pobrania ręcznie z Kaggle
│   └── fitzpatrick17k/       # ⏳ do pobrania ręcznie z GitHub
├── notebooks/                # notatniki Jupyter — po jednym na osobę
├── results/                  # CSV z wynikami CV i finalnymi metrykami
├── env/
│   └── requirements.txt      # pinowane wersje pakietów
└── DOC/                      # specyfikacja projektu od prowadzącego
```

## Wymagania

- **Python 3.12.x** (sprawdzone z 3.12.12)
- Środowisko wirtualne w korzeniu workspace: `.venv` (zarządzane przez [uv](https://docs.astral.sh/uv/))

### Instalacja (uv — zalecane)

```powershell
uv sync
```

### Instalacja (pip fallback)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r PROJEKT\env\requirements.txt
```

## Uruchomienie

```powershell
# aktywacja środowiska (jeśli pip)
.\.venv\Scripts\Activate.ps1

# uruchomienie Jupyter
jupyter notebook PROJEKT\notebooks\mateusz_mroz_uci_dermatology.ipynb
```

W VS Code: otwórz notebook → wybierz kernel `.venv (Python 3.12.12)`.

## Reprodukowalność

- Wszystkie eksperymenty używają `random_state=42`.
- Podział train/test jest **stratyfikowany** (zachowuje proporcje klas).
- Walidacja: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Wyniki są zapisywane automatycznie do `results/`.

## Źródła danych

| Zbiór | Link | Licencja |
|-------|------|----------|
| UCI Dermatology | https://archive.ics.uci.edu/dataset/33/dermatology | CC BY 4.0 |
| DermNet (Kaggle) | https://www.kaggle.com/datasets/shubhamgoel27/dermnet | educational use |
| Fitzpatrick17k | https://github.com/mattgroh/fitzpatrick17k | CC BY-NC-SA |

Dane **nie są wersjonowane w repo** (rozmiar). Szczegółowe instrukcje pobrania → `data/<nazwa_zbioru>/README.md`.

## Status prac

- ✅ Osoba 1 (Mateusz Mróz) — UCI Dermatology: notebook kompletny, accuracy test 95.95%
- ⏳ Osoba 2 (Dawid Kośka) — DermNet: do zrobienia
- ⏳ Osoba 3 (Wiktor Grzyb) — Fitzpatrick17k: do zrobienia
- ⏳ Raport `raport.md` — szablon gotowy, sekcje osób 2/3 do uzupełnienia
