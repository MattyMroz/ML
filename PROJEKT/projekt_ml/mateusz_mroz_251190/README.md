# Mateusz Mroz - UCI Dermatology

**Model:** Random Forest (tuned)  
**Walidacja:** 5-fold StratifiedKFold, `random_state=42`  
**Dataset:** UCI Dermatology, 366 rekordow, 34 cechy, 6 klas chorob  
**Metryki:** accuracy oraz macro-F1

## Dane

Dane sa juz w folderze:

- `data/uci_dermatology/dermatology.data`
- `data/uci_dermatology/dermatology.names`

## Uruchomienie

1. Utworzyc srodowisko Python.
2. Zainstalowac biblioteki z `env/requirements.txt`.
3. Uruchomic `notebooks/mateusz_mroz_uci_dermatology.ipynb`.
4. Wyniki sa zapisane w `results/`.

Najwazniejszy plik wynikowy:

- `results/mateusz_mroz_uci_dermatology_final.csv`
