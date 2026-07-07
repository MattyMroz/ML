# Wiktor Grzyb - Fitzpatrick17k

**Model:** ResNet-18 transfer learning (PyTorch/torchvision)  
**Walidacja:** 5-fold StratifiedKFold, `random_state=42`  
**Probka danych:** 10% zbioru Fitzpatrick17k (`FRACTION = 0.10`, 1657 rekordow)  
**Metryki:** accuracy oraz weighted-F1

## Dane

Notebook oczekuje nastepujacej struktury:

- `data/fitzpatrick17k/fitzpatrick17k.csv`
- `data/fitzpatrick17k/images/*.jpg`

## Uruchomienie

1. Utworzyc srodowisko Python.
2. Zainstalowac biblioteki z `requirements.txt`.
3. Uruchomic `wiktor_grzyb_fitzpatrick17k.ipynb`.
4. Wyniki walidacji sa zapisane w `results/wiktor_grzyb_fitzpatrick17k_results.csv`.

W notebooku uzywany jest ResNet-18 z wagami ImageNet, zamrozonymi warstwami bazowymi i nowa warstwa `fc` dopasowana do liczby klas.
