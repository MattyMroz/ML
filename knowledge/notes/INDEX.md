# INDEX Notatek

Krótki indeks wszystkich notatek z labów ML. Każdy wpis prowadzi do gotowego pliku markdown z TL;DR, kluczowymi konceptami i sekcją praktyczną pod projekt.

## Jak Korzystać

- Zacznij od labów 1-5, jeśli chcesz odświeżyć klasyczny pipeline ML.
- Laby 6-7 są ważne przy problemach rzadkich klas i outlierów.
- Laby 8-14 to bardziej zaawansowane tematy: wizualizacja, ensemble, AutoML, unsupervised, CV, obraz, szeregi i drift.
- Zasady tworzenia notatek są w [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Spis Labów

| Lab | Plik | Główne tematy |
|-----|------|---------------|
| 1 | [Lab01_podstawy_scikit_learn_i_metryki_modeli.md](Lab01_podstawy_scikit_learn_i_metryki_modeli.md) | Estimator API, klasyfikacja, regresja, PCA, K-Means, metryki |
| 2 | [Lab02_interpretacja_modeli_ml.md](Lab02_interpretacja_modeli_ml.md) | Feature importance, współczynniki regresji, SHAP, LIME |
| 3 | [Lab03_redukcja_wymiarowosci.md](Lab03_redukcja_wymiarowosci.md) | PCA, test Bartletta, KMO, ładunki czynnikowe, t-SNE, UMAP |
| 4 | [Lab04_poprawa_modeli_walidacja_krzyzowa.md](Lab04_poprawa_modeli_walidacja_krzyzowa.md) | Hold-out, k-fold, stratification, GroupKFold, TimeSeriesSplit |
| 5 | [Lab05_strojenie_modeli.md](Lab05_strojenie_modeli.md) | GridSearchCV, RandomizedSearchCV, BayesSearchCV, halving, pipeline |
| 6 | [Lab06_dane_niezbalansowane.md](Lab06_dane_niezbalansowane.md) | Recall, F1, AUC, class weights, SMOTE, undersampling, hybrydy |
| 7 | [Lab07_detekcja_anomalii.md](Lab07_detekcja_anomalii.md) | Isolation Forest, LOF, One-Class SVM, fraud detection |
| 8 | [Lab08_wizualizacja_modeli.md](Lab08_wizualizacja_modeli.md) | Learning curves, ROC, PR, DET, residuals, threshold tuning |
| 9 | [Lab09_metody_zespolowe.md](Lab09_metody_zespolowe.md) | Bagging, boosting, stacking, bias-variance trade-off |
| 10 | [Lab10_automl.md](Lab10_automl.md) | H2O AutoML, AutoGluon, MLJAR, logloss |
| 11 | [Lab11_uczenie_nienadzorowane.md](Lab11_uczenie_nienadzorowane.md) | K-Means, DBSCAN, hierarchiczna klasteryzacja, Apriori |
| 12 | [Lab12_klasyfikacja_obrazow.md](Lab12_klasyfikacja_obrazow.md) | CNN, transfer learning, embeddings, Fashion MNIST |
| 13 | [Lab13_szeregi_czasowe.md](Lab13_szeregi_czasowe.md) | dekompozycja, SMA, EWMA, Holt-Winters, SARIMA, Auto-ARIMA |
| 14 | [Lab14_dryft_modelu.md](Lab14_dryft_modelu.md) | data drift, concept drift, KS test, monitoring, retraining |

## Sugerowana Kolejność Nauki

1. [Lab01_podstawy_scikit_learn_i_metryki_modeli.md](Lab01_podstawy_scikit_learn_i_metryki_modeli.md)
2. [Lab04_poprawa_modeli_walidacja_krzyzowa.md](Lab04_poprawa_modeli_walidacja_krzyzowa.md)
3. [Lab05_strojenie_modeli.md](Lab05_strojenie_modeli.md)
4. [Lab02_interpretacja_modeli_ml.md](Lab02_interpretacja_modeli_ml.md)
5. [Lab06_dane_niezbalansowane.md](Lab06_dane_niezbalansowane.md)
6. [Lab07_detekcja_anomalii.md](Lab07_detekcja_anomalii.md)
7. [Lab08_wizualizacja_modeli.md](Lab08_wizualizacja_modeli.md)
8. [Lab09_metody_zespolowe.md](Lab09_metody_zespolowe.md)
9. [Lab10_automl.md](Lab10_automl.md)
10. [Lab11_uczenie_nienadzorowane.md](Lab11_uczenie_nienadzorowane.md)
11. [Lab03_redukcja_wymiarowosci.md](Lab03_redukcja_wymiarowosci.md)
12. [Lab12_klasyfikacja_obrazow.md](Lab12_klasyfikacja_obrazow.md)
13. [Lab13_szeregi_czasowe.md](Lab13_szeregi_czasowe.md)
14. [Lab14_dryft_modelu.md](Lab14_dryft_modelu.md)

## Szybka Mapa Pod Projekt

- Klasyczny model tabelaryczny: Lab 1, 4, 5, 8, 9.
- Interpretowalność i raport: Lab 2, 8.
- Dane niezbalansowane lub rzadkie przypadki: Lab 6, 7.
- Automatyzacja baseline'u: Lab 10.
- Obrazy: Lab 12.
- Dane czasowe: Lab 13.
- Monitoring po wdrożeniu: Lab 14.

## Egzaminowy Core

- Must know: Lab 1, 4, 5, 6, 8, 9.
- Warto umieć wyjaśnić: Lab 2, 3, 10, 11.
- Tematy bardziej specjalistyczne: Lab 12, 13, 14.
