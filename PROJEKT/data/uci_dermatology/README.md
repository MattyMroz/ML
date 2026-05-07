# UCI Dermatology

**Źródło:** https://archive.ics.uci.edu/dataset/33/dermatology
**Licencja:** CC BY 4.0
**Autorzy:** N. Ilter, H. A. Güvenir (1998)

## Pliki

- `dermatology.data` — 366 wierszy × 35 kolumn (34 cechy + etykieta klasy)
- `dermatology.names` — opis atrybutów i klas

## Pobranie (już zrobione)

```powershell
Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data" `
  -OutFile dermatology.data

Invoke-WebRequest `
  -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.names" `
  -OutFile dermatology.names
```

## Charakterystyka

- **Cel:** klasyfikacja wieloklasowa 6 chorób erytemato-łuskowych
- **Cechy:** 11 klinicznych (skala 0–3) + 22 histopatologiczne (skala 0–3) + wiek
- **Braki:** 8 braków w kolumnie `age` (oznaczone `?`)
- **Niezbalansowanie:** klasa większościowa (psoriasis, 112) / klasa mniejszościowa (pityriasis rubra pilaris, 20) ≈ 5.6×

## Klasy

| # | Nazwa | Liczba |
|---|-------|--------|
| 1 | psoriasis | 112 |
| 2 | seboreic dermatitis | 61 |
| 3 | lichen planus | 72 |
| 4 | pityriasis rosea | 49 |
| 5 | chronic dermatitis | 52 |
| 6 | pityriasis rubra pilaris | 20 |
