# Fitzpatrick17k

**Źródło:** https://github.com/mattgroh/fitzpatrick17k
**Publikacja:** Groh et al., *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*, CVPR 2021 Workshop
**Licencja:** CC BY-NC-SA 4.0 (non-commercial, share-alike)
**Orientacyjny rozmiar:** 16 577 obrazów + metadane CSV

## Jak pobrać (ręcznie)

1. Sklonuj repo GitHub: https://github.com/mattgroh/fitzpatrick17k
   ```powershell
   git clone https://github.com/mattgroh/fitzpatrick17k.git
   ```
2. W repo jest plik `fitzpatrick17k.csv` z URL-ami do obrazów
3. Użyj skryptu `download_images.py` z repo (wymaga `requests`)
4. Skopiuj/zsymlinkuj wynikowy folder `images/` tutaj

Alternatywnie — niektórzy utrzymują kopie na Kaggle. Sprawdź:
https://www.kaggle.com/datasets/maxwellmoelis/fitzpatrick17k

## Struktura docelowa

```
fitzpatrick17k/
├── fitzpatrick17k.csv          # metadane (URL, choroba, typ skóry 1-6)
└── images/
    ├── <hash1>.jpg
    ├── <hash2>.jpg
    └── ...
```

## Charakterystyka

- **Cel:** klasyfikacja chorób skóry z uwzględnieniem typu skóry Fitzpatrick'a (1–6)
- **Klasy:** 114 chorób (można agregować do 3 kategorii: benign, malignant, non-neoplastic)
- **Kluczowa cecha:** etykietowanie typu skóry → pozwala analizować bias modelu względem koloru skóry

## Użycie

Osoba 3 (Wiktor Grzyb) — zobacz `../../notebooks/wiktor_grzyb_fitzpatrick17k.ipynb`.
