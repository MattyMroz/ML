# DermNet (Kaggle mirror)

**Źródło:** https://www.kaggle.com/datasets/shubhamgoel27/dermnet
**Licencja:** educational use (sprawdź na Kaggle przed użyciem komercyjnym)
**Orientacyjny rozmiar:** ~19 500 obrazów, ~1.6 GB po rozpakowaniu

## Jak pobrać (ręcznie — bez Kaggle API)

1. Wejdź na https://www.kaggle.com/datasets/shubhamgoel27/dermnet
2. Zaloguj się (wymagane konto Kaggle)
3. Kliknij **Download** (przycisk w prawym górnym rogu)
4. Rozpakuj ZIP do tego folderu (`PROJEKT/data/dermnet/`)
5. Po rozpakowaniu powinieneś mieć strukturę typu:
   ```
   dermnet/
   ├── train/
   │   ├── Acne and Rosacea Photos/
   │   ├── Actinic Keratosis Basal Cell Carcinoma/
   │   ├── ...
   │   └── Warts Molluscum and other Viral Infections/
   └── test/
       ├── ...
   ```

## Jak pobrać (Kaggle API — opcjonalnie)

Jeśli masz zainstalowany `kaggle` CLI i skonfigurowany `~/.kaggle/kaggle.json`:

```powershell
pip install kaggle
kaggle datasets download -d shubhamgoel27/dermnet -p . --unzip
```

## Charakterystyka

- **Cel:** klasyfikacja obrazów chorób skóry
- **Klasy:** 23 kategorie
- **Wyzwania:** różna jakość zdjęć, różne perspektywy, heterogeniczność

## Użycie

Osoba 2 (Dawid Kośka) — zobacz `../../notebooks/dawid_koska_dermnet.ipynb`.
