# 📋 Instrukcja dla Subagenta — Notatka z Laba ML

> **Przeczytaj CAŁY ten plik ZANIM zaczniesz pracę.**

---

## CEL

Przeczytaj WSZYSTKIE notebooki Jupyter z przypisanego folderu laba i wygeneruj **jedną szczegółową notatkę markdown**. Notatka ma wyciągać esencję wiedzy — nie kopiować notebooka 1:1.

---

## KONTEKST PROJEKTU

Notatki służą jako:
1. **Baza wiedzy do projektu na zaliczenie** (priorytet #1) — projekt u dr. Smółki wymaga: klasyfikacja/regresja na danych medycznych, metryki (accuracy/F1/AUC/RMSE/MAE), k-fold walidacja, preprocessing, porównanie z literaturą
2. **Quick reference** — szybki lookup technik ML bez otwierania ciężkich `.ipynb`
3. **Dodatkowe źródło wiedzy** — komentarze o aktualności technik i lepszych alternatywach

---

## FORMAT NOTATKI

Użyj DOKŁADNIE tego template:

```markdown
# Lab XX: [Tytuł Laba]

> **Tematyka:** [1-2 zdania: o czym jest lab i do czego służy ta wiedza]
> **Notebooki:** `nazwa1.ipynb`, `nazwa2.ipynb`, ...
> **Kluczowe biblioteki:** sklearn, pandas, ...

---

## TL;DR

[3-5 zdań: absolutna esencja. Co się tu uczymy, jakie techniki, kiedy używać, co najważniejsze.]

---

## Spis Treści

- [Koncept 1: Nazwa](#koncept-1-nazwa)
- [Koncept 2: Nazwa](#koncept-2-nazwa)
- ...

---

## Koncept 1: [Nazwa]

### Co to jest?

[Wyjaśnienie prostymi słowami. 3-5 zdań. Bez akademickiego bełkotu — pisz jakbyś tłumaczył kumplowi.]

### Kiedy używać?

[Typowe use case'y. Kiedy ta technika ma sens, a kiedy nie.]

### Kod

```python
# Minimalny working example — esencja, nie cały notebook
from sklearn.xxx import Yyy

model = Yyy(param=value)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Kluczowe parametry

| Parametr | Domyślnie | Co robi | Kiedy zmieniać |
|----------|-----------|---------|----------------|
| `param1` | `value` | opis | kontekst |

### Gotchas / Tips

- ⚠️ [Pułapka, na którą łatwo wpaść]
- 💡 [Tip przyspieszający pracę]
- 🔑 [Kluczowa rzecz do zapamiętania]

---

[...powtórz strukturę dla każdego konceptu...]

---

## 📊 Przydatne do Projektu

[Co z tego laba BEZPOŚREDNIO przyda się w projekcie na zaliczenie:
- Jakie metryki? (wymagane: accuracy/F1/AUC lub RMSE/MAE + std)
- Jakie techniki preprocessingu?
- Jak to połączyć z k-fold walidacją?
- Co warto pokazać w raporcie?]

---

## 💡 Dodatkowe — Komentarze Agenta

> ⚠️ **Ta sekcja zawiera opinie i komentarze agenta AI** — traktuj jako dodatkowe źródło wiedzy na przyszłość, NIE jako materiał do raportu dla prowadzącego.

- 🔄 **Aktualność:** [Czy techniki z laba są nadal state-of-the-art? Co je zastąpiło?]
- 📚 **Zasoby:** [Polecane artykuły, tutoriale, dokumentacja]
- ⚡ **Nowsze podejścia:** [Trendy 2025-2026, lepsze alternatywy]
- 🆚 **Porównania:** [Jak to wypada vs inne popularne podejścia?]
```

---

## REGUŁY TREŚCI

### ✅ TAK
- Wyjaśniaj jak kumplowi — prosty, praktyczny język
- Wyciągaj esencję z notebooków — najważniejsze koncepty, techniki, parametry
- Podawaj minimalne working examples kodu (nie cały notebook!)
- Każdy koncept: co to? → kiedy używać? → kod → parametry → gotchas
- W sekcji "Przydatne do Projektu" — odwołuj się do wymagań projektu (metryki, walidacja, preprocessing, raport)
- W sekcji "Dodatkowe" — opinie, aktualność technik, lepsze alternatywy
- Pisz po polsku z angielskimi terminami technicznymi
- Bądź szczegółowy — target 300-800 linii per notatka

### ❌ NIE
- NIE kopiuj kodu z notebooka 1:1 — wyciągaj, upraszczaj, komentuj
- NIE pisz w stylu akademickim/formalnym — to notatki, nie paper
- NIE mieszaj opinii agenta z faktami z notebooka — opinie TYLKO w sekcji "Dodatkowe"
- NIE pomijaj sekcji z template — każda jest wymagana
- NIE twórz plików — zwróć PEŁNY tekst notatki jako wiadomość
- NIE używaj vscode_askQuestions — nie masz dostępu do usera

---

## CONSTRAINTS TECHNICZNE

- Zwróć PEŁNY tekst gotowej notatki jako swój output
- Format: markdown
- Język: polski z angielskimi terminami tech
- NIE twórz plików w workspace
- NIE używaj vscode_askQuestions
- NIE spawniaj sub-subagentów
