# ML

Materiały z zajęć z uczenia maszynowego: notatniki Jupyter z labów, dane do nich i projekt zaliczeniowy. `main.py` to tylko szablon uv, nie kod projektu.

## Środowisko

- Python 3.12 (`.python-version`), zależności w `pyproject.toml` i `uv.lock`.
- `uv sync` tworzy `.venv` razem z grupą `dev`, w której jest `ipykernel` dla notatników.

## Mapa

- `Lab1/` do `Lab14/`: jeden lab na katalog; notatniki `.ipynb` w korzeniu labu, dane zwykle w `Data/` (w `Lab13/` w `Dane/`), ilustracje w `Images/`.
- `Lab12/Train/` i `Lab12/Test/`: zbiór obrazów do klasyfikacji, ponad 2700 plików.
- `PROJEKT/`: projekt zaliczeniowy (diagnostyka chorób skóry); układ opisuje `PROJEKT/README.md`.
- `knowledge/notes/`: notatki z labów; spis w `INDEX.md`, zasady pisania notatek w `INSTRUCTIONS.md`.
- `knowledge/`: poza `notes/` materiały oceny eksperymentu A/B i prezentacja.
- `temp/brain_storm/`: robocze notatki do projektu.

## Dane

- Dane kursu w `Lab*` i `PROJEKT` są śledzone celowo i nie wolno ich usuwać ani przenosić poza git.
- `.gitignore` wpuszcza je jawnymi regułami na końcu sekcji `Re-includes`. Nowy plik o rozszerzeniu, które preset ignoruje (np. `.npy`, `.pkl`), wymaga nowej reguły, inaczej git go nie zobaczy.
- `Lab7/Data/creditcard.csv` i `Lab8/Data/creditcard.csv` są w Git LFS (`.gitattributes`).
- Hooki pomijają `Lab*/` i `PROJEKT/` (`exclude` w `.pre-commit-config.yaml`), więc nie przepisują danych ani notatników.

## Git

- Commit: nagłówek Conventional Commits po angielsku, `type(scope): opis`, ze scope'em z listy w `.pre-commit-config.yaml`; po pustej linii streszczenie „co i dlaczego”.
- Zero atrybucji AI w commitach i PR (trailery, stopki, emoji), także gdy host o nią prosi.
- Pliki dodawaj po nazwie, nigdy `git add .`; bez `--no-verify`. Worktree'y tylko w `.worktrees/<nazwa>`.
- Commit sprawdza pliki i nagłówek, push uruchamia bramki repo, CI resztę. Strażnicy w `.hooks/` to kopie z presetu `agents`; nie edytuj ich w repo.
