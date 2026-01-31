# 🚀 Quick Start - Klasyfikacja Owoców

## ⚡ 3 kroki do uruchomienia projektu

### Krok 1: Pobierz dataset (5 min)

```bash
# 1. Otwórz w przeglądarce:
https://www.kaggle.com/datasets/moltean/fruits

# 2. Pobierz ZIP (wymaga darmowego konta Kaggle)

# 3. Rozpakuj do:
/Users/thomasross/Projects/wsb/multimedia/data/raw/

# Struktura powinna być:
# data/raw/Training/Apple Braeburn/...
# data/raw/Test/Apple Braeburn/...
```

---

### Krok 2: Uruchom Jupyter (1 min)

```bash
cd /Users/thomasross/Projects/wsb/multimedia
source venv/bin/activate
jupyter notebook
```

**Otworzy się przeglądarka** → wybierz `notebooks/01_eksploracja_danych.ipynb`

---

### Krok 3: Uruchom notebooki kolejno

1. **`01_eksploracja_danych.ipynb`** (5 min)
   - Poznanie datasetu
   - Wizualizacje

2. **`02_ekstrakcja_cech.ipynb`** (30-60 min)
   - Ekstrakcja cech klasycznych
   - **TEN KROK ZAJMUJE NAJWIĘCEJ CZASU!**
   - Możesz użyć podzbioru (USE_SUBSET=True, 20 klas)

3. **`03_klasyfikacja.ipynb`** (10 min)
   - Trenowanie modeli
   - Porównanie wyników

4. **`04_raport_final.ipynb`** (15 min)
   - **TO JEST TWÓJ GŁÓWNY RAPORT!**
   - Użyj go do prezentacji

---

## 📋 Instrukcje szczegółowe

Przeczytaj: **`docs/NEXT_STEPS.md`** - pełny przewodnik krok po kroku

Dokumentacja projektu: **`docs/CURSOR.md`**

---

## ⏰ Szacowany czas

- **Setup + Notebook 01**: 30 min
- **Ekstrakcja cech (Notebook 02)**: 1-2 godziny (z full dataset) lub 30 min (subset)
- **Klasyfikacja (Notebook 03)**: 30 min
- **Raport (Notebook 04)**: 1 godzina
- **Prezentacja**: 2-3 godziny przygotowanie

**Łącznie**: 1 weekend pracy

---

## Problemy?

### "ModuleNotFoundError: No module named 'cv2'"
```bash
source venv/bin/activate  # Aktywuj środowisko!
```

### "Dataset nie znaleziony"
Sprawdź czy rozpakowałeś ZIP do `data/raw/` i czy są foldery `Training/` i `Test/`

### "Za długo trwa ekstrakcja"
W notebook 02 ustaw:
```python
USE_SUBSET = True
SUBSET_SIZE = 20  # Zamiast 131 klas
```

### "Niska accuracy"
- Użyj więcej klas (min 20)
- Sprawdź czy preprocessing działa
- Tune hyperparameters

---

## Status projektu

- [x] Struktura projektu
- [x] Dokumentacja (`docs/CURSOR.md`)
- [x] Kod modułów (`src/`)
- [x] Środowisko zainstalowane (`venv/`)
- [x] Notebook 01 gotowy
- [ ] **Pobierz dataset**
- [ ] Uruchom notebooks 02-04
- [ ] Przygotuj prezentację

---

## 🎓 Prezentacja (ostatnie zajęcia)

**Czas**: 10 minut (2 osoby po 5 min)

**Materiały**: Notebook 04 lub slajdy

**Co pokazać**:
1. Problem: Czy CNN konieczne?
2. Metody klasyczne (cechy)
3. Wyniki (tabela, wykresy)
4. Wnioski

---


