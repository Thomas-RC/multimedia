# 📝 Następne kroki - Dokończenie projektu

## ✅ Co zostało utworzone:

1. ✅ Struktura katalogów projektu
2. ✅ **docs/CURSOR.md** - Pełna dokumentacja projektu
3. ✅ **README.md** - Instrukcja instalacji
4. ✅ **requirements.txt** - Wszystkie zależności (ZAINSTALOWANE!)
5. ✅ **src/preprocessing.py** - Moduł preprocessingu obrazów
6. ✅ **src/feature_extraction.py** - Moduł ekstrakcji cech
7. ✅ **src/classification.py** - Moduł klasyfikacji
8. ✅ **notebooks/01_eksploracja_danych.ipynb** - Gotowy notebook EDA
9. ✅ Środowisko wirtualne `venv/` z zainstalowanymi pakietami

---

## 🎯 Co musisz jeszcze zrobić:

### 1. Pobrać dataset Fruits 360

**Krok 1**: Pobierz dataset
- Link: https://www.kaggle.com/datasets/moltean/fruits
- Pobierz ZIP (może wymagać konta Kaggle - darmowe)

**Krok 2**: Rozpakuj do projektu
```bash
cd /Users/thomasross/Projects/wsb/multimedia/data/raw/
# Tutaj rozpakuj zawartość ZIP
# Powinna powstać struktura:
# data/raw/
#   ├── Training/
#   └── Test/
```

---

### 2. Uruchomić Jupyter Notebook

```bash
cd /Users/thomasross/Projects/wsb/multimedia
source venv/bin/activate
jupyter notebook
```

Przejdź do `notebooks/01_eksploracja_danych.ipynb` i uruchom wszystkie komórki!

---

### 3. Utworzyć pozostałe notebooki

#### Notebook 02: Ekstrakcja cech

Utworzyłem dla Ciebie szablon. Utwórz plik `notebooks/02_ekstrakcja_cech.ipynb` i dodaj:

**Główne sekcje**:
1. Import modułów
2. Demonstracja ekstrakcji na 1 obrazie
3. Wizualizacja cech (histogramy, kształty, LBP)
4. Pętla po wszystkich obrazach
5. Zapis do CSV (`data/processed/features_train.csv`, `features_test.csv`)

**Kod - główna funkcja ekstrakcji**:
```python
from preprocessing import load_and_preprocess, get_all_image_paths, extract_label_from_path
from feature_extraction import extract_all_features, get_feature_names
import pandas as pd
from tqdm import tqdm

def extract_dataset_features(split='Training', use_subset=True, n_classes=20):
    """Ekstrahuje cechy z datasetu"""
    data_dir = Path(f'../data/raw/{split}')
    
    # Pobierz klasy
    all_classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if use_subset:
        selected_classes = all_classes[:n_classes]
    else:
        selected_classes = all_classes
    
    features_list = []
    labels = []
    
    for class_name in selected_classes:
        class_dir = data_dir / class_name
        for img_path in tqdm(list(class_dir.glob('*.jpg')), desc=class_name):
            try:
                img = load_and_preprocess(str(img_path))
                features = extract_all_features(img)
                features_list.append(features)
                labels.append(class_name)
            except:
                continue
    
    # DataFrame
    feature_names = get_feature_names()
    df = pd.DataFrame(features_list, columns=feature_names)
    df['label'] = labels
    
    return df

# Użyj:
df_train = extract_dataset_features('Training', use_subset=True, n_classes=20)
df_train.to_csv('../data/processed/features_train.csv', index=False)

df_test = extract_dataset_features('Test', use_subset=True, n_classes=20)
df_test.to_csv('../data/processed/features_test.csv', index=False)
```

---

#### Notebook 03: Klasyfikacja

Utwórz `notebooks/03_klasyfikacja.ipynb`:

**Główne sekcje**:
1. Wczytanie cech z CSV
2. Preprocessing (StandardScaler, train/test split)
3. Trenowanie modeli:
   - SVM
   - Random Forest
   - k-NN
   - Decision Tree
4. Porównanie wyników
5. Confusion matrix
6. Feature importance (dla RF)
7. Zapis najlepszego modelu

**Przykładowy kod**:
```python
from classification import (
    prepare_data,
    get_svm_classifier,
    get_random_forest_classifier,
    get_knn_classifier,
    train_and_evaluate,
    plot_confusion_matrix,
    plot_model_comparison,
    save_model
)

# Wczytaj dane
df_train = pd.read_csv('../data/processed/features_train.csv')
df_test = pd.read_csv('../data/processed/features_test.csv')

X_train = df_train.drop(['label', 'image_path'], axis=1, errors='ignore')
y_train = df_train['label']
X_test = df_test.drop(['label', 'image_path'], axis=1, errors='ignore')
y_test = df_test['label']

# Przygotuj dane
X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, scaler, le = prepare_data(
    X_train.values, y_train.values
)

# Trenuj modele
results = {}

# SVM
svm_model = get_svm_classifier(kernel='rbf', C=10)
results['SVM'] = train_and_evaluate(svm_model, X_train_scaled, y_train_enc, 
                                     X_test_scaled, y_test_enc, 'SVM')

# Random Forest
rf_model = get_random_forest_classifier(n_estimators=100, max_depth=20)
results['Random Forest'] = train_and_evaluate(rf_model, X_train_scaled, y_train_enc,
                                               X_test_scaled, y_test_enc, 'Random Forest')

# k-NN
knn_model = get_knn_classifier(n_neighbors=5)
results['k-NN'] = train_and_evaluate(knn_model, X_train_scaled, y_train_enc,
                                      X_test_scaled, y_test_enc, 'k-NN')

# Porównaj
plot_model_comparison(results, metric='test_accuracy')

# Confusion matrix dla najlepszego
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_result = results[best_model_name]
plot_confusion_matrix(y_test_enc, best_result['y_pred_test'], 
                     labels=le.classes_, title=f'Confusion Matrix - {best_model_name}')

# Zapisz najlepszy model
save_model(best_result['model'], f'../results/models/{best_model_name.lower()}_model.pkl',
          scaler, le)
```

---

#### Notebook 04: Raport Final (GŁÓWNY RAPORT)

To będzie Twój **główny raport do prezentacji**!

Utwórz `notebooks/04_raport_final.ipynb`:

**Struktura raportu**:

1. **Wprowadzenie**
   - Cel projektu
   - Teza: CNN nie są konieczne
   - Dataset Fruits 360

2. **Metodologia**
   - Klasyczne metody przetwarzania obrazów
   - Ekstrahowane cechy:
     - Kolor (histogramy RGB/HSV, momenty)
     - Kształt (area, perimeter, Hu moments)
     - Tekstura (LBP, GLCM)

3. **Eksperymenty**
   - Dataset: X klas, Y obrazów
   - Cechy: Z features
   - Modele: SVM, Random Forest, k-NN, Decision Tree

4. **Wyniki**
   - Tabela z accuracy dla każdego modelu
   - Wykresy porównawcze
   - Confusion matrix najlepszego modelu
   - Feature importance

5. **Wnioski**
   - Osiągnięta accuracy: ~85-90%?
   - Porównanie z CNN (literatura: ~98%)
   - **Wnioski**:
     - Klasyczne metody są wystarczające dla prostych zadań
     - CNN potrzebne gdy: złożone tła, różne skale, rotacje, occlusion
     - Zalety klasycznych: interpretowalność, szybkość, mniej danych
     - Wady: ręczne feature engineering, gorsze dla złożonych danych

6. **Podsumowanie prezentacji**

---

## 📊 Przykładowe wyniki (do raportu)

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|---------------|-----------|--------|----------|
| SVM | 0.92 | 0.87 | 0.86 | 0.87 | 0.86 |
| **Random Forest** | **0.95** | **0.89** | **0.88** | **0.89** | **0.88** |
| k-NN | 0.88 | 0.83 | 0.82 | 0.83 | 0.82 |
| Decision Tree | 0.89 | 0.75 | 0.74 | 0.75 | 0.74 |

**Najlepszy model**: Random Forest - 89% accuracy

---

## 🎤 Przygotowanie prezentacji

### Slajdy (opcjonalnie - lub tylko Jupyter Notebook)

1. **Slajd 1**: Tytuł
   - "Klasyfikacja Owoców - Metody Klasyczne vs CNN"
   - Autorzy, data

2. **Slajd 2**: Problem
   - Czy CNN są jedynym rozwiązaniem?
   - Dataset: Fruits 360 (131 klas)

3. **Slajd 3**: Metodologia
   - Wykres: preprocessing → feature extraction → classification
   - Typy cech: kolor, kształt, tekstura

4. **Slajd 4**: Wizualizacje cech
   - Przykładowy owoc
   - Histogramy, kontury, LBP

5. **Slajd 5**: Wyniki
   - Tabela accuracy
   - Wykres słupkowy porównania modeli

6. **Slajd 6**: Confusion Matrix
   - Najlepszy model

7. **Slajd 7**: Wnioski
   - Klasyczne metody: 85-90%
   - CNN: ~98% (literatura)
   - Kiedy klasyczne są OK, kiedy CNN konieczne

8. **Slajd 8**: Podsumowanie
   - Q&A

**LUB** wyeksportuj notebook 04 do HTML:
```bash
jupyter nbconvert --to html notebooks/04_raport_final.ipynb
```

---

## ⏰ Timeline

**Tydzień 1**:
- [x] Setup projektu
- [ ] Pobrać dataset
- [ ] Uruchomić notebook 01
- [ ] Utworzyć notebook 02 i wyekstrahować cechy

**Tydzień 2**:
- [ ] Utworzyć notebook 03 i wytrenować modele
- [ ] Utworzyć notebook 04 (raport)

**Tydzień 3**:
- [ ] Przygotować prezentację
- [ ] Przećwiczyć timing (10 min)
- [ ] Prezentacja na zajęciach

---

## 🆘 Troubleshooting

### Problem: Dataset zbyt duży, ekstrakcja trwa za długo
**Rozwiązanie**: Użyj podzbioru klas
```python
USE_SUBSET = True
SUBSET_SIZE = 20  # Zamiast wszystkich 131 klas
```

### Problem: Za mało RAM
**Rozwiązanie**: Przetwarzaj partiami i zapisuj do CSV częściami

### Problem: Niska accuracy (<70%)
**Rozwiązanie**: 
- Sprawdź czy preprocessing działa (białe tło usunięte?)
- Użyj więcej klas treningowych
- Tune hyperparameters (Grid Search)

---

## 📚 Dodatkowe zasoby

- OpenCV Tutorial: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- scikit-learn Guide: https://scikit-learn.org/stable/user_guide.html
- Fruits 360 Paper: https://arxiv.org/abs/1712.00580

---

## ✅ Checklist przed prezentacją

- [ ] Wszystkie 4 notebooki działają
- [ ] Wyniki zapisane (modele, wykresy)
- [ ] Raport/prezentacja gotowa
- [ ] Próba prezentacji (timing 10 min)
- [ ] Odpowiedzi na pytania przygotowane

---

**Powodzenia! 🍀**

Jeśli masz pytania, sprawdź `docs/CURSOR.md` lub dokumentację w kodzie.

