# 🍎 Projekt: Klasyfikacja Owoców - Metody Klasyczne vs CNN

## 📌 Informacje podstawowe

**Temat**: Sieci konwolucyjne nie są rozwiązaniem na wszystko - klasyczna analiza obrazów owoców

**Cel**: Udowodnić, że klasyczne metody przetwarzania obrazów i ekstrakcji cech mogą skutecznie klasyfikować owoce bez używania CNN (Convolutional Neural Networks)

**Grupa**: 2 osoby

**Dataset**: Fruits 360 - podzbiór (48 klas, ~47,000 obrazów)
- 3 kategorie: Apple (22), Cherry (12), Tomato (14)
- 34,800 train + 12,233 test obrazów

---

## 🎯 Założenia projektu

### ❌ ZABRONIONE:
- Sieci neuronowe (CNN, ANN, Deep Learning)
- Transfer learning z gotowych modeli
- Automatyczna ekstrakcja cech przez sieci

### ✅ DOZWOLONE:
- Klasyczne metody przetwarzania obrazów
- Ręczna ekstrakcja cech (kształt, kolor, tekstura)
- Klasyczne algorytmy ML (SVM, Random Forest, k-NN, Decision Tree)
- Biblioteki: OpenCV, scikit-image, scikit-learn

---

## 📊 Dataset: Fruits 360 (podzbiór)

**Źródło**: Kaggle - Fruits 360 Dataset
- **Link**: https://www.kaggle.com/datasets/moltean/fruits

**Charakterystyka używanego podzbioru**:
- **48 klas** owoców i warzyw (podzbiór z 131 dostępnych)
- **47,033 obrazów** łącznie
  - **34,800 obrazów treningowych**
  - **12,233 obrazów testowych**
- Rozmiar: 100x100 pikseli
- Czyste białe tło
- Różne kąty obrotu tego samego owocu
- Format: JPG

**Statystyki (z notebooka 01_eksploracja_danych_v2.ipynb)**:
- Min obrazów na klasę (train): 304
- Max obrazów na klasę (train): 984
- Średnia obrazów na klasę: 725.0

**Kategorie w projekcie (3 główne)**:
1. **Apple** (22 odmiany):
   - Apple 5, 7, 8, 9, 11, 12, 13, 14, 17, 18
   - Apple Braeburn, Crimson Snow, Golden 2, Golden 3
   - Apple Pink Lady, Red 2, Red 3, Red Delicious, Red Yellow 2
   - Apple Rotten, hit, worm

2. **Cherry** (12 odmian):
   - Cherry 1, 2, 4, 5
   - Cherry Rainier 2, Rainier 3, Sour 1
   - Cherry Wax Black 1, Wax Red 1, Wax Red 2, Wax Yellow 1
   - Cherry Wax not ripen 2

3. **Tomato** (14 odmian):
   - Tomato 2, 3, 4, 5, 8, 9, 10
   - Tomato Cherry Maroon 1, Cherry Orange 1, Cherry Red 2, Cherry Yellow 1
   - Tomato Heart 1, Maroon 2, not Ripen 1

**Uwaga**: Projekt używa podzbioru dla efektywności i szybszego przetwarzania. Pełny dataset (131 klas) można pobrać z Kaggle.

---

## 🔬 Metodologia

### 1. Preprocessing (Przetwarzanie wstępne)
```python
# Kroki:
1. Wczytanie obrazu
2. Usunięcie białego tła (segmentacja)
3. Normalizacja rozmiaru
4. Konwersja przestrzeni kolorów (RGB → HSV, LAB)
```

### 2. Ekstrakcja cech (Feature Extraction)

#### A. **Cechy koloru** (Color Features)
- **Histogram RGB**: rozkład intensywności w kanałach R, G, B
- **Histogram HSV**: odcień (Hue), nasycenie (Saturation), wartość (Value)
- **Momenty koloru**: średnia, odchylenie standardowe, skewness, kurtosis
- **Dominujący kolor**: k-means clustering w przestrzeni kolorów

#### B. **Cechy kształtu** (Shape Features)
- **Area**: pole powierzchni owocu
- **Perimeter**: obwód konturu
- **Circularity**: 4π × Area / Perimeter² (miara okrągłości)
- **Aspect Ratio**: stosunek szerokości do wysokości
- **Extent**: stosunek obszaru obiektu do obszaru bounding box
- **Solidity**: stosunek obszaru do convex hull area
- **Momenty Hu**: 7 niezmienników momentów (invariant to translation, rotation, scale)

#### C. **Cechy tekstury** (Texture Features)
- **LBP** (Local Binary Patterns): histogram LBP
- **GLCM** (Gray-Level Co-occurrence Matrix):
  - Contrast (kontrast)
  - Correlation (korelacja)
  - Energy (energia)
  - Homogeneity (jednorodność)
- **Haralick features**: 13 cech tekstury z GLCM

#### D. **Inne cechy**
- **HOG** (Histogram of Oriented Gradients) - opcjonalnie
- **Edge density**: gęstość krawędzi (Canny)

### 3. Klasyfikacja (Classification)

Porównanie klasycznych algorytmów:

#### **SVM** (Support Vector Machine)
- Kernel: RBF, Linear, Polynomial
- Hyperparameters: C, gamma

#### **Random Forest**
- Ensemble metoda
- Hyperparameters: n_estimators, max_depth, min_samples_split

#### **k-NN** (k-Nearest Neighbors)
- Hyperparameters: k (liczba sąsiadów), metric (euclidean, manhattan)

#### **Decision Tree**
- Hyperparameters: max_depth, min_samples_leaf

### 4. Ewaluacja

**Metryki**:
- Accuracy (dokładność)
- Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- ROC curves (dla binary classification)

**Cross-validation**: 5-fold lub 10-fold

---

## 📁 Struktura projektu

```
multimedia/
├── docs/
│   ├── CURSOR.md              # Ten plik - dokumentacja projektu
│   └── prezentacja.md         # Notatki do prezentacji końcowej
├── data/
│   ├── raw/                   # Surowe obrazy Fruits 360
│   │   ├── Training/          # Zbiór treningowy
│   │   └── Test/              # Zbiór testowy
│   └── processed/             # Przetworzone dane (CSV z cechami)
│       ├── features_train.csv
│       └── features_test.csv
├── notebooks/
│   ├── 01_eksploracja_danych.ipynb      # EDA - wersja 1
│   ├── 01_eksploracja_danych_v2.ipynb   # ✅ GŁÓWNY NOTEBOOK - EDA z komentarzami i źródłami
│   ├── 02_ekstrakcja_cech.ipynb         # TODO: Implementacja ekstrakcji cech
│   ├── 03_klasyfikacja.ipynb            # TODO: Trenowanie i porównanie modeli
│   └── 04_raport_final.ipynb            # TODO: GŁÓWNY RAPORT do prezentacji
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Funkcje przetwarzania obrazów
│   ├── feature_extraction.py  # Funkcje ekstrakcji cech
│   └── classification.py      # Funkcje klasyfikacji i ewaluacji
├── results/
│   ├── figures/              # Wykresy, wizualizacje
│   │   ├── eda/
│   │   ├── features/
│   │   └── models/
│   └── models/               # Zapisane wytrenowane modele
│       ├── svm_model.pkl
│       ├── rf_model.pkl
│       └── knn_model.pkl
├── .gitignore
├── requirements.txt          # Zależności Python
└── README.md                # Instrukcja instalacji i uruchomienia
```

---

## 🚀 Instalacja i uruchomienie

### 1. Klonowanie/Setup projektu
```bash
cd /Users/thomasross/Projects/wsb/multimedia
```

### 2. Utworzenie środowiska wirtualnego
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 4. Pobranie datasetu Fruits 360

**Opcja A - Kaggle API** (wymaga konta Kaggle):
```bash
pip install kaggle
kaggle datasets download -d moltean/fruits
unzip fruits.zip -d data/raw/
```

**Opcja B - Ręcznie**:
1. Pobierz z: https://www.kaggle.com/datasets/moltean/fruits
2. Rozpakuj do `data/raw/`
3. Struktura:
   ```
   data/raw/
   ├── Training/
   │   ├── Apple Braeburn/
   │   ├── Apple Golden 1/
   │   └── ...
   └── Test/
       ├── Apple Braeburn/
       ├── Apple Golden 1/
       └── ...
   ```

### 5. Uruchomienie notebooków
```bash
jupyter notebook
```

Otwórz notebooki w kolejności:
1. `01_eksploracja_danych_v2.ipynb` ← **✅ UKOŃCZONY - z dokumentacją i źródłami**
2. `02_ekstrakcja_cech.ipynb` ← TODO
3. `03_klasyfikacja.ipynb` ← TODO
4. `04_raport_final.ipynb` ← TODO (główny raport)

---

## 📈 Workflow projektu

### Krok 1: Eksploracja danych (EDA)
- Wczytanie przykładowych obrazów
- Wizualizacja różnych klas
- Analiza rozkładu klas
- Sprawdzenie rozmiaru datasetu

### Krok 2: Ekstrakcja cech
- Implementacja funkcji do ekstrakcji cech
- Przetworzenie wszystkich obrazów train/test
- Zapisanie cech do CSV
- Analiza ważności cech

### Krok 3: Klasyfikacja
- Trenowanie różnych klasyfikatorów
- Hyperparameter tuning (Grid Search)
- Porównanie wyników
- Wizualizacja confusion matrix

### Krok 4: Raport końcowy
- Podsumowanie wyników
- Najlepszy model
- Wnioski: czy CNN są konieczne?
- Przygotowanie do prezentacji

---

## 📊 Oczekiwane wyniki

### Hipotezy:
1. **SVM z cechami koloru i kształtu** powinien osiągnąć >80% accuracy
2. **Random Forest** może dać najlepsze wyniki (~85-90%)
3. **Cechy koloru** będą najważniejsze (różne owoce = różne kolory)
4. **Cechy kształtu** pomogą rozróżnić jabłka od bananów
5. **Tekstura** może być mniej istotna przy białym tle

### Benchmark bez CNN:
- **Target accuracy**: >85% na 48 klasach
- **Baseline**: Random guess = 1/48 ≈ 2.08%

### Porównanie z CNN:
- Typowa CNN na Fruits 360: ~98-99% accuracy
- **Nasza metoda**: pokaże, że można osiągnąć ~85-90% bez CNN!

### Analiza zbalansowania klas:
- Minimum: 304 obrazów na klasę
- Maximum: 984 obrazów na klasę  
- Średnia: 725 obrazów na klasę
- Dataset jest stosunkowo zbalansowany

---

## 🎤 Prezentacja (ostatnie zajęcia)

### Plan prezentacji (2 osoby):

#### Osoba 1 (5 min):
1. **Wprowadzenie**:
   - Problem: Czy CNN są konieczne?
   - Dataset: Fruits 360
   - Metodologia: klasyczne metody

2. **Ekstrakcja cech**:
   - Demonstracja cech na przykładach
   - Wizualizacje: histogramy kolorów, kontury, tekstury

#### Osoba 2 (5 min):
3. **Klasyfikacja i wyniki**:
   - Porównanie algorytmów (SVM, RF, k-NN)
   - Confusion matrix
   - Najlepszy model

4. **Wnioski**:
   - Osiągnięta accuracy
   - Kiedy CNN są konieczne, a kiedy nie?
   - Zalety metod klasycznych: interpretowalność, szybkość

**Materiały do prezentacji**:
- Jupyter Notebook: `04_raport_final.ipynb`
- Eksport do HTML/PDF z wynikami
- Slajdy (opcjonalnie): PowerPoint z kluczowymi wizualizacjami

---

## 🛠️ Użyte technologie

### Języki i narzędzia:
- **Python 3.8+**
- **Jupyter Notebook**

### Biblioteki:
- **OpenCV** (`cv2`): przetwarzanie obrazów
- **scikit-image**: segmentacja, ekstrakcja cech
- **scikit-learn**: klasyfikacja, metryki
- **NumPy, Pandas**: operacje na danych
- **Matplotlib, Seaborn**: wizualizacje
- **Pillow (PIL)**: obsługa obrazów

---

## 📚 Literatura i zasoby

### Dataset:
- Fruits 360: https://www.kaggle.com/datasets/moltean/fruits
- Paper: Mureșan, H., & Oltean, M. (2018). "Fruit recognition from images using deep learning"

### Metody klasyczne:
- **Hu Moments**: M. K. Hu (1962). "Visual pattern recognition by moment invariants"
- **LBP**: Ojala et al. (2002). "Multiresolution gray-scale and rotation invariant texture classification"
- **GLCM**: Haralick et al. (1973). "Textural Features for Image Classification"

### Tutorials:
- OpenCV Documentation: https://docs.opencv.org/
- scikit-image Examples: https://scikit-image.org/docs/stable/auto_examples/
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html

---

## ✅ Checklist projektu

### Przed prezentacją:
- [x] Pobranie i rozpakownie datasetu Fruits 360 (podzbiór 48 klas)
- [x] Ukończenie notebooka 01: Eksploracja danych (`01_eksploracja_danych_v2.ipynb`)
  - [x] Analiza struktury datasetu
  - [x] Wizualizacja przykładowych obrazów
  - [x] Preprocessing (usuwanie tła)
  - [x] Histogramy kolorów (RGB, HSV)
  - [x] CLAHE i sharpening
  - [x] LBP (Local Binary Patterns)
  - [x] Ekstrakcja cech kształtu
  - [x] Data augmentation
  - [x] Przestrzenie kolorów (RGB, HSV, LAB)
  - [x] Dodanie komentarzy z dokumentacją i źródłami
- [ ] Ukończenie notebooka 02: Ekstrakcja cech
- [ ] Ukończenie notebooka 03: Klasyfikacja
- [ ] Ukończenie notebooka 04: Raport końcowy
- [ ] Eksport raportu do HTML/PDF
- [ ] Przygotowanie slajdów (opcjonalnie)
- [ ] Próba prezentacji (timing 10 min)
- [ ] Zapisanie najlepszych modeli

### Podczas prezentacji:
- [ ] Pokazanie datasetu
- [ ] Demonstracja ekstrakcji cech na żywo
- [ ] Prezentacja wyników klasyfikacji
- [ ] Odpowiedzi na pytania prowadzącego

---

## 🎓 Kryteria oceny (przewidywane)

1. **Poprawność metodologiczna** (30%):
   - Czy użyto tylko metod klasycznych?
   - Czy ekstrakcja cech jest poprawna?

2. **Wyniki** (30%):
   - Osiągnięta accuracy
   - Porównanie różnych metod

3. **Prezentacja** (20%):
   - Jasność przekazu
   - Wizualizacje
   - Czas (10 min)

4. **Raport/Kod** (20%):
   - Jakość kodu
   - Dokumentacja
   - Reprodukowalność

---

## 💡 Wskazówki

### Do ekstrakcji cech:
- Użyj `cv2.findContours()` do wyznaczenia konturów
- `cv2.moments()` dla momentów obrazu
- `skimage.feature.greycomatrix()` dla GLCM
- Normalizuj cechy przed klasyfikacją!

### Do klasyfikacji:
- Użyj `StandardScaler` przed SVM/k-NN
- Grid Search dla hyperparameters
- Zapisuj modele: `joblib.dump(model, 'model.pkl')`

### Wizualizacje:
- Confusion matrix: `plot_confusion_matrix()`
- Feature importance dla Random Forest
- t-SNE dla wizualizacji cech w 2D

---

## 🐛 Rozwiązywanie problemów

### Problem: Białe tło zakłóca cechy koloru
**Rozwiązanie**: Segmentacja - usuń białe tło przed ekstrakcją cech

### Problem: Za dużo cech (overfitting)
**Rozwiązanie**: Feature selection (SelectKBest, PCA)

### Problem: Niezbalansowane klasy
**Rozwiązanie**: 
- `class_weight='balanced'` w SVM/RF
- SMOTE (oversampling)

### Problem: Długi czas trenowania
**Rozwiązanie**: 
- Użyj podzbioru danych na początku (np. 10 klas)
- Zmniejsz liczbę cech

---

## 📞 Kontakt i współpraca

**Członkowie zespołu**:
- Osoba 1: [Imię] - [email/kontakt]
- Osoba 2: Andrii - 98598

**Podział zadań**:
- Osoba 1: Preprocessing + Ekstrakcja cech koloru/kształtu
- Osoba 2: Ekstrakcja cech tekstury + Klasyfikacja

**Termin prezentacji**: Ostatnie zajęcia laboratoryjne

---

## 🔄 Historia zmian

- **2026-01-18**: 
  - ✅ Ukończono notebook `01_eksploracja_danych_v2.ipynb`
  - Dodano komentarze z dokumentacją i źródłami do wszystkich komórek
  - Zaktualizowano README.md i CURSOR.md do rzeczywistych statystyk (48 klas, 47,033 obrazów)
  - Zaimplementowano: preprocessing, segmentację, CLAHE, LBP, shape features, augmentację
- **2025-11-09**: Utworzenie projektu, wybór datasetu Fruits 360, struktura katalogów

---

**Powodzenia! 🍀**

