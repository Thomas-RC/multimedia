#  Klasyfikacja Owoców - Metody Klasyczne vs CNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)

**Projekt laboratoryjny**: Udowodnienie, że sieci konwolucyjne (CNN) nie są jedynym rozwiązaniem dla klasyfikacji obrazów.

##  Opis projektu

Ten projekt demonstruje, że **klasyczne metody przetwarzania obrazów** mogą skutecznie klasyfikować owoce bez używania głębokich sieci neuronowych (CNN). Wykorzystujemy podzbiór datasetu **Fruits 360** (48 klas, ~47,000 obrazów) i ekstrakcję cech takich jak:

-  **Cechy koloru** (histogramy RGB/HSV, momenty kolorów)
-  **Cechy kształtu** (momenty Hu, circularity, aspect ratio)
-  **Cechy tekstury** (LBP, GLCM, Haralick features)

Następnie klasyfikujemy owoce używając **SVM, Random Forest, k-NN** i **Decision Trees**.

##  Cel

**Pokazać, że można osiągnąć ~85-90% accuracy bez CNN!**

##  Dataset

**Fruits 360** (podzbiór) - Kaggle
- **48 klas** owoców i warzyw (3 kategorie: Apple, Cherry, Tomatoe)
- **34,800 obrazów treningowych**
- **12,233 obrazów testowych**
- **Łącznie: 47,033 obrazów**
- Rozmiar: 100x100px, białe tło

**Link**: https://www.kaggle.com/datasets/moltean/fruits

**Uwaga**: Projekt używa podzbioru datasetu (48 z 131 klas) dla efektywności.


##  Metodologia

### 1. Preprocessing (Przetwarzanie wstępne)
- **Wczytanie obrazu** - Ładowanie plików JPG (100x100px) używając OpenCV z konwersją BGR→RGB
- **Usunięcie białego tła** - Segmentacja obiektów metodą thresholding (cv2.inRange) oddzielająca owoc od białego tła
- **Normalizacja** - Standaryzacja wartości pikseli do zakresu [0,1] oraz resize do jednolitych wymiarów
- **Transformacje kolorów** - Konwersja do przestrzeni HSV (oddziela kolor od jasności) i LAB (percepcyjnie jednolita)
- **CLAHE** - Adaptacyjne wyrównywanie histogramu poprawiające kontrast lokalnie bez wzmacniania szumu

### 2. Ekstrakcja cech (Feature Extraction)

#### **Cechy koloru**
- **Histogramy RGB** - Rozkład intensywności w kanałach czerwonym, zielonym i niebieskim (3×256 wartości)
- **Histogramy HSV** - Hue (odcień koloru), Saturation (nasycenie), Value (jasność) - lepsze dla rozpoznawania kolorów owoców
- **Momenty kolorów** - Średnia, odchylenie standardowe, skewness i kurtosis dla każdego kanału (kompaktowa reprezentacja)
- **Dominujący kolor** - Wyznaczenie głównych kolorów metodą k-means clustering w przestrzeni RGB/HSV

#### **Cechy kształtu**
- **Area (pole)** - Liczba pikseli wewnątrz konturu owocu po segmentacji
- **Perimeter (obwód)** - Długość konturu w pikselach mierzona metodą cv2.arcLength
- **Circularity (okrągłość)** - Stosunek 4π×Area/Perimeter² równy 1.0 dla idealnego koła, mniejszy dla wydłużonych kształtów
- **Momenty Hu** - 7 niezmienników momentów odpornych na translację, rotację i skalowanie (cv2.HuMoments)
- **Aspect Ratio** - Stosunek szerokości do wysokości bounding box rozróżniający okrągłe od wydłużonych owoców
- **Eccentricity** - Miara eliptyczności od 0 (koło) do ~1 (linia), obliczona z dopasowanej elipsy

#### **Cechy tekstury**
- **LBP (Local Binary Patterns)** - Histogram lokalnych wzorów binarnych (porównanie każdego piksela z sąsiadami) opisujący teksturę skórki
- **GLCM (Gray-Level Co-occurrence Matrix)** - Macierz współwystępowania poziomów szarości analizująca przestrzenne relacje pikseli
- **Haralick features** - 13 cech z GLCM: contrast, correlation, energy, homogeneity opisujących gładkość/szorstkość tekstury
- **Edge density** - Gęstość krawędzi wykrytych algorytmem Canny wskazująca na złożoność powierzchni

### 3. Klasyfikacja

#### **SVM (Support Vector Machine)**
- Znajduje hiperpłaszczyznę maksymalnie oddzielającą klasy w przestrzeni cech, działa dobrze dla danych nieliniowych z RBF kernel

#### **Random Forest**
- Ensemble wielu drzew decyzyjnych głosujących o klasie, odporny na overfitting i szum w danych

#### **k-Nearest Neighbors (k-NN)**
- Klasyfikuje na podstawie k najbliższych sąsiadów w przestrzeni cech używając metryki euklidesowej lub manhattan

#### **Decision Tree**
- Pojedyncze drzewo decyzyjne dzielące przestrzeń cech przez threshold'y, łatwo interpretowalny ale podatny na overfitting

### 4. Ewaluacja

#### **Metryki podstawowe**
- **Accuracy** - Procent poprawnie sklasyfikowanych obrazów (TP+TN)/(TP+TN+FP+FN)
- **Precision** - Dokładność pozytywnych predykcji TP/(TP+FP), ważna gdy koszt false positive jest wysoki
- **Recall** - Czułość, procent znalezionych pozytywnych przykładów TP/(TP+FN)
- **F1-Score** - Średnia harmoniczna precision i recall (2×P×R)/(P+R), balansuje obie metryki

#### **Wizualizacje**
- **Confusion Matrix** - Macierz pomyłek pokazująca które klasy są mylone z innymi, pozwala zidentyfikować problematyczne pary
- **Cross-validation** - 5-fold lub 10-fold walidacja krzyżowa zapewniająca rzetelną ocenę na różnych podzbiorach danych
- **Feature Importance** - Ranking ważności cech (dla Random Forest) wskazujący które cechy najbardziej wpływają na klasyfikację

##  Oczekiwane wyniki

| Klasyfikator | Oczekiwana Accuracy |
|--------------|---------------------|
| SVM          | ~85%               |
| Random Forest| ~88-90%            |
| k-NN         | ~82%               |
| Decision Tree| ~75%               |

**Porównanie**: CNN na tym datasecie osiąga ~98-99%, ale wymaga dużo mocy obliczeniowej i długiego trenowania.

## Technologie

- **Python 3.8+**
- **OpenCV** - przetwarzanie obrazów
- **scikit-image** - segmentacja, ekstrakcja cech
- **scikit-learn** - klasyfikacja
- **Jupyter Notebook** - raportowanie
- **Matplotlib, Seaborn** - wizualizacje

##  Dokumentacja i źródła

### Notebooki:
- **`01_eksploracja_danych_v2.ipynb`** - Główny notebook z EDA i analizą

### Wykorzystane techniki (z dokumentacją):

**OpenCV:**
- [Dokumentacja główna](https://docs.opencv.org/4.x/)
- [Color conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)
- [Histogram calculation](https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html)

**CLAHE** (Contrast Limited Adaptive Histogram Equalization):
- [cv2.createCLAHE](https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#gad689d2607b7b3889453804f414ab1018)
- Paper: Pizer et al. (1987) - "Adaptive Histogram Equalization and Its Variations"

**LBP** (Local Binary Patterns):
- Paper: Ojala et al. (2002) - "Multiresolution Gray-Scale and Rotation Invariant Texture Classification"
- [IEEE TPAMI](https://ieeexplore.ieee.org/document/1017623)

**Shape Features:**
- [cv2.findContours](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
- Hu Moments: Hu, M. K. (1962) - "Visual Pattern Recognition by Moment Invariants"

**Data Augmentation:**
- Survey: Shorten & Khoshgoftaar (2019) - "A survey on Image Data Augmentation for Deep Learning"

## Statystyki datasetu

Z notebooka `01_eksploracja_danych_v2.ipynb`:
- **Liczba klas:** 48
- **Obrazy treningowe:** 34,800
- **Obrazy testowe:** 12,233
- **Min obrazów na klasę (train):** 304
- **Max obrazów na klasę (train):** 984
- **Średnia obrazów na klasę:** 725.0

**Kategorie:**
- Apple (22 odmiany)
- Cherry (12 odmian)
- Tomato (14 odmian)

