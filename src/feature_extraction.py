"""
Moduł do ekstrakcji cech klasycznych z obrazów owoców
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from typing import Dict, List


# ============================================================================
# CECHY KOLORU (Color Features)
# ============================================================================

def extract_color_histogram(img: np.ndarray, bins: int = 32, colorspace: str = 'RGB') -> np.ndarray:
    """
    Ekstrahuje histogram kolorów dla każdego kanału.
    
    Args:
        img: Obraz wejściowy
        bins: Liczba binów histogramu
        colorspace: 'RGB', 'HSV', 'LAB'
        
    Returns:
        Spłaszczony wektor histogramów dla wszystkich kanałów
    """
    # Konwersja przestrzeni kolorów jeśli potrzeba
    if colorspace == 'RGB':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == 'HSV':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'LAB':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img_color = img
    
    # Usuń czarne piksele (tło po segmentacji)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0
    
    histograms = []
    for i in range(3):  # 3 kanały
        hist = cv2.calcHist([img_color], [i], mask.astype(np.uint8), [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalizacja
        histograms.extend(hist)
    
    return np.array(histograms)


def extract_color_moments(img: np.ndarray, colorspace: str = 'HSV') -> np.ndarray:
    """
    Ekstrahuje momenty statystyczne kolorów (średnia, std, skewness, kurtosis).
    
    Args:
        img: Obraz wejściowy BGR
        colorspace: 'RGB', 'HSV', 'LAB'
        
    Returns:
        Wektor 12 wartości (4 momenty × 3 kanały)
    """
    from scipy.stats import skew, kurtosis
    
    # Konwersja
    if colorspace == 'RGB':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == 'HSV':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'LAB':
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img_color = img
    
    # Maska non-zero pixels (bez tła)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0
    
    moments = []
    for i in range(3):
        channel = img_color[:, :, i]
        pixels = channel[mask]
        
        if len(pixels) > 0:
            moments.append(np.mean(pixels))      # Średnia
            moments.append(np.std(pixels))       # Odchylenie standardowe
            moments.append(skew(pixels))         # Skośność
            moments.append(kurtosis(pixels))     # Kurtoza
        else:
            moments.extend([0, 0, 0, 0])
    
    return np.array(moments)


def extract_dominant_colors(img: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Ekstrahuje dominujące kolory używając K-means.
    
    Args:
        img: Obraz wejściowy BGR
        k: Liczba dominujących kolorów
        
    Returns:
        Wektor RGB wartości dominujących kolorów (k × 3)
    """
    # Konwersja do RGB i reshape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0
    
    pixels = img_rgb[mask].reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < k:
        return np.zeros(k * 3)
    
    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Sortuj według częstości
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    centers = centers[sorted_idx]
    
    return centers.flatten()


# ============================================================================
# CECHY KSZTAŁTU (Shape Features)
# ============================================================================

def extract_shape_features(img: np.ndarray) -> Dict[str, float]:
    """
    Ekstrahuje cechy kształtu z obrazu.
    
    Args:
        img: Obraz wejściowy BGR
        
    Returns:
        Słownik z cechami kształtu
    """
    # Konwersja do grayscale i binaryzacja
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'area': 0, 'perimeter': 0, 'circularity': 0,
            'aspect_ratio': 0, 'extent': 0, 'solidity': 0,
            'equivalent_diameter': 0
        }
    
    # Największy kontur (obiekt)
    contour = max(contours, key=cv2.contourArea)
    
    # Oblicz cechy
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Circularity (okrągłość): 1.0 dla idealnego koła
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Extent: stosunek obszaru obiektu do bounding box
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    # Solidity: stosunek obszaru do convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'solidity': solidity,
        'equivalent_diameter': equivalent_diameter
    }


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """
    Ekstrahuje 7 momentów Hu (niezmienników).
    
    Args:
        img: Obraz wejściowy BGR
        
    Returns:
        Wektor 7 momentów Hu
    """
    # Konwersja do grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Oblicz momenty obrazu
    moments = cv2.moments(gray)
    
    # Momenty Hu
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform (Hu moments mają bardzo duży zakres)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments


# ============================================================================
# CECHY TEKSTURY (Texture Features)
# ============================================================================

def extract_lbp_features(img: np.ndarray, P: int = 8, R: int = 1, bins: int = 32) -> np.ndarray:
    """
    Ekstrahuje cechy Local Binary Pattern (LBP).
    
    Args:
        img: Obraz wejściowy BGR
        P: Liczba punktów sąsiednich
        R: Promień
        bins: Liczba binów histogramu
        
    Returns:
        Histogram LBP
    """
    # Konwersja do grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Oblicz LBP
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    
    # Maska non-zero pixels
    mask = gray > 0
    
    # Histogram
    hist, _ = np.histogram(lbp[mask], bins=bins, range=(0, P + 2))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-7)  # Normalizacja
    
    return hist


def extract_glcm_features(img: np.ndarray, distances: List[int] = [1], 
                          angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
    """
    Ekstrahuje cechy z Gray-Level Co-occurrence Matrix (GLCM).
    
    Args:
        img: Obraz wejściowy BGR
        distances: Lista dystansów do rozważenia
        angles: Lista kątów (w radianach)
        
    Returns:
        Wektor cech GLCM (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    """
    # Konwersja do grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Zmniejsz liczbę poziomów szarości (256 → 32) dla wydajności
    gray_reduced = (gray // 8).astype(np.uint8)
    
    # Oblicz GLCM
    glcm = graycomatrix(
        gray_reduced, 
        distances=distances, 
        angles=angles, 
        levels=32,
        symmetric=True, 
        normed=True
    )
    
    # Ekstrahuj właściwości
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        features.extend([
            np.mean(values),
            np.std(values)
        ])
    
    return np.array(features)


def extract_haralick_features(img: np.ndarray) -> np.ndarray:
    """
    Ekstrahuje cechy Haralick z GLCM.
    Uproszczona wersja - używa graycoprops z kilkoma właściwościami.
    
    Args:
        img: Obraz wejściowy BGR
        
    Returns:
        Wektor cech Haralick
    """
    return extract_glcm_features(img, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])


# ============================================================================
# GŁÓWNA FUNKCJA EKSTRAKCJI
# ============================================================================

def extract_all_features(img: np.ndarray, feature_config: Dict = None) -> np.ndarray:
    """
    Ekstrahuje wszystkie cechy z obrazu.
    
    Args:
        img: Obraz wejściowy BGR (już przetworzony)
        feature_config: Konfiguracja cech do ekstrakcji (opcjonalne)
        
    Returns:
        Wektor wszystkich cech
    """
    if feature_config is None:
        feature_config = {
            'color_histogram_rgb': True,
            'color_histogram_hsv': True,
            'color_moments_hsv': True,
            'dominant_colors': True,
            'shape_features': True,
            'hu_moments': True,
            'lbp': True,
            'glcm': True
        }
    
    features = []
    
    # Cechy koloru
    if feature_config.get('color_histogram_rgb', True):
        hist_rgb = extract_color_histogram(img, bins=16, colorspace='RGB')
        features.extend(hist_rgb)
    
    if feature_config.get('color_histogram_hsv', True):
        hist_hsv = extract_color_histogram(img, bins=16, colorspace='HSV')
        features.extend(hist_hsv)
    
    if feature_config.get('color_moments_hsv', True):
        moments = extract_color_moments(img, colorspace='HSV')
        features.extend(moments)
    
    if feature_config.get('dominant_colors', True):
        dom_colors = extract_dominant_colors(img, k=3)
        features.extend(dom_colors)
    
    # Cechy kształtu
    if feature_config.get('shape_features', True):
        shape_dict = extract_shape_features(img)
        features.extend(shape_dict.values())
    
    if feature_config.get('hu_moments', True):
        hu = extract_hu_moments(img)
        features.extend(hu)
    
    # Cechy tekstury
    if feature_config.get('lbp', True):
        lbp = extract_lbp_features(img, P=8, R=1, bins=16)
        features.extend(lbp)
    
    if feature_config.get('glcm', True):
        glcm = extract_glcm_features(img)
        features.extend(glcm)
    
    return np.array(features, dtype=np.float32)


def get_feature_names(feature_config: Dict = None) -> List[str]:
    """
    Zwraca nazwy wszystkich ekstrahowanych cech.
    
    Args:
        feature_config: Konfiguracja cech
        
    Returns:
        Lista nazw cech
    """
    if feature_config is None:
        feature_config = {
            'color_histogram_rgb': True,
            'color_histogram_hsv': True,
            'color_moments_hsv': True,
            'dominant_colors': True,
            'shape_features': True,
            'hu_moments': True,
            'lbp': True,
            'glcm': True
        }
    
    names = []
    
    # Histogramy kolorów
    if feature_config.get('color_histogram_rgb', True):
        for channel in ['R', 'G', 'B']:
            names.extend([f'hist_rgb_{channel}_{i}' for i in range(16)])
    
    if feature_config.get('color_histogram_hsv', True):
        for channel in ['H', 'S', 'V']:
            names.extend([f'hist_hsv_{channel}_{i}' for i in range(16)])
    
    # Momenty kolorów
    if feature_config.get('color_moments_hsv', True):
        for channel in ['H', 'S', 'V']:
            for moment in ['mean', 'std', 'skew', 'kurt']:
                names.append(f'moment_hsv_{channel}_{moment}')
    
    # Dominujące kolory
    if feature_config.get('dominant_colors', True):
        for i in range(3):
            for channel in ['R', 'G', 'B']:
                names.append(f'dominant_color_{i}_{channel}')
    
    # Cechy kształtu
    if feature_config.get('shape_features', True):
        names.extend([
            'area', 'perimeter', 'circularity', 'aspect_ratio',
            'extent', 'solidity', 'equivalent_diameter'
        ])
    
    # Momenty Hu
    if feature_config.get('hu_moments', True):
        names.extend([f'hu_moment_{i}' for i in range(7)])
    
    # LBP
    if feature_config.get('lbp', True):
        names.extend([f'lbp_{i}' for i in range(16)])
    
    # GLCM
    if feature_config.get('glcm', True):
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            names.extend([f'glcm_{prop}_mean', f'glcm_{prop}_std'])
    
    return names


# Przykład użycia
if __name__ == "__main__":
    print("Test ekstrakcji cech...")
    
    # Utworzenie przykładowego obrazu
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (0, 255, 0), -1)  # Zielone koło
    
    # Ekstrakcja cech
    features = extract_all_features(img)
    feature_names = get_feature_names()
    
    print(f"Liczba cech: {len(features)}")
    print(f"Liczba nazw cech: {len(feature_names)}")
    print(f"\nPrzykładowe cechy:")
    for i, (name, value) in enumerate(zip(feature_names[:10], features[:10])):
        print(f"  {name}: {value:.4f}")

