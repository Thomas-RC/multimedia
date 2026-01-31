"""
Moduł do preprocessingu obrazów owoców
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Wczytuje obraz z pliku.
    
    Args:
        image_path: Ścieżka do pliku obrazu
        
    Returns:
        Obraz w formacie BGR (OpenCV)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    return img


def remove_white_background(img: np.ndarray, threshold: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    """
    Usuwa białe tło z obrazu owocu (segmentacja).
    
    Args:
        img: Obraz wejściowy BGR
        threshold: Próg bieli (domyślnie 240)
        
    Returns:
        Tuple (obraz z usuniętym tłem, maska binarna)
    """
    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Utworzenie maski: wszystko poniżej threshold to obiekt (czarne), powyżej to tło (białe)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Operacje morfologiczne do wygładzenia maski
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Aplikacja maski
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask


def crop_to_object(img: np.ndarray, mask: np.ndarray, padding: int = 5) -> np.ndarray:
    """
    Przycina obraz do bounding box obiektu z opcjonalnym paddingiem.
    
    Args:
        img: Obraz wejściowy
        mask: Maska binarna obiektu
        padding: Padding wokół obiektu (piksele)
        
    Returns:
        Przycięty obraz
    """
    # Znajdź kontury
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Znajdź największy kontur (obiekt owocu)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Dodaj padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Przetnij
    cropped = img[y:y+h, x:x+w]
    
    return cropped


def resize_image(img: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Zmienia rozmiar obrazu do określonego rozmiaru.
    
    Args:
        img: Obraz wejściowy
        size: Docelowy rozmiar (width, height)
        
    Returns:
        Przeskalowany obraz
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def convert_colorspace(img: np.ndarray, colorspace: str = 'HSV') -> np.ndarray:
    """
    Konwertuje obraz do innej przestrzeni kolorów.
    
    Args:
        img: Obraz wejściowy BGR
        colorspace: Docelowa przestrzeń ('HSV', 'LAB', 'RGB', 'GRAY')
        
    Returns:
        Obraz w nowej przestrzeni kolorów
    """
    colorspace = colorspace.upper()
    
    if colorspace == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif colorspace == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Nieznana przestrzeń kolorów: {colorspace}")


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizuje wartości pikseli do zakresu [0, 1].
    
    Args:
        img: Obraz wejściowy
        
    Returns:
        Znormalizowany obraz
    """
    return img.astype(np.float32) / 255.0


def load_and_preprocess(
    image_path: str,
    remove_bg: bool = True,
    crop: bool = True,
    resize: Optional[Tuple[int, int]] = (100, 100)
) -> np.ndarray:
    """
    Pipeline preprocessingu: wczytanie → usunięcie tła → przycięcie → resize.
    
    Args:
        image_path: Ścieżka do obrazu
        remove_bg: Czy usunąć białe tło
        crop: Czy przyciąć do obiektu
        resize: Docelowy rozmiar lub None
        
    Returns:
        Przetworzony obraz BGR
    """
    # Wczytanie
    img = load_image(image_path)
    
    # Usunięcie tła
    if remove_bg:
        img, mask = remove_white_background(img)
        
        # Przycięcie
        if crop:
            img = crop_to_object(img, mask)
    
    # Resize
    if resize is not None:
        img = resize_image(img, resize)
    
    return img


def get_all_image_paths(data_dir: str, split: str = 'Training') -> list:
    """
    Pobiera wszystkie ścieżki do obrazów w datasecie.
    
    Args:
        data_dir: Ścieżka do katalogu data/raw/
        split: 'Training' lub 'Test'
        
    Returns:
        Lista ścieżek do obrazów
    """
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        raise ValueError(f"Katalog nie istnieje: {data_path}")
    
    # Pobierz wszystkie pliki jpg/png
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(data_path.rglob(ext))
    
    return sorted([str(p) for p in image_paths])


def extract_label_from_path(image_path: str) -> str:
    """
    Wyciąga nazwę klasy (etykietę) ze ścieżki obrazu.
    
    Args:
        image_path: Ścieżka do obrazu np. 'data/raw/Training/Apple Red 1/0_100.jpg'
        
    Returns:
        Nazwa klasy np. 'Apple Red 1'
    """
    return Path(image_path).parent.name


# Przykład użycia
if __name__ == "__main__":
    # Test preprocessingu
    test_path = "data/raw/Training/Apple Braeburn/0_100.jpg"
    
    if Path(test_path).exists():
        print("Test preprocessingu...")
        img = load_and_preprocess(test_path)
        print(f"Rozmiar przetworzonego obrazu: {img.shape}")
        print(f"Etykieta: {extract_label_from_path(test_path)}")
    else:
        print(f"Przykładowy obraz nie istnieje: {test_path}")
        print("Pobierz dataset Fruits 360 do data/raw/")

