"""
Moduł do klasyfikacji i ewaluacji modeli
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# PREPROCESSING DANYCH
# ============================================================================

def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple:
    """
    Przygotowuje dane do trenowania: split + skalowanie.
    
    Args:
        X: Macierz cech
        y: Etykiety
        test_size: Rozmiar zbioru testowego
        random_state: Seed dla reprodukowalności
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    # Kodowanie etykiet
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Skalowanie
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


# ============================================================================
# KLASYFIKATORY
# ============================================================================

def get_svm_classifier(kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale') -> SVC:
    """
    Tworzy klasyfikator SVM.
    
    Args:
        kernel: Typ kernela ('linear', 'rbf', 'poly')
        C: Parametr regularyzacji
        gamma: Parametr kernela
        
    Returns:
        Model SVM
    """
    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)


def get_random_forest_classifier(n_estimators: int = 100, max_depth: int = None,
                                  min_samples_split: int = 2) -> RandomForestClassifier:
    """
    Tworzy klasyfikator Random Forest.
    
    Args:
        n_estimators: Liczba drzew
        max_depth: Maksymalna głębokość drzewa
        min_samples_split: Minimalna liczba sampli do splitu
        
    Returns:
        Model Random Forest
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )


def get_knn_classifier(n_neighbors: int = 5, metric: str = 'euclidean') -> KNeighborsClassifier:
    """
    Tworzy klasyfikator k-NN.
    
    Args:
        n_neighbors: Liczba sąsiadów
        metric: Metryka dystansu ('euclidean', 'manhattan', 'minkowski')
        
    Returns:
        Model k-NN
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)


def get_decision_tree_classifier(max_depth: int = None, 
                                  min_samples_leaf: int = 1) -> DecisionTreeClassifier:
    """
    Tworzy klasyfikator Decision Tree.
    
    Args:
        max_depth: Maksymalna głębokość drzewa
        min_samples_leaf: Minimalna liczba sampli w liściu
        
    Returns:
        Model Decision Tree
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )


# ============================================================================
# TRENOWANIE I EWALUACJA
# ============================================================================

def train_and_evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = 'Model') -> Dict:
    """
    Trenuje model i oblicza metryki ewaluacji.
    
    Args:
        model: Klasyfikator scikit-learn
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        model_name: Nazwa modelu (do logowania)
        
    Returns:
        Słownik z metrykami
    """
    print(f"\n{'='*60}")
    print(f"Trenowanie: {model_name}")
    print(f"{'='*60}")
    
    # Trenowanie
    model.fit(X_train, y_train)
    
    # Predykcje
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metryki
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # Dla testowego
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    
    results = {
        'model': model,
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    return results


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, 
                         cv: int = 5) -> Dict:
    """
    Przeprowadza cross-validation.
    
    Args:
        model: Klasyfikator
        X: Cechy
        y: Etykiety
        cv: Liczba foldów
        
    Returns:
        Słownik z wynikami CV
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    return {
        'cv_scores': scores,
        'cv_mean': scores.mean(),
        'cv_std': scores.std()
    }


def hyperparameter_tuning(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                          cv: int = 3) -> Tuple[Any, Dict]:
    """
    Przeprowadza Grid Search dla hyperparametrów.
    
    Args:
        model_type: 'svm', 'rf', 'knn', 'dt'
        X_train, y_train: Dane treningowe
        cv: Liczba foldów
        
    Returns:
        Tuple (najlepszy model, najlepsze parametry)
    """
    print(f"\nHyperparameter tuning dla: {model_type}")
    
    if model_type == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_jobs=-1)
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'metric': ['euclidean', 'manhattan']
        }
    elif model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', 
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepszy score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


# ============================================================================
# WIZUALIZACJE
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          labels: list = None, normalize: bool = False,
                          title: str = 'Confusion Matrix', figsize: Tuple = (12, 10)):
    """
    Rysuje confusion matrix.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety
        labels: Nazwy klas
        normalize: Czy normalizować
        title: Tytuł wykresu
        figsize: Rozmiar figury
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    
    # Dla dużej liczby klas (>20), użyj mniejszej czcionki
    if len(np.unique(y_true)) > 20:
        sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.tight_layout()
    
    return plt.gcf()


def plot_feature_importance(model: RandomForestClassifier, feature_names: list,
                            top_n: int = 20, figsize: Tuple = (10, 8)):
    """
    Rysuje ważność cech dla Random Forest.
    
    Args:
        model: Wytrenowany model Random Forest
        feature_names: Nazwy cech
        top_n: Liczba najważniejszych cech do pokazania
        figsize: Rozmiar figury
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=figsize)
    plt.title(f'Top {top_n} najważniejszych cech')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel('Ważność cechy')
    plt.tight_layout()
    
    return plt.gcf()


def plot_model_comparison(results_dict: Dict, metric: str = 'test_accuracy',
                         figsize: Tuple = (10, 6)):
    """
    Porównuje różne modele.
    
    Args:
        results_dict: Słownik {nazwa_modelu: results}
        metric: Metryka do porównania
        figsize: Rozmiar figury
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Porównanie modeli - {metric.replace("_", " ").title()}')
    plt.ylim([0, 1])
    
    # Dodaj wartości na słupkach
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()


# ============================================================================
# ZAPISYWANIE/WCZYTYWANIE MODELI
# ============================================================================

def save_model(model: Any, filepath: str, scaler: StandardScaler = None, 
               label_encoder: LabelEncoder = None):
    """
    Zapisuje model, scaler i label encoder.
    
    Args:
        model: Wytrenowany model
        filepath: Ścieżka do zapisu
        scaler: Scaler (opcjonalnie)
        label_encoder: Label encoder (opcjonalnie)
    """
    data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    joblib.dump(data, filepath)
    print(f"Model zapisany: {filepath}")


def load_model(filepath: str) -> Dict:
    """
    Wczytuje zapisany model.
    
    Args:
        filepath: Ścieżka do modelu
        
    Returns:
        Słownik z modelem, scalerem, label encoderem
    """
    data = joblib.load(filepath)
    print(f"Model wczytany: {filepath}")
    return data


def predict_single_image(img_features: np.ndarray, model_data: Dict) -> str:
    """
    Przewiduje klasę dla pojedynczego obrazu.
    
    Args:
        img_features: Wektor cech obrazu
        model_data: Słownik z modelem, scalerem, label encoderem
        
    Returns:
        Przewidziana klasa (nazwa)
    """
    model = model_data['model']
    scaler = model_data['scaler']
    le = model_data['label_encoder']
    
    # Skalowanie
    features_scaled = scaler.transform(img_features.reshape(1, -1))
    
    # Predykcja
    pred_encoded = model.predict(features_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    # Prawdopodobieństwa (jeśli dostępne)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        return pred_label, confidence
    
    return pred_label, None


# ============================================================================
# RAPORTOWANIE
# ============================================================================

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   label_encoder: LabelEncoder = None) -> str:
    """
    Generuje szczegółowy raport klasyfikacji.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Przewidziane etykiety
        label_encoder: Label encoder do dekodowania nazw klas
        
    Returns:
        String z raportem
    """
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = None
    
    report = classification_report(y_true, y_pred, target_names=target_names, 
                                   zero_division=0)
    return report


# Przykład użycia
if __name__ == "__main__":
    print("Test klasyfikacji...")
    
    # Wygeneruj przykładowe dane
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=30,
                               n_classes=10, random_state=42)
    
    # Przygotuj dane
    X_train, X_test, y_train, y_test, scaler, le = prepare_data(X, y)
    
    # Trenuj model
    model = get_random_forest_classifier(n_estimators=50)
    results = train_and_evaluate(model, X_train, y_train, X_test, y_test, 'Random Forest')
    
    print(f"\nTest accuracy: {results['test_accuracy']:.4f}")

