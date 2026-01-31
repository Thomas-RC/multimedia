#!/usr/bin/env python3
"""
Skrypt do ekstrakcji cech z datasetu Fruits 360

Użycie:
    python extract_features.py --subset --n-classes 20
    python extract_features.py --full  # Cały dataset (długo!)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Dodaj src do path
sys.path.append('src')

from preprocessing import load_and_preprocess
from feature_extraction import extract_all_features, get_feature_names


def extract_features_from_dataset(data_dir, split='Training', use_subset=True, 
                                  n_classes=20, max_images_per_class=None):
    """
    Ekstrahuje cechy z datasetu.
    
    Args:
        data_dir: Katalog z danymi (np. 'data/raw')
        split: 'Training' lub 'Test'
        use_subset: Czy użyć tylko podzbioru klas
        n_classes: Ile klas użyć (jeśli use_subset=True)
        max_images_per_class: Max obrazów na klasę (None = wszystkie)
        
    Returns:
        DataFrame z cechami
    """
    split_dir = Path(data_dir) / split
    
    if not split_dir.exists():
        print(f"❌ Katalog nie istnieje: {split_dir}")
        print(f"   Pobierz dataset Fruits 360 i rozpakuj do {data_dir}")
        return None
    
    # Pobierz klasy
    all_classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    
    if use_subset:
        # Użyj pierwszych n klas (alfabetycznie)
        selected_classes = all_classes[:min(n_classes, len(all_classes))]
        print(f"\n📋 Używam {len(selected_classes)} klas (subset)")
    else:
        selected_classes = all_classes
        print(f"\n📋 Używam wszystkich {len(selected_classes)} klas")
    
    print(f"Klasy: {', '.join(selected_classes[:5])}{'...' if len(selected_classes) > 5 else ''}")
    
    # Zbierz ścieżki do obrazów
    image_paths = []
    labels = []
    
    for class_name in selected_classes:
        class_dir = split_dir / class_name
        class_images = list(class_dir.glob('*.jpg'))
        
        if max_images_per_class:
            class_images = class_images[:max_images_per_class]
        
        image_paths.extend(class_images)
        labels.extend([class_name] * len(class_images))
    
    print(f"\n📸 Łącznie obrazów do przetworzenia: {len(image_paths)}")
    print(f"⏰ Szacowany czas: ~{len(image_paths) * 0.5 / 60:.1f} minut")
    
    # Ekstrahuj cechy
    features_list = []
    valid_labels = []
    valid_paths = []
    
    print("\n🔄 Ekstrahowanie cech...")
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        try:
            # Przetwórz obraz
            img = load_and_preprocess(str(img_path))
            
            # Ekstrahuj cechy
            features = extract_all_features(img)
            
            features_list.append(features)
            valid_labels.append(label)
            valid_paths.append(str(img_path))
            
        except Exception as e:
            # Pomiń obrazy z błędami
            continue
    
    # Utwórz DataFrame
    feature_names_list = get_feature_names()
    
    df = pd.DataFrame(features_list, columns=feature_names_list)
    df['label'] = valid_labels
    df['image_path'] = valid_paths
    
    print(f"\n✅ Ukończono!")
    print(f"  Przetworzono: {len(df)} obrazów")
    print(f"  Liczba cech: {len(feature_names_list)}")
    print(f"  Liczba klas: {df['label'].nunique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Ekstrakcja cech z datasetu Fruits 360')
    
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Katalog z danymi (default: data/raw)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Katalog wyjściowy (default: data/processed)')
    
    # Subset vs Full
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--subset', action='store_true', default=True,
                      help='Użyj podzbioru klas (default)')
    group.add_argument('--full', action='store_true',
                      help='Użyj wszystkich klas (długo!)')
    
    parser.add_argument('--n-classes', type=int, default=20,
                       help='Liczba klas dla subsetu (default: 20)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Max obrazów na klasę (default: wszystkie)')
    
    parser.add_argument('--train-only', action='store_true',
                       help='Tylko zbiór treningowy')
    parser.add_argument('--test-only', action='store_true',
                       help='Tylko zbiór testowy')
    
    args = parser.parse_args()
    
    # Określ czy subset czy full
    use_subset = not args.full
    
    # Utwórz katalog wyjściowy
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🍎 Ekstrakcja cech - Fruits 360")
    print("="*60)
    print(f"Tryb: {'SUBSET' if use_subset else 'FULL DATASET'}")
    if use_subset:
        print(f"Liczba klas: {args.n_classes}")
    if args.max_images:
        print(f"Max obrazów/klasę: {args.max_images}")
    print("="*60)
    
    # Ekstrahuj Training
    if not args.test_only:
        print("\n📦 ZBIÓR TRENINGOWY")
        df_train = extract_features_from_dataset(
            data_dir=args.data_dir,
            split='Training',
            use_subset=use_subset,
            n_classes=args.n_classes,
            max_images_per_class=args.max_images
        )
        
        if df_train is not None:
            output_path = Path(args.output_dir) / 'features_train.csv'
            df_train.to_csv(output_path, index=False)
            print(f"\n💾 Zapisano do: {output_path}")
            print(f"   Rozmiar: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Ekstrahuj Test
    if not args.train_only:
        print("\n\n📦 ZBIÓR TESTOWY")
        df_test = extract_features_from_dataset(
            data_dir=args.data_dir,
            split='Test',
            use_subset=use_subset,
            n_classes=args.n_classes,
            max_images_per_class=args.max_images
        )
        
        if df_test is not None:
            output_path = Path(args.output_dir) / 'features_test.csv'
            df_test.to_csv(output_path, index=False)
            print(f"\n💾 Zapisano do: {output_path}")
            print(f"   Rozmiar: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ GOTOWE!")
    print("="*60)
    print("\nNastępny krok:")
    print("  1. Uruchom Jupyter: jupyter notebook")
    print("  2. Otwórz: notebooks/03_klasyfikacja.ipynb")
    print("="*60)


if __name__ == "__main__":
    main()

