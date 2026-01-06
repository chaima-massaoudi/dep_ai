import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE pour Docker / Azure

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def detect_drift(reference_file, production_file, threshold=0.05, output_dir="drift_reports"):
    """
    Détecte le data drift entre les données de référence et de production
    en utilisant le test de Kolmogorov-Smirnov.
    
    Args:
        reference_file: Chemin vers le fichier CSV de référence
        production_file: Chemin vers le fichier CSV de production
        threshold: Seuil de p-value pour détecter le drift (défaut: 0.05)
        output_dir: Répertoire pour sauvegarder les rapports
    
    Returns:
        dict: Résultats du drift pour chaque feature
    """
    os.makedirs(output_dir, exist_ok=True)

    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)

    results = {}

    for col in ref.columns:
        if col != "Exited" and col in prod.columns:
            stat, p = ks_2samp(ref[col].dropna(), prod[col].dropna())
            results[col] = {
                "p_value": float(p),
                "statistic": float(stat),
                "drift_detected": bool(p < threshold),
                "type": "numerical"
            }

    report_path = f"{output_dir}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def generate_drift_report(results, output_dir="drift_reports"):
    """
    Génère un rapport visuel du drift détecté.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    features = list(results.keys())
    p_values = [results[f]["p_value"] for f in features]
    drifted = [results[f]["drift_detected"] for f in features]
    
    colors = ['red' if d else 'green' for d in drifted]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(features, p_values, color=colors)
    plt.axvline(x=0.05, color='black', linestyle='--', label='Seuil (0.05)')
    plt.xlabel('P-Value')
    plt.title('Drift Detection - P-Values par Feature')
    plt.legend()
    plt.tight_layout()
    
    report_path = f"{output_dir}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(report_path)
    plt.close()
    
    return report_path
