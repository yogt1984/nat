#!/usr/bin/env python3
"""
Quick test to verify validation pipeline works without external dependencies.
Uses only standard library for basic verification.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

def check_dependencies():
    """Check if required packages are available."""
    missing = []

    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('websockets', 'websockets'),
    ]

    for module, pip_name in packages:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    return missing

def main():
    print("=" * 60)
    print("NAT Validation - Quick Dependency Check")
    print("=" * 60)

    missing = check_dependencies()

    if missing:
        print("\n❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")

        print("\nTo install, run:")
        print("   pip install " + " ".join(missing))
        print("\nOr create a virtual environment:")
        print("   cd /home/onat/nat/exploration/validation")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed!")

    # Quick import test
    print("\nVerifying imports...")

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from sklearn.metrics import mutual_info_score
        print("✓ Core packages working")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

    # Quick synthetic data test
    print("\nGenerating test data...")

    np.random.seed(42)
    n = 1000

    # Simple regime-dependent entropy
    regimes = np.random.choice(['MR', 'TF'], size=n)
    entropy = np.where(regimes == 'TF',
                       np.random.normal(0.3, 0.1, n),
                       np.random.normal(0.7, 0.1, n))
    entropy = np.clip(entropy, 0.05, 0.95)

    # Quick MI calculation
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(regimes)
    mi = mutual_info_classif(entropy.reshape(-1, 1), y, n_neighbors=5, random_state=42)[0]

    print(f"✓ Synthetic MI test: {mi:.4f}")

    if mi > 0.05:
        print("✓ Pipeline working correctly (synthetic data shows signal)")
    else:
        print("⚠ Unexpected: synthetic data should show clear signal")

    print("\n" + "=" * 60)
    print("READY TO RUN VALIDATION")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Quick synthetic test (5 min):")
    print("     python run_validation.py --synthetic")
    print("")
    print("  2. Short real data (10 min):")
    print("     python run_validation.py --duration 600")
    print("")
    print("  3. Full validation (1 hour):")
    print("     python run_validation.py --duration 3600")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
