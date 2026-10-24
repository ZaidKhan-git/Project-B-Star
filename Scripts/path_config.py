"""
path_config.py - Centralized Path & Dependency Management for Project B-Star
=============================================================================
Provides deterministic, environment-agnostic path resolution for all scripts,
data files, and models, eliminating path conflicts whether executed from:
- Project root (e.g. `python scripts/train_clean_b_star_model.py` or `streamlit run app.py`)
- Inside scripts/ directory (e.g. `cd scripts && python app.py`)
- External tooling or subshells
"""

import os
import sys

# Locate root directory relative to this config file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATAFILES_DIR = os.path.join(PROJECT_ROOT, "datafiles")
MODELS_DIR = os.path.join(PROJECT_ROOT, "3 Models")

# Guarantee importability of all sibling scripts and root packages
for path in [SCRIPT_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


def get_data_file(filename: str = "2014-2025.csv") -> str:
    """Resolve the absolute path to a dataset file with fallback searching."""
    candidates = [
        os.path.join(DATAFILES_DIR, filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(SCRIPT_DIR, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Default to datafiles directory path even if file not yet created
    return os.path.abspath(os.path.join(DATAFILES_DIR, filename))


def get_db_path(dbname: str = "satellite_data.db") -> str:
    """Resolve the absolute path to SQLite database."""
    return get_data_file(dbname)


def get_model_path(*subpaths: str) -> str:
    """Resolve absolute path to a model file, ensemble, or metadata."""
    candidates = [
        os.path.join(MODELS_DIR, *subpaths),
        os.path.join(PROJECT_ROOT, *subpaths),
        os.path.join("3 Models", *subpaths)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(MODELS_DIR, *subpaths))
