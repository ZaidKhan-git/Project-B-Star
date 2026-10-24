"""
app.py - Project B-Star Streamlit Root Entrypoint
==================================================
Root launcher that executes the Mission Control dashboard located at scripts/app.py.
Allows launching from either the repository root:
    streamlit run app.py
or directly targeting the script:
    streamlit run scripts/app.py
"""

import os
import sys
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")

# Configure search paths so all modules are discoverable
for p in [SCRIPTS_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

target = os.path.join(SCRIPTS_DIR, "app.py")

if __name__ == "__main__":
    runpy.run_path(target, run_name="__main__")
