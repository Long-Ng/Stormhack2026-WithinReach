import sys
from pathlib import Path

# Tests import the app modules (config, features, ...) the same way main.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
