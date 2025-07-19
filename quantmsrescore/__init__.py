# Apply warning filter before any imports
from warnings import filterwarnings

# Suppress warnings about OPENMS_DATA_PATH
filterwarnings("ignore", message=".*OPENMS_DATA_PATH.*", category=UserWarning)

__version__ = "0.0.11"