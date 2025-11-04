"""Version information for DHC-SSM Architecture."""

__version__ = "3.0.0"
__author__ = "DHC-SSM Development Team"
__license__ = "MIT"
__description__ = "Deterministic Hierarchical Causal State Space Model - Enhanced v3.0"

VERSION_INFO = {
    "major": 3,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "build": "2025.11.02"
}

def get_version() -> str:
    """Get the version string."""
    return __version__

def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO
