import os

# Default version
__version__ = None

# Read version from version.txt file
FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
if os.path.exists(FILE):
    with open(FILE) as f:
        __version__ = f.read().split("\n")[0]
