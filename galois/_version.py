import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")

# Default version
__version__ = None

if os.path.exists(FILE):
    with open(FILE, "r", encoding="utf-8") as f:
        __version__ = f.read().split("\n")[0]
