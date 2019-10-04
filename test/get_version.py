import sys
sys.path.append('../tfdeterminism')
from version import __version__

def get_version():
    return __version__

if __name__ == "__main__":
    print(__version__)
