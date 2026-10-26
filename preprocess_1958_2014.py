"""Thin wrapper. Same as `python preprocess.py 1958_2014`."""
from preprocess import main

if __name__ == "__main__":
    main('1958_2014', save_flag=False)
