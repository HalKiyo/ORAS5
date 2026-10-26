"""Thin wrapper. Same as `python preprocess.py 2015_2023`."""
from preprocess import main

if __name__ == "__main__":
    main('2015_2023', save_flag=False)
