"""Thin wrapper. Same as `python cdo_regrid.py 1958_2014`."""
from cdo_regrid import main

if __name__ == "__main__":
    main('1958_2014', save_flag=False)
