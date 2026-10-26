"""Thin wrapper. Same as `python cdo_regrid.py 2015_2023`."""
from cdo_regrid import main

if __name__ == "__main__":
    main('2015_2023', save_flag=False)
