"""Regrid ORAS5 monthly SST (sosstsst) to a 1 degree grid (r360x180) with CDO.

Usage
    python cdo_regrid.py 1958_2014
    python cdo_regrid.py 2015_2023

The thin wrappers cdo_regrid_1958_2014.py and cdo_regrid_2015_2023.py call
main() with the matching period.
"""
import argparse

from cdo import Cdo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

DATADIR = '/work/kajiyama/data/ORAS5'
SAVEDIR = '/work/kajiyama/cdo/ORAS5'
TIME_LENGTH = 1

# Each period lists its year range, the months as (number, name) pairs and
# the ORAS5 stream tag in the input filename (CONS is consolidated, OPER is operational).
PERIODS = {
    '1958_2014': dict(begin_year=1958, end_year=2014,
                      months=[('04', 'Apr'), ('07', 'Jul')], stream='CONS'),
    '2015_2023': dict(begin_year=2015, end_year=2023,
                      months=[('04', 'Apr')], stream='OPER'),
}


def main(period, save_flag=False):
    cfg = PERIODS[period]
    RE = Remap(save_flag=save_flag)
    for year in range(cfg['begin_year'], cfg['end_year']+1):
        for m_num, m_str in cfg['months']:
            infile = DATADIR + f"/sosstsst_control_monthly_highres_2D_{year}{m_num}_{cfg['stream']}_v0.1.nc"
            outfile = SAVEDIR + f"/sst_{year}_{m_str}.nc"
            RE.regrid(infile, outfile, TIME_LENGTH)
    RE._plot(infile)


class Remap():
    def __init__(self, save_flag=False):
        self.save_flag = save_flag
        self.variable = 'sosstsst'

    def regrid(self, infile, outfile, time_length):
        if self.save_flag is True:
            cdo = Cdo()
            cdo.remapbil('r360x180', input=f"-seltimestep,1/{time_length} -selvar,{self.variable} "+infile, output=outfile)
            print(f'{outfile}: saved')
        else:
            print("outfile is NOT saved yet")

    def _plot(self, infile, origin='lower'):
        cdo = Cdo()
        projection = ccrs.PlateCarree(central_longitude=0)
        val = cdo.remapbil('r360x180', input='-seltimestep,1 '+infile, returnXArray=f"{self.variable}")
        data = val.plot(subplot_kws=dict(projection=projection, facecolor='gray'),
                        transform=projection)
        data.axes.coastlines()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('period', choices=sorted(PERIODS))
    parser.add_argument('--save', action='store_true', help='actually write the regridded files')
    args = parser.parse_args()
    main(args.period, save_flag=args.save)
