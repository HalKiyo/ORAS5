import numpy as np
import numpy.ma as ma
import netCDF4
from cdo import Cdo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def main():
    save_flag = False
    datadir = '/work/kajiyama/data/ORAS5'
    savedir  = '/work/kajiyama/cdo/ORAS5'
    begin_year = 2015
    end_year = 2023
    months_num = ['04']
    monnths_str = ['Apr']
    time_length = 1

    RE = Remap(save_flag=save_flag)
    for year in range(begin_year, end_year+1):
        for m_num, m_str in zip(months_num, monnths_str):
            infile = datadir + f"/sosstsst_control_monthly_highres_2D_{year}{m_num}_OPER_v0.1.nc"
            outfile = savedir + f"/sst_{year}_{m_str}.nc"
            RE.regrid(infile, outfile, time_length)
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
        img_extent = (-180, 180, -90, 90)
        val = cdo.remapbil('r360x180', input=f'-seltimestep,1 '+infile, returnXArray=f"{self.variable}")
        data = val.plot(subplot_kws=dict(projection=projection, facecolor='gray'),
                        transform=projection)
        data.axes.coastlines()
        plt.show()

if __name__ == "__main__":
    main()
