import pickle
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from mpl_toolkits import basemap
import colorcet as cc
import cmaps

def main():
    save_flag = False
    PRE = Preprocess()
    DRA = Draw()
    months_str = ['Apr', 'Jul']

    # sst: ndarray filled with -99999
    for m_str in months_str:
        sst = PRE.make_var(m_str=m_str)
        sst_anom = PRE.anomaly(sst)
        sst_clim, sst_variance, sst_std = PRE.standardize(sst)
        sst_coarse = PRE.interpolation(sst)
        sst_coarse_anom = PRE.anomaly(sst_coarse)
        sst_coarse_clim, sst_coarse_variance, sst_coarse_std = PRE.standardize(sst_coarse)
        picklepath = PRE.pickledir + f"tos_ORAS5_{m_str}.pickle"
        PRE.save_pickle(picklepath, sst, sst_clim, sst_variance, sst_anom, sst_std,
                        sst_coarse, sst_coarse_clim, sst_coarse_variance, sst_coarse_anom, sst_coarse_std,
                        save_flag=save_flag)
        PRE.save_npy(m_str, sst, sst_clim, sst_variance, sst_anom, sst_std,
                     sst_coarse, sst_coarse_clim, sst_coarse_variance, sst_coarse_anom, sst_coarse_std,
                     save_flag=save_flag)


class Preprocess():
    def __init__(self):
        self.ulim, self.llim = 30, -30
        self.lt, self.ln = 120, 360
        self.upscale_rate = 5
        self.begin_year = 1958
        self.end_year = 2014
        self.tm = 57
        self.datadir = '/work/kajiyama/cdo/ORAS5'
        self.pickledir = "/work/kajiyama/preprocessed/ORAS5"
        self.npydir = "/work/kajiyama/cnn/transfer_input/tos/"

        self.first_file = self.datadir + "/sst_1958_Apr.nc"
        with Dataset(self.first_file, 'r') as nc:
            self.lons = nc.variables['lon'][:]
            self.lats = nc.variables['lat'][self.ulim:self.llim]
        self.lons_sub, self.lats_sub = np.meshgrid(self.lons[::self.upscale_rate],
                                                   self.lats[::self.upscale_rate])

    def _fill(self, x):
        f = ma.filled(x, fill_value=-99999)
        return f

    def _mask(self, x):
        m = ma.masked_where(x <= -99999, x)
        return m

    def _load(self, file):
        """
        reverse.shape = (120, 360)
        """
        ds = Dataset(file, 'r')
        var = ds.variables['sosstsst'][:]
        var = np.squeeze(var)
        reverse = var[::-1, :]
        extracted = reverse[self.ulim:self.llim, :]
        return extracted

    def make_var(self, m_str='Apr'):
        var_filled = np.empty((self.tm, self.lt, self.ln))
        for i, year in enumerate(np.arange(self.begin_year, self.end_year+1)):
            file = f"{self.datadir}/sst_{year}_{m_str}.nc"
            var_filled[i, :, :] = self._fill(self._load(file))
        return var_filled

    def anomaly(self, x):
        """
        sst_anom.shape = (57, 120, 360)
        """
        dup = x.copy()
        dup_masked = self._mask(dup)
        sst_anom = np.empty(dup.shape)
        clim_masked = np.mean(dup_masked, axis=0)
        anom_masked = dup_masked - clim_masked
        anom_filled = self._fill(anom_masked)
        sst_anom[:, :, :] = anom_filled
        return sst_anom

    def standardize(self, x):
        dup = x.copy()
        dup_masked = self._mask(dup)
        sst_std = np.empty(dup.shape)
        sst_clim = np.empty((dup.shape[1], dup.shape[2]))
        sst_variance = np.empty((dup.shape[1], dup.shape[2]))

        clim_masked = np.mean(dup_masked, axis=0)
        clim_filled = self._fill(clim_masked)
        sst_clim[:, :] = clim_filled

        variance_masked = np.std(dup_masked, axis=0)
        variance_filled = self._fill(variance_masked)
        sst_variance[:, :] = variance_filled

        std_masked = (dup_masked - clim_masked) / variance_masked
        std_filled = self._fill(std_masked)
        sst_std[:, :, :] = std_filled

        return sst_clim, sst_variance, sst_std

    def interpolation(self, x):
        dup = x.copy()
        dup_masked = self._mask(dup)
        lat_scale = int(self.lt/self.upscale_rate)
        lon_scale = int(self.ln/self.upscale_rate)
        sst_coarse = dup[:, :lat_scale, :lon_scale]
        for time in range(len(sst_coarse)):
            interp_masked = basemap.interp(dup_masked[time, :, :],
                                           self.lons, self.lats,
                                           self.lons_sub, self.lats_sub,
                                           order=0)
            interp_filled = self._fill(interp_masked)
            sst_coarse[time, :, :] = interp_filled
        return sst_coarse

    def save_pickle(self, picklepath,
                    raw, clim, variance, anom, std,
                    coarse, coarse_clim, coarse_variance, coarse_anom, coarse_std,
                    save_flag=False):
        save_dict = {"sst_raw": raw,
                     "sst_clim": clim,
                     "sst_variance": variance,
                     "sst_anom": anom,
                     "sst_std": std,
                     "sst_coarse": coarse,
                     "sst_coarse_clim": coarse_clim,
                     "sst_coarse_variance": coarse_variance,
                     "sst_coarse_anom": coarse_anom,
                     "sst_coarse_std": coarse_std,
                     }

        if save_flag is True:
            with open(picklepath, 'wb') as f:
                pickle.dump(save_dict, f)
            print(f'{picklepath } has been SAVED')
        else:
            print('sst picke file has ***NOT*** been saved yet')

    def save_npy(self, m_str,
                 raw, clim, variance, anom, std,
                 coarse, coarse_clim, coarse_variance, coarse_anom, coarse_std,
                 save_flag=False):
        if save_flag is True:
            np.save(self.npydir + f"tos_raw_{m_str}.npy", raw)
            np.save(self.npydir + f"tos_clim_{m_str}.npy", clim)
            np.save(self.npydir + f"tos_variance_{m_str}.npy", variance)
            np.save(self.npydir + f"tos_anom_{m_str}.npy", anom)
            np.save(self.npydir + f"tos_std_{m_str}.npy", std)
            np.save(self.npydir + f"tos_coarse_{m_str}.npy", coarse)
            np.save(self.npydir + f"tos_coarse_clim_{m_str}.npy", coarse_clim)
            np.save(self.npydir + f"tos_coarse_variance_{m_str}.npy", coarse_variance)
            np.save(self.npydir + f"tos_coarse_anom_{m_str}.npy", coarse_anom)
            np.save(self.npydir + f"tos_coarse_std_{m_str}.npy", coarse_std)
            print(f'{m_str} sst npy file has been SAVED')
        else:
            print('sst npy file has ***NOT*** been saved yet')


class Draw():
    def _imshow(self, image):
        plt.register_cmap('cc_rainbow', cc.cm.rainbow)
        cmap = plt.get_cmap('cc_rainbow', 10)
        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin='upper',
                         extent=img_extent,
                         transform=projection,
                         cmap=cmap,
                         )
        cbar = fig.colorbar(mat, ax=ax, orientation='horizontal')
        plt.show()


    def _plot_anom(self, image, reverse_flag=False):
        cmap = plt.cm.get_cmap('RdBu_r', 10)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin = 'upper',
                         extent=img_extent,
                         transform=projection,
                         norm=colors.CenteredNorm(),
                         cmap= cmap
                        )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal'
                            )
        plt.show()

    def _plot_std(self, image, reverse_flag=False):
        cmap = plt.cm.get_cmap('RdBu_r', 10)

        projection = ccrs.PlateCarree(central_longitude=180)
        img_extent = (-180, 180, -60, 60)
        fig = plt.figure()
        ax = plt.subplot(projection=projection)
        ax.coastlines()
        mat = ax.matshow(image,
                         origin = 'upper',
                         extent=img_extent,
                         transform=projection,
                         norm=colors.Normalize(vmin=-3, vmax=3),
                         cmap= cmap
                         )
        cbar = fig.colorbar(mat,
                            ax=ax,
                            orientation='horizontal'
                            )
        plt.show()


if __name__ == '__main__':
    main()
