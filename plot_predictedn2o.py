"""
File: plot_predictedn2o.py
-----------------------------
Created on Weds March 12, 2025

Plot maps of predicted n2o and uncertainties;
plot histogram of uncertainties.

@author: Colette Kelly (colette.kelly@whoi.edu)
"""

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.parquet as pq
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_data(path_to_data):
    t = pq.read_table(f"{path_to_data}/n2opredictions.parquet")
    df = t.to_pandas()

    return df

def plotfronts(ax,lon_pf,lat_pf,lon_siz,lat_siz,lon_saf,lat_saf,lon_stf,lat_stf):
    ax.plot(lon_pf[0:1409],lat_pf[0:1409],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_pf[1410:len(lon_pf)],lat_pf[1410:len(lon_pf)],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_siz[0:537],lat_siz[0:537],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_siz[538:len(lon_siz)],lat_siz[538:len(lon_siz)],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_saf[0:2016],lat_saf[0:2016],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_saf[2017:2911],lat_saf[2017:2911],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_stf[0:1871],lat_stf[0:1871],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())
    ax.plot(lon_stf[1872:len(lon_stf)],lat_stf[1872:len(lon_stf)],c="k",lw=2,zorder=4,transform=ccrs.PlateCarree())

def setupmap(ax):
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,facecolor='gray',zorder=4)
    #g = ax.gridlines(linestyle='--',zorder=5, draw_labels=False)
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    import matplotlib.path as mpath
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.OCEAN, facecolor='lightskyblue')

def plotmaps(df):
    # load fronts from Gray et al. 2018
    mat = scipy.io.loadmat('datasets/fronts_Gray.mat')
    lat_pf=mat['lat_pf'][0] #Polar Front
    lon_pf=mat['lon_pf'][0] #Polar Front
    lat_siz=mat['lat_siz'][0] #Seasonal Ice Zone
    lon_siz=mat['lon_siz'][0] #Seasonal Ice Zone
    lat_saf=mat['lat_saf'][0] #Subantarctic Front
    lon_saf=mat['lon_saf'][0] #Subantarctic Front
    lat_stf=mat['lat_stf'][0] #Subtropical Front
    lon_stf=mat['lon_stf'][0] #Subtropical Front

    fig = plt.figure(figsize=(7.09, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1], height_ratios=[1], wspace=0.05, hspace=0.05)

    ax1 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
    ax1.text(0.05, 0.95, "a", fontweight="bold",
        horizontalalignment = "left",
        verticalalignment = "top",
        transform = ax1.transAxes)
    setupmap(ax1)

    pltdf = df[df["PRES_ADJUSTED"]<20].sort_values("pN2O_pred", ascending=True)
    lats = np.array(pltdf.LATITUDE)
    lons = np.array(pltdf.LONGITUDE)
    n2o = np.array(pltdf.pN2O_pred)
    cax = ax1.scatter(lons, lats,c = n2o, cmap='viridis',
                    vmax = 550, vmin = 300,
                     transform=ccrs.PlateCarree())
    plotfronts(ax1,lon_pf,lat_pf,lon_siz,lat_siz,lon_saf,lat_saf,lon_stf,lat_stf)
    bounds=[i for i in np.arange(300,600,50)]
    cbar = fig.colorbar(cax, ax=ax1, orientation='horizontal', pad=0.05,
                        extend='both', 
                        boundaries=bounds,
                        ticks = [i for i in np.arange(300,600,50)]
                       )
    cbar.set_label('mean predicted $pN_2O$ (natm)', fontsize = 12)
    cbar.ax.tick_params(labelsize=12)
    ax1.tick_params(direction="in", top=True, right=True)

    ax2 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
    ax2.text(0.05, 0.95, "b", fontweight="bold",
        horizontalalignment = "left",
        verticalalignment = "top",
        transform = ax2.transAxes)
    setupmap(ax2)
    pltdf = df[df["PRES_ADJUSTED"]<20].sort_values("pN2O_predstd", ascending=True)
    lats = np.array(pltdf.LATITUDE)
    lons = np.array(pltdf.LONGITUDE)
    n2o = np.array(pltdf.pN2O_predstd)
    cax = ax2.scatter(lons, lats,c = n2o, cmap='viridis',
                    vmax = 50, vmin = 0,
                     transform=ccrs.PlateCarree())
    plotfronts(ax2,lon_pf,lat_pf,lon_siz,lat_siz,lon_saf,lat_saf,lon_stf,lat_stf)
    bounds=[i for i in np.arange(0,60,10)]
    cbar = fig.colorbar(cax, ax=ax2, orientation='horizontal', pad=0.05,
                        extend='both', 
                        boundaries=bounds,
                        ticks = [i for i in np.arange(0,60,10)]
                       )
    cbar.set_label('$\sigma$ (natm)', fontsize = 12)
    cbar.ax.tick_params(labelsize=12)
    ax2.tick_params(direction="in", top=True, right=True)
    plt.savefig("figures/FigureS18.png", dpi=300, bbox_inches="tight")
    plt.show()

def main():
	df = load_data('datasets')
	plotmaps(df)

if __name__ == "__main__":
	main()
