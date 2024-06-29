import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cf
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import matplotlib as mpl
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
from PIL import ImageColor
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
import geopandas as gpd
import xarray as xr
from shapely.geometry import mapping
import rioxarray
from matplotlib import gridspec
import rasterio
import math
import cartopy.io.shapereader as shpreader
warnings.filterwarnings('ignore')

def read_data(image_data):
    dataset = gdal.Open(image_data)
    geotransform = dataset.GetGeoTransform()
    origin_x,origin_y = geotransform[0],geotransform[3]
    pixel_width, pixel_height = geotransform[1],geotransform[5]
    
    width, height = dataset.RasterXSize, dataset.RasterYSize
    lon = origin_x + pixel_width * np.arange(width)
    lat = origin_y + pixel_height * np.arange(height)
    
    data = dataset.GetRasterBand(1).ReadAsArray()
    if data.max()>10000:
        data = data/1000
    data[data < 0.01] = np.nan
    min_value = np.nanpercentile(data, 2.5)
    max_value = np.nanpercentile(data, 97.5)
    return data,lon,lat,min_value,max_value
def base_map(ax):
    states_provinces = cf.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',
                                              scale='50m',facecolor='none')
    ax.add_feature(cf.LAND,alpha=0.1)
    ax.add_feature(cf.BORDERS, linestyle='--',lw=0.4, alpha=0.5)
    ax.add_feature(cf.LAKES, alpha=0.5)
    ax.add_feature(cf.OCEAN,alpha=0.1,zorder = 2)
    ax.add_feature(cf.COASTLINE,lw=0.4)
    ax.add_feature(cf.RIVERS,lw=0.2)
    ax.add_feature(states_provinces,lw=0.2,edgecolor='gray')
    return

path = "../0_data/2_Forest_edge_dynamics/"
file_name = ["stable.tif","ii.tif","id.tif","di.tif","dd.tif"]
files = [f"{path}{x}" for x in file_name]

cmaps = ["YlGnBu", "magma", "magma","magma", "magma"]
text = ["(a) Stable forest edge", "(b) Forest edge increase due to forest area increase", "(c) Forest edge increase due to forest area decrease",
        "(d) Forest edge decrease due to forest area increase","(e) Forest edge decrease due to forest area decrease"]

fig = plt.figure(figsize = (12,10))
gs = gridspec.GridSpec(15, 20)
config = {"font.family":'Helvetica'}
plt.subplots_adjust(hspace =0, wspace = 0)
plt.rcParams.update(config)

k = 0
for file in files:
    data,lon,lat,min_value,max_value = read_data(file)
    
    if k == 0:
        ax = plt.subplot(gs[0:5, 5:15],projection = ccrs.Robinson(central_longitude=0.0))
    elif k == 1:
        ax = plt.subplot(gs[5:10, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
    elif k == 2:
        ax = plt.subplot(gs[5:10, 10:20],projection = ccrs.Robinson(central_longitude=0.0))
    elif k == 3:
        ax = plt.subplot(gs[10:15, 0:10],projection = ccrs.Robinson(central_longitude=0.0))
    else:
        ax = plt.subplot(gs[10:15, 10:20],projection = ccrs.Robinson(central_longitude=0.0))
        
    ax.set_extent([-179.999, 179.999, -90, 90])
    base_map(ax)
    grd = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
    grd.top_labels = False

    ax.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                    bottom=True,left=True,top=False,right=False)
    ax.spines['geo'].set_linewidth(0.7)

    lons, lats = np.meshgrid(lon, lat)
    p = ax.pcolormesh(lons,lats,data,transform=ccrs.PlateCarree(),cmap=cmaps[k],vmin = int(min_value), vmax = math.ceil(max_value), zorder = 3)

    lev = np.linspace(int(min_value),math.ceil(max_value),4)
    cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.06, shrink=0.15,anchor = (0.55,3.0))
    cbar = plt.colorbar(p, cax=cax, orientation='horizontal',ticks=lev)
    cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7,pad = 0.1,length=1)
    cbar.set_label('Forest Edge (km)',fontsize = 8,labelpad=2)
    ax.text(-0.05,1.08, text[k], transform=ax.transAxes, fontsize = 9,fontweight='bold')
    k = k+1

plt.savefig('../2_figures/Figure S2_Forest edge changes.png', dpi=400, bbox_inches='tight')