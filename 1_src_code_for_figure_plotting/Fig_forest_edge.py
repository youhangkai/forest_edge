import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import warnings
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
import matplotlib.ticker as mticker
from matplotlib import ticker
import matplotlib.ticker as mtick
import matplotlib as mpl
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
from matplotlib.ticker import ScalarFormatter
warnings.filterwarnings('ignore')
import gc
import os


start_t = datetime.datetime.now()
print('start:', start_t)

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
    # data[data < 0] = 0         #######
    return data,lon,lat


years = [2000,2020]
idx = ['A ','B ']

fig = plt.figure(figsize = (10,8))
config = {"font.family":'Helvetica'}
plt.subplots_adjust(hspace =0.3)
plt.rcParams.update(config)

k = 1
for year in years:
    if year == 2000:
        data,lon,lat = read_data(f"../0_data/1_Forest_edge/{year}Edge.tif")
    else:
        data1,lon,lat = read_data(f"../0_data/1_Forest_edge/{year-20}Edge.tif")
        data2,lon,lat = read_data(f"../0_data/1_Forest_edge/{year}Edge.tif")
        data = data2-data1
        del data1, data2
        gc.collect()
    
    lon_sum, lat_sum = np.nansum(data, axis = 0), np.nansum(data, axis = 1)
    
    lon_sum = np.split(lon_sum, len(lon_sum)/10)
    lon_sum = np.array([subarray.sum() for subarray in lon_sum])
    lat_sum = np.split(lat_sum, len(lat_sum)/10)
    lat_sum = np.array([subarray.sum() for subarray in lat_sum])
    lat_statistical,lon_statistical = zoom(lat,0.1),zoom(lon,0.1)
    
    # data = zoom(data,0.1)                          #####
    # lat,lon = zoom(lat,0.1),zoom(lon,0.1)          #####
    # data[data < 0.01] = np.nan                     
    
    ax = fig.add_subplot(2,1,k,projection = ccrs.Robinson(central_longitude=0.0))

    states_provinces = cf.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',
                                              scale='110m',facecolor='none')
    ax.set_extent([-179.999, 179.999, lat.min(), lat.max()])
    ax.add_feature(cf.LAND,alpha=0.2)
    ax.add_feature(cf.BORDERS, linestyle='--',alpha=0.5)
    ax.add_feature(cf.LAKES, alpha=0.5)
    ax.add_feature(cf.COASTLINE,lw=0.5)
    ax.add_feature(cf.RIVERS,lw=0.2)
    ax.add_feature(states_provinces,lw=0.3,edgecolor='gray')

    ax.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = False, labelbottom = False,
                   bottom=False,left=False,top=False,right=False)
    ax.spines['geo'].set_linewidth(0)
    
    if year ==2000:
        max_value = 0
        min_value = 15
        lev = np.arange(0,15.1,5)
        ax.text(-0.05,1.05, f'{idx[k-1]} Global Forest Edge Length in {year}', transform=ax.transAxes, fontsize = 9,fontweight='bold')
    else:
        max_value = 2
        min_value = -2
        lev = np.arange(-2,2.1,1)
        ax.text(-0.05,1.05, f'{idx[k-1]} Global Forest Edge Length Difference from 2020 to 2000', transform=ax.transAxes, fontsize = 9,fontweight='bold')
    
    xy_lables = {2000:'Forest Edge Length (km)    ',2020:"\u0394 Forest Edge Length (km)  "}
    cmaps = {2000:'YlGnBu',2020:"coolwarm"}
    lons, lats = np.meshgrid(lon, lat)
    p = ax.pcolormesh(lons,lats,data,transform=ccrs.PlateCarree(),cmap=cmaps[year],vmin = min_value, vmax = max_value)

    cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.06, shrink=0.15,anchor = (0.92,0.65))
    cbar = plt.colorbar(p, cax=cax, orientation='horizontal',ticks=lev)
    cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7,pad = 0.1,length=1)
    cbar.set_label(xy_lables[year],fontsize = 8,labelpad=2)
        
    #####
    ax1 = inset_axes(ax, width="88%", height="40%", loc='lower center', bbox_to_anchor=(0, -0.35, 1, 1),bbox_transform=ax.transAxes)
    ax2 = inset_axes(ax, width="15%", height="100%", loc='upper right', bbox_to_anchor=(0.2, 0, 1, 1),bbox_transform=ax.transAxes)

    ax1.plot(lon_statistical, lon_sum, c = 'dodgerblue', lw = 1.5)
    ax1.set_xlim(lon_statistical.min(),lon_statistical.max())
    ax1.xaxis.set_major_formatter(LongitudeFormatter())     
    ax1.set_facecolor('none')
    ax1.tick_params(axis='both',which='major',labelsize=7,direction='in',pad=1,bottom=True, left=True,top=False,right=False)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xlabel('Longitude',fontsize = 8,labelpad = 5)
    ax1.set_ylabel(xy_lables[year],fontsize = 8,labelpad = 5)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax1.yaxis.set_major_formatter(formatter)

    # ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

    ax1.yaxis.get_offset_text().set_size(7) 

    ax2.plot(lat_sum,lat_statistical, c = 'dodgerblue', alpha=0.8,lw = 1.5)
    ax2.set_facecolor('none')
    ax2.spines[['bottom', 'right']].set_visible(False)
    ax2.tick_params(axis='both',which='major',labelsize=7,direction='in',labeltop=True, 
                    labelbottom=False,pad=1,bottom=False, left=True,top=True,right=False)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_ylim(lat_statistical.min(),lat_statistical.max())
    ax2.set_xlabel(xy_lables[year],fontsize = 8,labelpad = 5)
    ax2.xaxis.set_major_formatter(formatter)

    ax2.xaxis.get_offset_text().set_size(7) 
    offset_label = ax2.xaxis.get_offset_text()
    pos = {2000:1.65, 2020:1.8}
    offset_label.set_position((pos[year],0))

    ax2.set_ylabel('Latitude',fontsize = 8,labelpad = 5)
    ax2.set_yticklabels([x.get_text() for x in ax2.get_yticklabels()],rotation=90, va='center')
    ax2.yaxis.set_major_formatter(LatitudeFormatter())
    
    k = k+1
plt.savefig('../2_figures/Figure 1_global edge mapping_SCIENCE_FORMAT_new.png', dpi=600, bbox_inches='tight') 

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')