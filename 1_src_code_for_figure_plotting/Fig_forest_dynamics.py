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
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, FuncFormatter

warnings.filterwarnings('ignore')

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
    dataset = None
    # data[data == data[0][0]] = np.nan
    return data,lon,lat,geotransform

def custom_colormap(i, j, n):
    # Normalize the indices to the range [0, 1]
    x = i / (n - 1)
    y = j / (n - 1)
    # Compute the distance from the center
    distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
    # Define the color components based on distance from center
    g = np.clip(distance + (x - 0.5), 0, 1)
    r = np.clip(distance + (y - 0.5), 0, 1)
    b = np.clip(1 - distance, 0, 1)
    return (r, g, b, 1)

def country_extraction(global_data,lat,lon,country_shp):
    file = xr.DataArray(global_data, coords=[('lat', lat), ('lon', lon), ('channel', [1, 2, 3, 4])])
    ds = xr.Dataset({'data': file})
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = ds.rename({'lon': 'x'})
    ds = ds.rename({'lat': 'y'})
    clipped = ds.rio.clip(country_shp.geometry.apply(mapping), country_shp.crs, drop=False)
    region_data = clipped.variables['data']
    return region_data
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

edge_file = '../0_data/2_Forest_edge_dynamics/edge_diff_per_range.tif'
area_file = '../0_data/2_Forest_edge_dynamics/area_diff_per_range.tif'

data1,lon1,lat1,transform1 = read_data(edge_file)
data2,lon2,lat2,transform2 = read_data(area_file)

mask = (data1>= -10) & (data1 <= 10) & (data2 >= -10) & (data2 <= 10)
data1[data1 < -1] = -1
data2[data2 < -1] = -1
data1[data1 > 1] = 1
data2[data2 > 1] = 1

print(np.nanmax(data1))
print(np.nanmin(data1))
print(np.nanmax(data2))
print(np.nanmin(data2))

data1_clipped = np.clip(data1, -0.2, 0.2)
data2_clipped = np.clip(data2, -0.2, 0.2)
data1 = None
data2 = None

data1_clipped_norm = (data1_clipped+0.2)/0.4
data2_clipped_norm = (data2_clipped+0.2)/0.4
data1_clipped = None
data2_clipped = None

print(np.nanmax(data1_clipped_norm))
print(np.nanmin(data1_clipped_norm))
print(np.nanmax(data2_clipped_norm))
print(np.nanmin(data2_clipped_norm))

n = 40  # Number of steps (same as before)
edge_indices = (data1_clipped_norm * (n - 1)).astype(int)
area_indices = (data2_clipped_norm * (n - 1)).astype(int)

bivariate_colors = np.empty((n, n, 4))
for i in range(n):
    for j in range(n):
        bivariate_colors[i, j] = custom_colormap(i, j, n)

final_rgba = np.zeros((*data1_clipped_norm.shape, 4))
for x in tqdm(range(final_rgba.shape[0])):
    for y in range(final_rgba.shape[1]):
        if mask[x, y]:
            i = area_indices[x, y]
            j = edge_indices[x, y]
            final_rgba[x, y] = bivariate_colors[i, j]
        else:
            final_rgba[x, y] = [0,0,0,0]  
        
world_filepath = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
world = gpd.read_file(world_filepath)
countries = ['United States of America','China', 'Taiwan','Russia','Brazil','Canada']
selected_countries = world[world['SOVEREIGNT'].isin(countries)]
china = selected_countries[(selected_countries['SOVEREIGNT']=='China')|(selected_countries['SOVEREIGNT']=='Taiwan')]
us = selected_countries[(selected_countries['SOVEREIGNT']=='United States of America')&(selected_countries['TYPE']=='Country')]
russia = selected_countries[selected_countries['SOVEREIGNT']=='Russia']
brazil = selected_countries[selected_countries['SOVEREIGNT']=='Brazil']
canada = selected_countries[selected_countries['SOVEREIGNT']=='Canada']
contries_shp = [china, us, russia, brazil, canada]

fig = plt.figure(figsize = (10,10))
gs = gridspec.GridSpec(21, 6)
config = {"font.family":'Helvetica'}
plt.subplots_adjust(hspace =0,wspace =0.25)
plt.rcParams.update(config)

####################### global data
ax = plt.subplot(gs[0:10, :],projection = ccrs.Robinson(central_longitude=0.0))

base_map(ax)
grd = ax.gridlines(draw_labels=True, xlocs=range(-180, 181, 90), ylocs=range(-60, 61, 30), color='gray',linestyle='--', linewidth=0.5, zorder=7)
grd.top_labels = False

ax.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                bottom=True,left=True,top=False,right=False)
ax.spines['geo'].set_linewidth(0.7)
ax.text(-0.05,1.03, 'A  Global Forest Edge Dynamics', transform=ax.transAxes, fontsize = 9,fontweight='bold')
ax.imshow(final_rgba, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(),zorder = 3)
china.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
us.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
russia.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
brazil.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
canada.plot(ax = ax,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)

axx = ax.inset_axes([0.08,0.18,0.25,0.25])
axx.set_aspect('equal', adjustable='box')

axx.imshow(bivariate_colors, origin='lower', extent=[0, 1, 0, 1])

axx.tick_params(axis='both',which='major',bottom=False, left=False,top=False,right=False,labelleft = False, labelbottom = False)
axx.spines[['top', 'right','left','bottom']].set_visible(False)
# axx.set_title('Forest Dynamics',fontsize = 8)
axx.set_xlabel('edge changes',fontsize = 6,labelpad = 8)
axx.set_ylabel('area chenges',fontsize = 6,labelpad = 8)
axx.text(1,-0.05, '+20%', transform=axx.transAxes, fontsize = 6)
axx.text(-0.15,0.78, '+20%', transform=axx.transAxes, fontsize = 6,rotation= 90)
axx.text(-0.22,-0.22, '-20%', transform=axx.transAxes, fontsize = 6,rotation= 45)

####################### regional data
ax1 = plt.subplot(gs[11:16, 0:2],projection = ccrs.PlateCarree())
ax2 = plt.subplot(gs[11:16, 2:4],projection = ccrs.PlateCarree())
ax4 = plt.subplot(gs[11:16, 4:6],projection = ccrs.PlateCarree())
ax3 = plt.subplot(gs[16:21, 0:4],projection = ccrs.PlateCarree())
ax5 = plt.subplot(gs[16:21, 4:6],projection = ccrs.PlateCarree())

ax1.text(0,1.05, 'B  Forest Edge Dynamics in Different Countries', transform=ax1.transAxes, fontsize = 9,fontweight='bold')
ax1.text(0.02,0.9, 'B.1 China', transform=ax1.transAxes, fontsize = 8)
ax2.text(0.02,0.9, 'B.2 U.S. mainland', transform=ax2.transAxes, fontsize = 8)
ax4.text(0.02,0.9, 'B.3 Brazil', transform=ax4.transAxes, fontsize = 8)
ax3.text(0.02,0.9, 'B.4 Russia', transform=ax3.transAxes, fontsize = 8)
ax5.text(0.02,0.9, 'B.5 Canada', transform=ax5.transAxes, fontsize = 8)

d1 = country_extraction(final_rgba,lat1,lon1,china)
d2 = country_extraction(final_rgba,lat1,lon1,us)
d3 = country_extraction(final_rgba,lat1,lon1,russia)
d4 = country_extraction(final_rgba,lat1,lon1,brazil)
d5 = country_extraction(final_rgba,lat1,lon1,canada)

base_map(ax1)
base_map(ax2)
base_map(ax3)
base_map(ax4)
base_map(ax5)

ax1.imshow(d1, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3)
ax2.imshow(d2, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3)
ax3.imshow(d3, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3)
ax4.imshow(d4, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3)
ax5.imshow(d5, extent = [lon1.min(), lon1.max(), lat1.min(), lat1.max()],transform=ccrs.PlateCarree(), zorder = 3)

china.plot(ax = ax1,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
us.plot(ax = ax2,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
russia.plot(ax = ax3,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
brazil.plot(ax = ax4,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)
canada.plot(ax = ax5,transform=ccrs.PlateCarree(),color = 'none',edgecolor="k",lw = 0.5)

ax1.set_extent([69, 137, 14, 50])
ax2.set_extent([-128, -60, 18, 54])
ax3.set_extent([20, 179, 40, 75])
ax4.set_extent([-90, -22, -31, 5])
ax5.set_extent([-145, -50, 30, 85])

ax1.set_xticks([80,100,120])
ax2.set_xticks([-120,-100,-80,-60])
ax3.set_xticks([30,60,90,120,150])
ax4.set_xticks([-80,-60,-40])
ax5.set_xticks([-140,-120,-100,-80, -60])

ax1.set_yticks([20,35,50])                 
ax2.set_yticks([30,50])
ax3.set_yticks([45,60,75])
ax4.set_yticks([-30,-15, 0])
ax5.set_yticks([40,60,80])

ax1.set_yticklabels([x.get_text() for x in ax1.get_yticklabels()],rotation=90, va='center')
ax2.set_yticklabels([x.get_text() for x in ax2.get_yticklabels()],rotation=90, va='center')
ax3.set_yticklabels([x.get_text() for x in ax3.get_yticklabels()],rotation=90, va='center')
ax4.set_yticklabels([x.get_text() for x in ax4.get_yticklabels()],rotation=90, va='center')
ax5.set_yticklabels([x.get_text() for x in ax5.get_yticklabels()],rotation=90, va='center')

ax1.xaxis.set_major_formatter(LongitudeFormatter()) 
ax2.xaxis.set_major_formatter(LongitudeFormatter()) 
ax3.xaxis.set_major_formatter(LongitudeFormatter()) 
ax4.xaxis.set_major_formatter(LongitudeFormatter()) 
ax5.xaxis.set_major_formatter(LongitudeFormatter()) 
                   
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax4.yaxis.set_major_formatter(LatitudeFormatter())
ax5.yaxis.set_major_formatter(LatitudeFormatter())

ax1.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax2.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax3.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax4.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)
ax5.tick_params(axis='both',which='major',labelsize=8,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True, bottom=True,left=True,top=False,right=False)

ax1.spines['geo'].set_linewidth(0.7)
ax2.spines['geo'].set_linewidth(0.7)
ax3.spines['geo'].set_linewidth(0.7)
ax4.spines['geo'].set_linewidth(0.7)
ax5.spines['geo'].set_linewidth(0.7)

######### statistics
df = pd.read_csv('../0_data/2_Forest_edge_dynamics/processed_country_data_with_area.csv')
countries = ['United States of America','China', 'Taiwan','Russia','Brazil','Canada']
df = df[df['country'].isin(countries)]
df = df[['country','increase increase','increase decrease','decrease increase','decrease decrease']]

china_df = df[(df['country'] == 'China')|(df['country'] == 'Taiwan')]
china_df = pd.DataFrame(china_df.sum(axis = 0)).T
china_df['country'] = 'China'

other_df = df[(df['country'] == 'United States of America')|(df['country'] == 'Russia')|
              (df['country'] == 'Canada')|(df['country'] == 'Brazil')]
df = pd.concat([china_df,other_df], axis = 0)
var = True
for i in ['China', 'United States of America','Brazil', 'Russia','Canada']:
    temp = df[df['country'] == i]
    temp = temp[['increase increase','increase decrease','decrease increase','decrease decrease']]
    temp = temp.T
    temp.reset_index(inplace = True)
    temp.columns = ['dynamics','values']
    temp['country'] = 'U.S.' if i == 'United States of America' else i
    if var:
        final_df = temp
        var = False
    else:
        final_df = pd.concat([final_df,temp], axis = 0)    
final_df.reset_index(drop = True, inplace = True)

axes = inset_axes(ax, width="60%", height="35%", loc='lower left', bbox_to_anchor=(-0.02, -1.65, 1, 1),bbox_transform=ax.transAxes)
axes.text(-0.06,1.25, 'C  Forest Edge Dynamics Statistics', transform=axes.transAxes, fontsize = 9,fontweight='bold')
sns.barplot(x='country', y= 'values', hue = 'dynamics', ax = axes, data=final_df,saturation=0.9, errcolor='k',errwidth = 0.7,
            palette=['yellow','orangered','limegreen','dimgrey'],capsize = 0.07,edgecolor="k",linewidth = 0.7)

axes.spines[['top', 'right']].set_visible(False)
axes.spines[['bottom','left']].set_linewidth(1)
axes.tick_params(axis='both',which='major',labelsize=10,direction='in',pad=5,bottom=True, left=True,top=False,right=False)
axes.set_xlabel('Countries',fontsize = 10,labelpad = 2)
axes.set_ylabel('Forest Edge Length (km)',fontsize = 10,labelpad = 2)
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 

legend = axes.legend()
handles = legend.legendHandles
labels=['Forest edge length increase due to area increase', 'Forest edge length increase due to area decrease',
        'Forest edge length decrease due to area increase','Forest edge length decrease due to area decrease']
axes.legend(handles = handles, labels = labels, loc = 'lower right',fontsize=8,facecolor= 'none',edgecolor = 'none',bbox_to_anchor=(1.78, -0.06))

plt.savefig('../2_figures/Figure_2_Forest_edge_dynamics_Science_version_V1.png', dpi=600, bbox_inches='tight')

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')