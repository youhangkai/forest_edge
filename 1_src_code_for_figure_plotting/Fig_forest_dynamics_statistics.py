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
from scipy import stats
from PIL import ImageColor
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
import geopandas as gpd
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
from shapely.geometry import mapping
from matplotlib import gridspec
from matplotlib.lines import Line2D
import cartopy.io.shapereader as shpreader
warnings.filterwarnings('ignore')

# import forest edge dynamics statistics data.
df = pd.read_csv('../0_data/2_Forest_edge_dynamics/processed_country_data_with_area.csv')
# import the world shapefiles
world_filepath = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
reader = shpreader.Reader(world_filepath)
countries = [country.attributes['NAME'] for country in reader.records()]
continents = [country.attributes['CONTINENT'] for country in reader.records()]
country_to_continent = dict(zip(countries, continents))
df['continent'] = df['country'].map(country_to_continent)

colors = {'Asia': 'red','Africa': 'green','North America': 'blue','South America': 'yellow','Europe': 'purple','Oceania': 'cyan','Antarctica': 'gray'}
df['Edge Change'] = df['Edge Change'].astype(float)
df['Area Change'] = df['Area Change'].astype(float)
df = df.dropna(subset=['Edge Change', 'Area Change'])

edge_change_threshold = 3 * df['Edge Change'].std()
area_change_threshold = 3 * df['Area Change'].std()
subset_df = df[(abs(df['Edge Change']) <= edge_change_threshold)& (abs(df['Area Change']) <= area_change_threshold)]


country_data = df.copy()
country_data = country_data[['country','Total Forest Area 2000','Total Forest Area 2020','forest edge 2000','forest edge 2020']]

selected_rows =country_data[(country_data['country'] == 'China') | (country_data['country'] == 'Taiwan')]
filtered_country_data = country_data[~((country_data['country'] == 'China') | (country_data['country'] == 'Taiwan'))]
new_china = pd.DataFrame(selected_rows.sum(axis = 0)).T
new_china['country'] = 'China'

country_data = pd.concat([filtered_country_data, new_china],axis = 0)
country_data.reset_index(drop = True, inplace = True)

# Taking logarithms of the 'Total Forest Area 2000' and 'forest edge 2020' for power-law fitting
log_forest_area_2000 = np.log(country_data['Total Forest Area 2000'].astype(float))
log_forest_edge_2000 = np.log(country_data['forest edge 2000'].astype(float))
slope_2000, intercept_2000, r_value_2000, p_value_2000, std_err_2000 = stats.linregress(log_forest_edge_2000, log_forest_area_2000)

log_estimated_forest_area_2000 = intercept_2000 + slope_2000 * log_forest_edge_2000
estimated_forest_area_2000 = np.exp(log_estimated_forest_area_2000)

country_data['log_residuals2000'] = log_forest_area_2000 - log_estimated_forest_area_2000
country_data['log_fragmentation_rank2000'] = country_data['log_residuals2000'].abs().rank(method='min', ascending=True)

### 2020
log_forest_area_2020 = np.log(country_data['Total Forest Area 2020'].astype(float))
log_forest_edge_2020 = np.log(country_data['forest edge 2020'].astype(float))
slope_2020, intercept_2020, r_value_2020, p_value_2020, std_err_2020 = stats.linregress(log_forest_edge_2020, log_forest_area_2020)

log_estimated_forest_area_2020 = intercept_2020 + slope_2020 * log_forest_edge_2020
estimated_forest_area_2020 = np.exp(log_estimated_forest_area_2020)

country_data['log_residuals2020'] = log_forest_area_2020 - log_estimated_forest_area_2020
country_data['log_fragmentation_rank2020'] = country_data['log_residuals2020'].abs().rank(method='min', ascending=True)

def get_color1(deviation):
    max_deviation = max(abs(country_data['log_residuals2000']))
    color_intensity = np.abs(deviation) / max_deviation
    if deviation > 0:
        return (1, 1-color_intensity, 1-color_intensity)  # shades of red
    elif deviation < 0:
        return (1-color_intensity, 1-color_intensity, 1)  # shades of blue
    else:
        return 'grey'
    
def get_color2(deviation):
    max_deviation = max((country_data['log_residuals2020']-country_data['log_residuals2000']))
    color_intensity = np.abs(deviation) / max_deviation
    if color_intensity > 1:
        color_intensity = 1
    if deviation < 0:
        return (1, 1-color_intensity, 1-color_intensity)  # shades of red
    elif deviation > 0:
        return (1-color_intensity, 1-color_intensity, 1)  # shades of blue
    else:
        return 'white'
def base_map(ax):
    states_provinces = cf.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',
                                              scale='50m',facecolor='none')
    ax.set_extent([-180, 180, -60, 85])
    ax.add_feature(cf.LAND,alpha=0.1)
    ax.add_feature(cf.BORDERS, linestyle='--',lw=0.4, alpha=0.5)
    ax.add_feature(cf.LAKES, alpha=0.5)
    ax.add_feature(cf.OCEAN,alpha=0.05)
    ax.add_feature(cf.COASTLINE,lw=0.4)
    ax.add_feature(cf.RIVERS,lw=0.2)
    ax.add_feature(states_provinces,lw=0.2,edgecolor='gray')
    ax.set_xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])                      
    ax.set_yticks([-30,0, 30, 60])
    ax.set_yticklabels([x.get_text() for x in ax.get_yticklabels()],rotation=90, va='center')
    ax.xaxis.set_major_formatter(LongitudeFormatter())                    
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both',which='major',labelsize=9,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                    bottom=True,left=True,top=False,right=False)
    ax.spines['geo'].set_linewidth(0.7)
    return
def rsquared(x, y): 
    """Return the metriscs coefficient of determination (R2)
    Parameters:
    -----------
    x (numpy array or list): Predicted variables
    y (numpy array or list): Observed variables
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a

world_filepath = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
countries_shp = gpd.read_file(world_filepath)
# countries_shp = countries_shp.iloc[0:10,:]



fig,(ax1,ax2) = plt.subplots(1,2, figsize = (9,3))
config = {"font.family":'Helvetica'}
plt.subplots_adjust(hspace =0.2,wspace =0.2)
plt.rcParams.update(config)

# bubble sizes
min_size = 10
max_size = 500

# calculate the bubble size for each country
min_area = df['forest edge 2000'].min()
max_area = df['forest edge 2000'].max()
bubble_sizes = ((df['forest edge 2000'] - min_area) / (max_area - min_area) * (max_size - min_size) + min_size)

for continent, color in colors.items():
    subset = subset_df[subset_df['continent'] == continent]
    subset_bubble_sizes = bubble_sizes[subset.index] # ?????????
    ax1.scatter(subset['Edge Change'], subset['Area Change'], s=subset_bubble_sizes, color=color, alpha=0.6, edgecolors="w", linewidth=0.5)
    ax1.scatter([], [], s=150, color=color, label=continent, alpha=0.6, edgecolors="w", linewidth=0.5)

# ax1.text(0,1.05, '(a) Forest Dynamics (Country level)', transform=ax1.transAxes, fontsize = 10,fontweight='bold')
# ax1.text(0.01, 0.95, 'The size of bubbles indicate the edge of 2000',transform=ax1.transAxes, fontsize = 8)
ax1.tick_params(axis='both',which='major',labelsize=9,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,bottom=True,left=True,top=False,right=False)

ax1.set_xlabel('Edge Change (2020-2000)/2000',fontsize=9,labelpad = 1)
ax1.set_ylabel('Area Change (2020-2000)/2000',fontsize=9,labelpad = 1)

legend = ax1.legend()
handles = legend.legendHandles
labels = [text.get_text() for text in legend.get_texts()]
original_colors = [handle.get_facecolor() for handle in handles]
# new_handles = [Line2D([0], [0], marker='o', label=label, color=color, markersize=6, linestyle='None') for label, color in zip(labels, original_colors)]

# Creating new handles for the legend, excluding 'Antarctica'
new_handles = [Line2D([0], [0], marker='o', label=label, color=color, markersize=6, linestyle='None')
               for label, color in zip(labels, original_colors) if label != 'Antarctica']

legend.remove()
# ax1.legend(handles=new_handles, labels=labels, title='Continent', bbox_to_anchor=(0, 0.97),
#           title_fontsize='small', loc='upper left', fontsize=7, facecolor='none', edgecolor='none')

ax1.legend(handles=new_handles, labels=[label for label in labels if label != 'Antarctica'], title='Continent', bbox_to_anchor=(0, 0.97),
           title_fontsize='small', loc='upper left', fontsize=7, facecolor='none', edgecolor='none')

ax1.axvline(0, color='black', linestyle='-', linewidth=1)  # x=0 line
ax1.axhline(0, color='black', linestyle='-', linewidth=1)  # y=0 line

axes = inset_axes(ax1, width="30%", height="30%", loc='lower right', bbox_to_anchor=(-0.02, 0.55, 1, 1),bbox_transform=ax1.transAxes)
for continent, color in colors.items():
    outliers = df[(abs(df['Edge Change']) > edge_change_threshold)
        | (abs(df['Area Change']) > area_change_threshold)]
    subset = outliers[outliers['continent'] == continent]
    subset_bubble_sizes = bubble_sizes[subset.index] # ?????????
    axes .scatter(subset['Edge Change'], subset['Area Change'], s=subset_bubble_sizes, color=color, alpha=0.6, edgecolors="w", linewidth=0.5)
    
axes.set_title('Outliers',fontsize = 8)
axes.axvline(0, color='black', linestyle='-', linewidth=1) 
axes.axhline(0, color='black', linestyle='-', linewidth=1)
axes.tick_params(axis='both',which='major',labelsize=6,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,bottom=True,left=True,top=False,right=False)

########################################
ax2.scatter(df['forest edge 2000'], df['Total Forest Area 2000'], color='orangered', label='2000', alpha=0.4, edgecolors='w', linewidth=0.5, marker = "P",s = 50)
ax2.scatter(df['forest edge 2020'], df['Total Forest Area 2020'], color='dodgerblue', label='2020', alpha=0.4, edgecolors='w', linewidth=0.5,marker = "o", s = 50)
coeffs_2000 = np.polyfit(np.log(df['forest edge 2000']), np.log(df['Total Forest Area 2000']), 1)
ax2.plot(df['forest edge 2000'], np.exp(coeffs_2000[1]) * df['forest edge 2000'] ** coeffs_2000[0], color='black', linestyle='-')
coeffs_2020 = np.polyfit(np.log(df['forest edge 2020']), np.log(df['Total Forest Area 2020']), 1)
#ax2.plot(df['forest edge 2020'], np.exp(coeffs_2020[1]) * df['forest edge 2020'] ** coeffs_2020[0], color='dodgerblue', linestyle='-.')

a_2000 = np.exp(coeffs_2000[1])
b_2000 = coeffs_2000[0]
a_2020 = np.exp(coeffs_2020[1])
b_2020 = coeffs_2020[0]

# ax2.text(0,1.05, '(b) Forest Edge --- Area Relationships', transform=ax2.transAxes, fontsize = 10,fontweight='bold')
ax2.text(0.02, 0.93, f"2000: Area = {a_2000:.4f} * Edge^{b_2000:.5f}", transform=ax2.transAxes, color='black', fontsize=8)
#ax2.text(0.02, 0.83, f"2020: Area = {a_2020:.4f} * Edge^{b_2020:.5f}", transform=ax2.transAxes, color='dodgerblue', fontsize=8)

R2_2000 = rsquared(df['forest edge 2000'], df['Total Forest Area 2000'])
R2_2020 = rsquared(df['forest edge 2020'], df['Total Forest Area 2020'])
ax2.text(0.02, 0.73, f"$R^2$ = {round(R2_2000,3)}", transform=ax2.transAxes, color='black', fontsize=8)
#ax2.text(0.02, 0.63, f"$R^2$ = {round(R2_2020,3)}", transform=ax2.transAxes, color='dodgerblue', fontsize=8)

_, p_value_2000 = stats.ttest_ind(df['forest edge 2000'], df['Total Forest Area 2000'])
_, p_value_2020 = stats.ttest_ind(df['forest edge 2020'], df['Total Forest Area 2020'])
ax2.text(0.02, 0.53, f"$p$ = {round(p_value_2000,5)}", transform=ax2.transAxes, color='black', fontsize=8)
# ax2.text(0.02, 0.43, f"$p$ = {round(p_value_2020,5)}", transform=ax2.transAxes, color='dodgerblue', fontsize=8)

# Labeling and titling
ax2.set_xlabel('Edge ($km$)',fontsize=9,labelpad = 1)
ax2.set_ylabel('$Area (km^2)$',fontsize=9,labelpad = 1)
# ax2.set_title('Scatter Plot: Area vs Edge for 2000 and 2020 with Fitted Lines')
ax2.legend(title='Year',title_fontsize='small', scatterpoints=1, loc = 'lower right',fontsize=9,facecolor= 'none',edgecolor = 'none')
ax2.tick_params(axis='both',which='major',labelsize=9,direction='out',length=3,width=0.5,pad=1.3,labelleft = True, labelbottom = True,
                bottom=True,left=True,top=False,right=False)

# Setting the axes to logarithmic scale
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.savefig('../2_figures/Figure 3_Forest edge dynamics_statistics_V2.png', dpi=600, bbox_inches='tight')