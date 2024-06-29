import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask


# 1. Load data
forest_2000_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Edge'
forest_2020_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2020Edge'

forest_2000_files = os.listdir(forest_2000_dir)
forest_2020_files = os.listdir(forest_2020_dir)

common_files = set(forest_2000_files).intersection(set(forest_2020_files))

countries = gpd.read_file('/mnt/cephfs/scratch/groups/chen_group/hangkai/country_shp/ne_10m_admin_0_countries.shp')

# 2. Calculation of forest area
result = []
for idx, row in countries.iterrows():
    if idx <200:
        continue
    if idx >301:
        break


    country = row
    forest_edge2000 = 0.0
    forest_edge2020 = 0.0
    unchanged_forest_edge = 0.0
    pure_unchanged_forest_edge = 0.0
    increased_forest_edge = 0.0
    decreased_forest_edge = 0.0
    pure_increased_forest_edge = 0.0
    pure_decreased_forest_edge = 0.0

    for file in common_files:
        with rasterio.open(os.path.join(forest_2000_dir, file)) as src_2000, rasterio.open(os.path.join(forest_2020_dir, file)) as src_2020:
            # Read the data
            data_2000 = src_2000.read(1)
            data_2020 = src_2020.read(1)

            mask_2000 = geometry_mask([geom for geom in [country.geometry]],
                      transform=src_2000.transform,
                      out_shape=src_2000.shape,
                      invert=True)
            mask_2020 = geometry_mask([geom for geom in [country.geometry]],
                      transform=src_2020.transform,
                      out_shape=src_2020.shape,
                      invert=True)

            forest_edge_2000 = (data_2000 > 0) & mask_2000 & mask_2020
            forest_edge_2020 = (data_2020 > 0) & mask_2000 & mask_2020
            unchanged = (data_2000 == data_2020) & mask_2000 & mask_2020
            increased = (data_2020 > data_2000) & mask_2000 & mask_2020
            decreased = (data_2000 > data_2020) & mask_2000 & mask_2020

            unchanged_forest_edge += data_2000[unchanged].sum()/(1000*10)
            forest_edge2000 += data_2000[forest_edge_2000].sum()/(1000*10)
            forest_edge2020 += data_2020[forest_edge_2020].sum()/(1000*10)
            pure_unchanged_forest_edge += (unchanged_forest_edge + data_2000[increased].sum()/(1000*10) + data_2020[decreased].sum()/(1000*10))
            pure_increased_forest_edge += (data_2020[increased].sum()/(1000*10)-data_2000[increased].sum()/(1000*10))
            pure_decreased_forest_edge += (data_2000[decreased].sum()/(1000*10)-data_2020[decreased].sum()/(1000*10))

    only_2000_files = set(forest_2000_files) - set(forest_2020_files)
    for file in only_2000_files:
        with rasterio.open(os.path.join(forest_2000_dir, file)) as src_2000:
            # Read the data
            data_2000 = src_2000.read(1)

            mask_2000, _ = rasterio.mask.mask(src_2000, [country.geometry], crop=True)
            valid_pixels = (data_2000 > 0) & (data_2000 < 10000) & mask_2000
            decreased_forest_edge += data_2000[valid_pixels].sum()/(1000*10)

    only_2020_files = set(forest_2020_files) - set(forest_2000_files)
    for file in only_2020_files:
        with rasterio.open(os.path.join(forest_2020_dir, file)) as src_2020:
            # Read the data
            data_2020 = src_2020.read(1)

            mask_2020, _ = rasterio.mask.mask(src_2020, [country.geometry], crop=True)
            valid_pixels = (data_2020 > 0) & (data_2020 < 10000) & mask_2020
            increased_forest_edge += data_2020[valid_pixels].sum()/(1000*10)

    result.append({
        'Country': country['SOVEREIGNT'],
        'Edge of Unchanged Forests': unchanged_forest_edge,
        'Pure Unchanged Forest Edge': pure_unchanged_forest_edge,
        'Pure Increased Forest Edge': pure_increased_forest_edge,
        'Pure Decreased Forest Edge': pure_decreased_forest_edge,
        'Total Forest Edge 2000': forest_edge2000,
        'Total Forest Edge 2020': forest_edge2020
    })

# 3. output result
output_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/country_forest_area'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(result)
df.to_csv(os.path.join(output_dir, 'forest_edge_changes201to301.csv'), index=False)