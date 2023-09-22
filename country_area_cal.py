import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask

# 1. ????
forest_2000_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Area'
forest_2020_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2020Area'

forest_2000_files = os.listdir(forest_2000_dir)
forest_2020_files = os.listdir(forest_2020_dir)

common_files = set(forest_2000_files).intersection(set(forest_2020_files))

countries = gpd.read_file('/mnt/cephfs/scratch/groups/chen_group/hangkai/country_shp/ne_10m_admin_0_countries.shp')

# 2. ??????
result = []
for idx, row in countries.iterrows():
    if idx <200:
        continue
    if idx >301:
        break

    country = row
    forest_area2000 = 0.0
    forest_area2020 = 0.0
    unchanged_forest_area = 0.0
    increased_forest_area = 0.0
    decreased_forest_area = 0.0

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

            fa2000 = (data_2000 > 0) & mask_2000 & mask_2020
            fa2020 = (data_2020 > 0) & mask_2000 & mask_2020
            unchanged = (data_2000 == data_2020) & mask_2000 & mask_2020
            increased = (data_2020 > data_2000) & mask_2000 & mask_2020
            decreased = (data_2000 > data_2020) & mask_2000 & mask_2020

            forest_area2000 += data_2000[fa2000].sum()/(1000*1000*10)
            forest_area2020 += data_2000[fa2020].sum()/(1000*1000*10)
            unchanged_forest_area += data_2000[unchanged].sum()/(1000*1000*10)
            increased_forest_area += data_2020[increased].sum()/(1000*1000*10)
            decreased_forest_area += data_2000[decreased].sum()/(1000*1000*10)

    only_2000_files = set(forest_2000_files) - set(forest_2020_files)
    for file in only_2000_files:
        with rasterio.open(os.path.join(forest_2000_dir, file)) as src_2000:
            # Read the data
            data_2000 = src_2000.read(1)

            mask_2000, _ = rasterio.mask.mask(src_2000, [country.geometry], crop=True)
            valid_pixels = (data_2000 > 0) & (data_2000 < 10000) & mask_2000
            decreased_forest_area += data_2000[valid_pixels].sum()/(1000*1000*10)

    only_2020_files = set(forest_2020_files) - set(forest_2000_files)
    for file in only_2020_files:
        with rasterio.open(os.path.join(forest_2020_dir, file)) as src_2020:
            # Read the data
            data_2020 = src_2020.read(1)

            mask_2020, _ = rasterio.mask.mask(src_2020, [country.geometry], crop=True)
            valid_pixels = (data_2020 > 0) & (data_2020 < 10000) & mask_2020
            increased_forest_area += data_2020[valid_pixels].sum()/(1000*1000*10)

    result.append({
        'Country': country['SOVEREIGNT'],
        'Unchanged Forest Area': unchanged_forest_area,
        'Increased Forest Area': increased_forest_area,
        'Decreased Forest Area': decreased_forest_area,
        'Total Forest Area 2000': forest_area2000,
        'Total Forest Area 2020': forest_area2020
    })

# 3. ????
output_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/country_forest_area1to5'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(result)
df.to_csv(os.path.join(output_dir, 'forest_area_changes201to301.csv'), index=False)
