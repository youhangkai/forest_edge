import os
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask


forest_2000_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Edge'
forest_area_2000_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Area'
forest_2020_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2020Edge'
forest_area_2020_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/2020Area'

forest_2000_files = os.listdir(forest_2000_dir)
forest_area_2000_files = os.listdir(forest_area_2000_dir)
forest_2020_files = os.listdir(forest_2020_dir)
forest_area_2020_files = os.listdir(forest_area_2020_dir)

common_files = set(forest_2000_files).intersection(set(forest_2020_files))

countries = gpd.read_file('/mnt/cephfs/scratch/groups/chen_group/hangkai/country_shp/ne_10m_admin_0_countries.shp')

front_num = 200
back_num = 301
result = []
for idx, row in countries.iterrows():
    if idx < front_num:
        continue
    if idx > back_num:
        break


    country = row
    total_forest_edge2000 = 0.0
    total_forest_edge2020 = 0.0
    increased_forest_edge_due_to_area_increase_2020 = 0.0
    decreased_forest_edge_due_to_area_increase_2020 = 0.0
    increased_forest_edge_due_to_area_decrease_2020 = 0.0
    decreased_forest_edge_due_to_area_decrease_2020 = 0.0


    unchanged_forest_edge = 0.0
    edge_decrease_due_to_area_decrease_2000 = 0.0
    edge_increase_due_to_area_increase = 0.0
    edge_increase_due_to_area_decrease = 0.0
    edge_decrease_due_to_area_increase = 0.0


    for file in common_files:
        with rasterio.open(os.path.join(forest_2000_dir, file)) as src_2000, rasterio.open(os.path.join(forest_area_2000_dir, file)) as src_area_2000, rasterio.open(os.path.join(forest_area_2020_dir, file)) as src_area_2020, rasterio.open(os.path.join(forest_2020_dir, file)) as src_2020:
            # Read the data
            data_2000 = src_2000.read(1)
            area_2000 = src_area_2000.read(1)
            area_2020 = src_area_2020.read(1)
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
            increase_due_to_area_increase = (data_2020 > data_2000) & (area_2000 <= 0) & (area_2020 > 0) & mask_2000 & mask_2020
            increase_due_to_area_decrease = (data_2020 > data_2000) & (area_2000 > 0) & (area_2020 > 0) & mask_2000 & mask_2020
            decrease_due_to_area_increase = (data_2020 < data_2000) & (area_2000 > 0) & (area_2020 > 0) & mask_2000 & mask_2020
            decrease_due_to_area_decrease_2000 = (data_2020 < data_2000) & (area_2000 <= 0) & mask_2000 & mask_2020


            total_forest_edge2000 += data_2000[forest_edge_2000].sum()/(1000*10)
            total_forest_edge2020 += data_2020[forest_edge_2020].sum()/(1000*10)

            unchanged_forest_edge += (data_2000[unchanged].sum()/(1000*10)) + data_2000[increase_due_to_area_decrease].sum()/(1000*10) + abs(data_2000[decrease_due_to_area_increase].sum()/(1000*10) - data_2020[decrease_due_to_area_increase].sum()/(1000*10))

            edge_increase_due_to_area_increase += abs(data_2020[increase_due_to_area_increase].sum()/(1000*10) - data_2000[increase_due_to_area_increase].sum()/(1000*10))
            edge_increase_due_to_area_decrease += abs(data_2020[increase_due_to_area_decrease].sum()/(1000*10) - data_2000[increase_due_to_area_decrease].sum()/(1000*10))
            edge_decrease_due_to_area_increase += abs(data_2020[decrease_due_to_area_increase].sum()/(1000*10) - data_2000[decrease_due_to_area_increase].sum()/(1000*10))
            edge_decrease_due_to_area_decrease_2000 += (data_2000[decrease_due_to_area_decrease_2000].sum()/(1000*10))


    result.append({
        'Country': country['SOVEREIGNT'],
        'Edge of Unchanged Forests': unchanged_forest_edge,
        'Edge increase due to area increase': edge_increase_due_to_area_increase,
        'Edge increase due to area decrease': edge_increase_due_to_area_decrease,
        'Edge decrease due to area increase': edge_decrease_due_to_area_increase,
        'Edge decrease due to area decrease for 2000': edge_decrease_due_to_area_decrease_2000,
        'Total Forest Edge 2000': total_forest_edge2000,
        'Total Forest Edge 2020': total_forest_edge2020
    })

# 3. Output
output_dir = '/mnt/cephfs/scratch/groups/chen_group/hangkai/country_forest_area'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(result)
file_name = f"forest_edge_changes_with_area_{front_num}_to_{back_num}.csv"
df.to_csv(os.path.join(output_dir, file_name), index=False)




