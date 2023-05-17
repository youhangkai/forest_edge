import os
import cv2
import numpy as np
from osgeo import gdal, ogr, osr
from multiprocessing import Pool
import math
import pandas as pd

# 配置路径
base_path = "/mnt/cephfs/scratch/groups/chen_group/hangkai"
years = ["2000extent", "2020extent"]
output_path_shapefile = os.path.join(base_path, "forest_edge_shapefile")
output_path_data = os.path.join(base_path, "forest_edge_data")

os.makedirs(output_path_shapefile, exist_ok=True)
os.makedirs(output_path_data, exist_ok=True)

# 从raster获取像素尺寸
def get_pixel_size(raster_file):
    raster = gdal.Open(raster_file)
    geotransform = raster.GetGeoTransform()
    return geotransform[1], geotransform[5]  # x pixel size, y pixel size

# 从raster获取projection
def get_projection(raster_file):
    raster = gdal.Open(raster_file)
    return raster.GetProjection()

# 将经纬度差转换为米
def latlon_to_meters(lat_diff, lon_diff, lat):
    # 平均地球半径，单位：米
    R = 6371e3  

    # 把经纬度转换为弧度
    lat1 = math.radians(lat)
    lat2 = math.radians(lat + lat_diff)
    lon_diff = math.radians(lon_diff)

    # 应用haversine公式
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(lon_diff/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# 获取像素尺寸（米）
def get_pixel_size_meters(raster_file):
    x_pixel_size, y_pixel_size = get_pixel_size(raster_file)
    raster = gdal.Open(raster_file)
    lat = raster.GetGeoTransform()[3] # 获取纬度
    lat = lat + 5
    x_pixel_size_meters = latlon_to_meters(0, x_pixel_size, lat)
    y_pixel_size_meters = latlon_to_meters(y_pixel_size, 0, lat)
    print(x_pixel_size_meters,y_pixel_size_meters)
    return x_pixel_size_meters, y_pixel_size_meters

# 创建Shapefile来存储边缘
def create_shapefile(path, projection):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = data_source.CreateLayer("forest_edge", srs, ogr.wkbLineString)
    return data_source, layer

# 处理每个tif文件
def process_tif(args):
    year, tif_file = args
    print(f"Processing {tif_file}...")

    # 加载数据
    raster = gdal.Open(tif_file)
    data = raster.ReadAsArray()

    # 计算像素尺寸
    x_pixel_size, y_pixel_size = get_pixel_size_meters(tif_file)
    pixel_area = abs(y_pixel_size) * abs(y_pixel_size)

    # 找到边缘并计算面积
    contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_forest_area = np.count_nonzero(data) * pixel_area
    total_edge_length = sum(cv2.arcLength(cnt, True) for cnt in contours) * x_pixel_size

    # 创建输出文件
    output_folder_shapefile = os.path.join(output_path_shapefile, f"{year}_results")
    os.makedirs(output_folder_shapefile, exist_ok=True)
    shapefile_path = os.path.join(output_folder_shapefile, os.path.basename(tif_file).replace(".tif", ".shp"))
    data_source, layer = create_shapefile(shapefile_path, get_projection(tif_file))

    # 保存边缘到shapefile
    for cnt in contours:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in cnt:
            ring.AddPoint(float(point[0][0]), float(point[0][1]))
        ring.CloseRings()

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        feature = None
    
    data_source = None
    
    return tif_file, shapefile_path, total_forest_area, total_edge_length

# 并行处理所有tif文件
with Pool() as pool:
    for year in years:
        tif_folder = os.path.join(base_path, year)
        tif_files = sorted([os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith(".tif")])
        results = pool.map(process_tif, [(year, tif_file) for tif_file in tif_files])

        # 将结果保存到CSV文件
        df = pd.DataFrame(results, columns=["tif_file", "shapefile_path", "total_forest_area", "total_edge_length"])
        df["year"] = year
        csv_output_path = os.path.join(output_path_data, f"{year}_results.csv")
        df.to_csv(csv_output_path, index=False)