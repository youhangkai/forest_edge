import os
import numpy as np
import rasterio
from geopy.distance import geodesic
import gc

# Function to calculate lookup table for pixel widths
def compute_width_lookup(num_rows, transform):
    latitudes = [transform[5] + i * transform[4] for i in range(num_rows)]
    return [compute_width(lat, abs(transform[0])) for lat in latitudes]

# Function to calculate pixel width
def compute_width(lat, cellsize):
    point1 = (lat, 0)
    point2 = (lat, cellsize)
    return geodesic(point1, point2).meters

# Function to calculate pixel height
def compute_height(cellsize):
    point1 = (0, 0)
    point2 = (0, cellsize)
    return geodesic(point1, point2).meters

# Function to compute forest edge length
def compute_edge_length(i, j, forest, width_lookup, height):
    edge_length = 0.0  # Initialize edge length

    # Check left pixel
    if j > 0 and not forest[i, j-1]:
        edge_length += height  # Add height to edge length

    # Check right pixel
    if j < forest.shape[1] - 1 and not forest[i, j+1]:
        edge_length += height  # Add height to edge length

    # Check upper pixel
    if i > 0 and not forest[i-1, j]:
        edge_length += width_lookup[i-1]  # Add width of upper pixel to edge length

    # Check lower pixel
    if i < forest.shape[0] - 1 and not forest[i+1, j]:
        edge_length += width_lookup[i]  # Add width of current pixel to edge length

    return edge_length

# Function to process a single TIFF file
def process_tiff(filename):
    # Full paths to input and output files
    input_file = os.path.join(input_folder, filename)
    output_file_area = os.path.join(output_folder_area, filename)
    output_file_edge = os.path.join(output_folder_edge, filename)
    print(f"Processing file: {filename}")

    # Open tif file
    with rasterio.open(input_file) as ds:
        forest = ds.read(1).astype(bool)
        transform = ds.transform
        cellsize = abs(transform[0])
        print(f"Cellsize: {cellsize}")

        # Compute lookup table for pixel widths and pixel height
        width_lookup = compute_width_lookup(forest.shape[0], transform)
        height = compute_height(cellsize)
        print("Computed lookup table for pixel widths and pixel height")

        # Create new tifs
        area_tif = np.zeros_like(forest, dtype=np.float32)
        edge_tif = np.zeros_like(forest, dtype=np.float32)
        print("Created new tifs")

        # Loop through each pixel in the forest tif
        for i in range(forest.shape[0]):
            for j in range(forest.shape[1]):
                # Check if the pixel is a forest pixel
                if forest[i, j]:
                    # Calculate the area of the pixel and assign it to area_tif
                    area_tif[i, j] = width_lookup[i] * height

                    # Calculate the edge length of the pixel and assign it to edge_tif
                    edge_tif[i, j] = compute_edge_length(i, j, forest, width_lookup, height)

        # Convert to unsigned integers with rounding
        scale_factor = 10
        area_tif = np.around(area_tif * scale_factor).astype(np.uint16)
        edge_tif = np.around(edge_tif * scale_factor).astype(np.uint16)

        # Write new tif to file
        with rasterio.open(output_file_area, 'w', driver='GTiff', height=area_tif.shape[0],
                           width=area_tif.shape[1], count=1, dtype=area_tif.dtype,
                           crs=ds.crs, transform=transform) as dst:
            dst.write(area_tif, 1)

        with rasterio.open(output_file_edge, 'w', driver='GTiff', height=edge_tif.shape[0],
                           width=edge_tif.shape[1], count=1, dtype=edge_tif.dtype,
                           crs=ds.crs, transform=transform) as dst:
            dst.write(edge_tif, 1)

    print(f"Finished processing file: {filename}")
    del forest
    del area_tif
    del edge_tif
    gc.collect()  # Force garbage collection

# Paths to input and output folders
input_folder = "/mnt/cephfs/scratch/groups/chen_group/hangkai/2000extent"
output_folder_area = "/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Area"
output_folder_edge = "/mnt/cephfs/scratch/groups/chen_group/hangkai/2000Edge"

# Create output folders if they don't exist
if not os.path.exists(output_folder_area):
    os.makedirs(output_folder_area)
if not os.path.exists(output_folder_edge):
    os.makedirs(output_folder_edge)

# Get list of tif files
tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
print("Start to loop!")
# Loop over each file in the list
for filename in tif_files:
    process_tiff(filename)