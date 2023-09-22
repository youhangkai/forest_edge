import os
import numpy as np
from rasterio.enums import Resampling
import rasterio
import gc
from skimage.measure import block_reduce

input_folder = "/mnt/cephfs/scratch/groups/chen_group/hangkai/2020Area"
output_folder = "/mnt/cephfs/scratch/groups/chen_group/hangkai/01/2020Area_processed01"
target_resolution = 0.1

for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, filename)

        with rasterio.open(input_filepath) as src:
            width = src.width
            height = src.height
            resolution = src.transform[0]
            original_image = src.read(1)
            
            new_height = int(height * (resolution / target_resolution))
            new_width = int(width * (resolution / target_resolution))
            
            block_size_x = width // new_width
            block_size_y = height // new_height

            new_image = block_reduce(original_image, block_size=(block_size_y, block_size_x), func=np.nansum)
            new_image = new_image*0.1

            # Prepare the new metadata and transform
            out_meta = src.meta
            new_transform = rasterio.transform.from_origin(
                west=src.transform[2],
                north=src.transform[5],
                xsize=target_resolution,
                ysize=target_resolution
            )
            out_meta.update({
                "driver": "GTiff",
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "dtype": rasterio.float32
            })

            with rasterio.open(output_filepath, "w", **out_meta) as dest:
                dest.write(new_image, 1)
            
            del new_image
            gc.collect()