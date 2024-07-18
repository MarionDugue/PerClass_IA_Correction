import rasterio
import numpy as np
from scipy.ndimage import generic_filter


mask_path = 'C:\\Users\MarionD\Desktop\AGP\Thesis\EPSG_Mask.tif'
mask_path_EW = ''

def std_3x3(array):
    return np.std(array)

def extract_bands_glacier(file_path, std):
    with rasterio.open(file_path) as src, rasterio.open(mask_path) as mask_src:
        hh_band = src.read(1)
        hv_band = src.read(2)
        ia_band = src.read(3)

        hh_std = generic_filter(hh_band, std_3x3, size=8, mode='constant', cval=0.0)
        hv_std = generic_filter(hv_band, std_3x3, size=8, mode='constant', cval=0.0)

        mask = mask_src.read(1)
        valid_pixels = mask == 1
    
    if std==False:
        data = np.stack((hh_band, hv_band), axis=-1)
    else:
        data = np.stack((hh_band, hv_band, hh_std, hv_std), axis=-1)
    masked_data = data[valid_pixels]
    pixels = masked_data.reshape(-1, 2)
    masked_data_IA = ia_band[valid_pixels]
    pixels_IA = masked_data_IA.reshape(-1)
    
    return pixels, pixels_IA, valid_pixels, hh_band.shape

def extract_bands(file_path):
    with rasterio.open(file_path) as src:
        hh_band = src.read(1)
        hv_band = src.read(2)
        ia_band = src.read(3)
    return hh_band, hv_band, ia_band