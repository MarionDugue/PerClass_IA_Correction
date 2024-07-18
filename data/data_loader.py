import rasterio
import numpy as np
from scipy.ndimage import generic_filter


mask_path = 'C:\\Users\MarionD\Desktop\AGP\Thesis\EPSG_Mask.tif'
mask_path_EW = ''

def std_3x3(array):
    return np.std(array)

def load_data(directory, zones_titles, std):
    df = pd.DataFrame()
    if std: 
        bands_by_zone = {zone: {'HH': [], 'HV': [], 'ia': [], 'std_HH':[], 'std_HV':[]} for zone in zones_titles}
    else:
        bands_by_zone = {zone: {'HH': [], 'HV': [], 'ia': []} for zone in zones_titles}

    S1_products = [subdir for subdir in os.listdir(directory) if subdir.startswith('S1')]

    for product in S1_products:
        try:
            zone_files = {zone: glob.glob(f'{directory}/{product}/*{zone}.tif')[0] for zone in zones_titles}
            zone_bands = {zone: extract_bands(file_path) for zone, file_path in zone_files.items()}

            zone_stats = []
            for zone, bands in zone_bands.items():
                hh_band, hv_band, ia_band = bands
                bands_by_zone[zone]['HH'].append(hh_band)
                bands_by_zone[zone]['HV'].append(hv_band)
                bands_by_zone[zone]['ia'].append(ia_band)
                if std:
                    hh_std = generic_filter(hh_band, std_3x3, size=8, mode='constant', cval=0.0)
                    hv_std = generic_filter(hv_band, std_3x3, size=8, mode='constant', cval=0.0)
                    bands_by_zone[zone]['std_HH'].append(hh_std)
                    bands_by_zone[zone]['std_HV'].append(hv_std)

                timestamp = product.split('_')[5]
                parsed_date = datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
                formatted_date = parsed_date.strftime("%d-%m-%Y")
                zone_stats.append({
                    'Product': product, 
                    'Date': formatted_date,
                    'Zone': zone,
                    'HH_mean': np.mean(hh_band),
                    'HH_std': np.std(hh_band),
                    'HV_mean': np.mean(hv_band),
                    'HV_std': np.std(hv_band),
                    'IA_mean': np.round(np.mean(ia_band), 2),
                    'IA_std': np.std(ia_band),
                    'std_HH': np.mean(hh_std),
                    'std_HV': np.mean(hv_std)
                })

            df = pd.concat([df, pd.DataFrame(zone_stats)], ignore_index=True)
        except Exception as e:
            print(f'Error processing {product}: {e}')
            df = df[df['Product'] != product]
            continue

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values(by='Date')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.drop_duplicates(subset=['Date', 'Zone'])
    df = df[df['IA_mean'] != 0]
    return df, bands_by_zone, zone_bands

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