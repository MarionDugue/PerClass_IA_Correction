import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime
from .data_loader import extract_bands
from scipy.ndimage import generic_filter


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
                if not (hh_band is None or hv_band is None or ia_band is None):
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

def prepare_training_data(bands_by_zone, zones_titles, std=False):
    X = []
    y = []
    for idx, zone in enumerate(zones_titles):
        if std:
            hh_band = np.array(bands_by_zone[zone]['HH']).flatten()
            hv_band = np.array(bands_by_zone[zone]['HV']).flatten()
            hh_std = np.array(bands_by_zone[zone]['std_HH']).flatten()
            hv_std= np.array(bands_by_zone[zone]['std_HV']).flatten()
            zone_data = np.column_stack((hh_band, hv_band,hh_std, hv_std ))
        else:
            hh_band = np.array(bands_by_zone[zone]['HH']).flatten()
            hv_band = np.array(bands_by_zone[zone]['HV']).flatten()
            zone_data = np.column_stack((hh_band, hv_band))
        X.append(zone_data)
        y.append(np.full(zone_data.shape[0], idx))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y

def prep_X_IA(bands_by_zone, zone_titles, slopes_dict, IA_ref=30):    
    IA_X_list = []
    y = []
    for idx, zone in enumerate(zone_titles):
        hh_band = np.array(bands_by_zone[zone]['HH']).flatten()
        hv_band = np.array(bands_by_zone[zone]['HV']).flatten()
        ia_band = np.array(bands_by_zone[zone]['ia']).flatten()
        zone_data = np.column_stack((hh_band, hv_band))

        X_array = np.column_stack((hh_band, hv_band))
        slopes = slopes_dict[zone]
        y.append(np.full(zone_data.shape[0], idx+1))

        projected_array = np.zeros(X_array.shape)
        for dimension in range(X_array.shape[1]):
            projected_array[:, dimension] = X_array[:, dimension] + (slopes[dimension] * (ia_band - IA_ref))
        IA_X = np.hstack((projected_array, np.expand_dims(ia_band, 1)))
        IA_X_list.append(IA_X)

    y = np.concatenate(y, axis=0)
    X_IA = np.vstack(IA_X_list)
    
    return X_IA, y
