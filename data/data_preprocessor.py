import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime
from .data_loader import extract_bands
from scipy.ndimage import generic_filter


def std_3x3(array):
    return np.std(array)

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
