import pandas as pd
import numpy as np
from scipy import stats

def fit_line_to_zone_data_year(zone_df, year, data_type):
    data = zone_df[zone_df['Year'] == year]
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['IA_mean'], data[data_type])
    return slope, intercept, r_value**2, p_value, std_err

def calculate_slopes_and_intercepts(zone_bands, zone_titles, std):
    
    average_data_list = []
    average_data = pd.DataFrame()

    for zone, zone_data in zone_bands.items():
        slopes = []
        intercepts = []

        for year in zone_data['Year'].unique():
            year_data = zone_data[zone_data['Year'] == year]
            if not year_data.empty:
                slope_HH, intercept_HH, r_squared_HH, p_value_HH, std_err_HH = fit_line_to_zone_data_year(year_data, year, 'HH_mean')
                slope_HV, intercept_HV, r_squared_HV, p_value_HV, std_err_HV = fit_line_to_zone_data_year(year_data, year, 'HV_mean')
                if std:
                    slope_std_HH, intercept_std_HH, r_squared_std_HH, p_value_std_HH, std_err_std_HH = fit_line_to_zone_data_year(year_data, year, 'std_HH')
                    slope_std_HV, intercept_std_HV, r_squared_std_HV, p_value_std_HV, std_err_std_HV = fit_line_to_zone_data_year(year_data, year, 'std_HV')
                    average_data_list.append({
                'Zone': zone,
                'Year': year,
                'slope_HH': slope_HH,
                'intercept_HH': intercept_HH,
                'slope_HV': slope_HV,
                'intercept_HV': intercept_HV,
                'slope_std_HH': slope_std_HH,
                'intercept_std_HH': intercept_std_HH,
                'slope_std_HV': slope_std_HV,
                'intercept_std_HV': intercept_std_HV,
            })
                else:
                    average_data_list.append({
                        'Zone': zone,
                        'Year': year,
                        'slope_HH': slope_HH,
                        'intercept_HH': intercept_HH,
                        'slope_HV': slope_HV,
                        'intercept_HV': intercept_HV
                    })

    average_data = pd.concat([average_data, pd.DataFrame(average_data_list)], ignore_index=True)
    average_year_data = average_data.groupby('Zone').mean().reset_index()
    average_year_data['Year'] = 'Average'

    slopes_dict = {}
    
    for zone in zone_titles:
        slopes = average_year_data[average_year_data['Zone'] == zone][['slope_HH', 'slope_HV']].values.flatten().tolist()
        slopes_dict[zone] = slopes

    return average_year_data, slopes_dict


def slopes_and_projection():
    return