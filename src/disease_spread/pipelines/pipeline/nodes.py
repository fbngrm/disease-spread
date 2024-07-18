import pandas as pd
 
def preprocess_data(features, labels):
    # select features we want
    selected_features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    features = features[selected_features]
    
    # fill missing values
    features.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    features = features.join(labels)

    # separate san juan and iquitos
    city_sj = features.query("city=='sj'")
    city_iq = features.query("city=='iq'")
    
    return city_sj, city_iq