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

def split_up(city_sj, city_iq):
    city_sj_train = city_sj.head(800)
    city_sj_test = city_sj.tail(city_sj.shape[0] - 800)

    city_iq_train = city_iq.head(400)
    city_iq_test = city_iq.tail(city_iq.shape[0] - 400)

    return city_sj_train, city_sj_test, city_iq_train, city_iq_test

