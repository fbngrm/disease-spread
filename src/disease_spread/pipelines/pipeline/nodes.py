import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def lag(x, n):
    if n == 0:
        return x
    if isinstance(x, pd.Series):
        return x.shift(n)
    else:
        x = pd.Series(x)
        return x.shift(n)
 
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
    city_sj, city_iq  = split(features)
    city_sj['reanalysis_specific_humidity_g_per_kg_lag_t1'] = \
        lag(city_sj['reanalysis_specific_humidity_g_per_kg'], 1)
    city_sj['reanalysis_specific_humidity_g_per_kg_lag_t2'] = \
            lag(city_sj['reanalysis_specific_humidity_g_per_kg'], 2)
    city_iq['reanalysis_specific_humidity_g_per_kg_lag_t1'] = \
        lag(city_iq['reanalysis_specific_humidity_g_per_kg'], 1)
    city_iq['reanalysis_specific_humidity_g_per_kg_lag_t2'] =  \
        lag(city_iq['reanalysis_specific_humidity_g_per_kg'], 2)

    return city_sj, city_iq

def split_up(city_sj, city_iq):
    city_sj_train = city_sj.head(800)
    city_sj_test = city_sj.tail(city_sj.shape[0] - 800)

    city_iq_train = city_iq.head(400)
    city_iq_test = city_iq.tail(city_iq.shape[0] - 400)

    return city_sj_train, city_sj_test, city_iq_train, city_iq_test

def split(features):
    return features.query("city=='sj'"), features.query("city=='iq'")

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c + " \
                    "reanalysis_specific_humidity_g_per_kg_lag_t1 + " \
                    "reanalysis_specific_humidity_g_per_kg_lag_t2"
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

def best_model(city_sj_train, city_sj_test, city_iq_train, city_iq_test):
    sj_best_model = get_best_model(city_sj_train, city_sj_test)
    iq_best_model = get_best_model(city_iq_train, city_iq_test)
    return sj_best_model, iq_best_model

def predict_model(sj_best_model, iq_best_model, sj_predict, iq_predict):

    sj_predict['reanalysis_specific_humidity_g_per_kg_lag_t1'] = \
        lag(sj_predict['reanalysis_specific_humidity_g_per_kg'], 1)
    sj_predict['reanalysis_specific_humidity_g_per_kg_lag_t2'] = \
            lag(sj_predict['reanalysis_specific_humidity_g_per_kg'], 2)
    iq_predict['reanalysis_specific_humidity_g_per_kg_lag_t1'] = \
        lag(iq_predict['reanalysis_specific_humidity_g_per_kg'], 1)
    iq_predict['reanalysis_specific_humidity_g_per_kg_lag_t2'] =  \
        lag(iq_predict['reanalysis_specific_humidity_g_per_kg'], 2)

    # predict city sj
    sj_predicted = sj_best_model.predict(sj_predict)
    # convert non-finite values (NA or inf) to 0
    sj_predicted = sj_predicted.fillna(0) 

    # predict city iq
    iq_predicted = iq_best_model.predict(iq_predict)
    # convert non-finite values (NA or inf) to 0
    iq_predicted = iq_predicted.fillna(0)

    # combine the predicted data for the two cities
    predicted_data = np.concatenate([sj_predicted, iq_predicted])

    # convert all prediction to astype(int)
    predicted_data=predicted_data.astype(int)

    return predicted_data


def predict(predict_data,sj_best_model, iq_best_model, submission_data):
    sj_predict, iq_predict = split(predict_data)
    predicted_data = predict_model(sj_best_model, iq_best_model, sj_predict, iq_predict)

    #submission format
    submission_data.total_cases = predicted_data
    
    return submission_data

