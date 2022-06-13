def get_selected_features(X):
    # based on analysis done in DecisionTrees.py
    return X[[ 'rapJitter', 'meanAutoCorrHarmonicity'
        , 'meanNoiseToHarmHarmonicity', 'GQ_prc5_95', 'mean_1st_delta'
        , 'mean_7th_delta_delta', 'mean_9th_delta_delta', 'std_1st_delta'
        , 'std_delta_delta_log_energy', 'std_12th_delta_delta'
        , 'det_entropy_log_10_coef', 'det_TKEO_mean_2_coef', 'det_TKEO_std_9_coef'
        , 'app_entropy_shannon_5_coef', 'app_entropy_log_3_coef'
        , 'app_det_TKEO_mean_6_coef', 'app_det_TKEO_mean_7_coef'
        , 'app_TKEO_std_5_coef', 'app_TKEO_std_6_coef', 'app_TKEO_std_7_coef'
        , 'app_TKEO_std_9_coef', 'det_LT_entropy_shannon_1_coef'
        , 'det_LT_entropy_shannon_10_coef', 'det_LT_entropy_log_5_coef'
        , 'app_LT_entropy_shannon_3_coef', 'app_LT_entropy_log_3_coef'
        , 'app_LT_entropy_log_7_coef', 'app_LT_TKEO_mean_8_coef'
        , 'app_LT_TKEO_std_8_coef', 'PPE', 'RPDE', 'DFA', 'numPulses'
        , 'meanHarmToNoiseHarmonicity', 'locShimmer']]