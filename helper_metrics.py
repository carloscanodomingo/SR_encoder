import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
def selected_freq(data):
    low_limit = 7.5
    up_limit = 8.2
    est = data["freq_est"][:,0];
    lorentz = data["freq_values"][:,0] * 50
    fit = data["freq_fit_first"];

    not_selected_fit =  1 - (sum(np.logical_and(fit < 8.2, fit > 7.5)) / len(fit ))
    not_selected_est =  1 - (sum(np.logical_and(est < 8.2, est > 7.5)) / len(est ))
    not_selected_lorentz =  1 - (sum(np.logical_and(lorentz < 8.2, lorentz > 7.5)) / len(lorentz ))
    metrics = {"perct_lorentz": not_selected_lorentz, "perct_estimated": not_selected_est, "perct_fit" : not_selected_fit}
    return metrics

def corr_noise(data):
    est = data["freq_est"][:,0];
    lorentz = data["freq_values"][:,0] * 50
    noise = data["noise"] 
    error_freq = np.absolute(lorentz - est)
    sr_correlation_freq = []
    for index_sr_mode in range (0, 6):
        est = data["freq_est"][:,index_sr_mode];
        lorentz = data["freq_values"][:,index_sr_mode] * 50
        error_freq = np.absolute(lorentz - est)
        correlation_freq = np.corrcoef(noise, error_freq)
        sr_correlation_freq.append(correlation_freq[0,1])
    
    #Focus on First Mode
    est1 = data["freq_est"][:,0];
    lorentz1 = data["freq_values"][:,0] * 50
    # Compute the error freq
    error_freq1 = np.absolute(lorentz1 - est1)
    # Get the linear regression model to get the slope
    model_freq1 = LinearRegression().fit( noise.reshape(-1,1), error_freq1.reshape(-1,1))
    # Compute correlation and take the cross correlation between the two var
    correlation_freq1 = np.corrcoef(noise, error_freq1)[0,1]
    
    sr_correlation_freq = np.array(sr_correlation_freq)
    sr_mean_freq = np.mean(sr_correlation_freq)        
    
    raw = np.reshape(data["raw"],(-1, 256))
    lorentz = np.reshape(data["decoded"],(-1, 256))
    rmse_value = np.array( mean_squared_error(raw.transpose(), lorentz.transpose(), multioutput='raw_values'))  
    model_rmse = LinearRegression().fit(noise.reshape(-1,1), rmse_value.reshape(-1,1))
    correlation_rmse = np.corrcoef(noise, rmse_value)[0,1]

    
    metrics = {"corr_rmse": correlation_rmse, 
               "slope rmse": model_rmse.coef_[0][0],
              "corr_freq": sr_mean_freq,
              "slope freq1":  model_freq1.coef_[0][0],
              "corr_freq1": correlation_freq1}
    return(metrics)