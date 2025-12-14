import numpy as np
from sklearn.metrics import mean_squared_error

def calc_test_ll(model, train, test):
    full_history = list(train) + list(test)
    t_start = full_history[0]
    
    T_train = (train[-1] - t_start).total_seconds() / 3600
    T_full = (test[-1] - t_start).total_seconds() / 3600
    
    ll_test = model.nll(train, T_train) - model.nll(full_history, T_full)
    return ll_test / len(test)

def evaluate_forecast(model, train, test, window_hours=1.0):
    current_history = list(train)
    preds, targets, probs, y_window = [], [], [], []
    
    for event in test:
        last_event = current_history[-1]
        
        intensity = model.get_intensity(last_event, current_history)
        if intensity < 1e-3: intensity = 1e-3
            
        pred_delay = 1.0 / intensity
        actual_delay = (event - last_event).total_seconds() / 3600
        
        prob = 1 - np.exp(-intensity * window_hours)
        
        preds.append(pred_delay)
        targets.append(actual_delay)
        probs.append(prob)
        y_window.append(1 if actual_delay < window_hours else 0)
        
        current_history.append(event)
        
    rmse = np.sqrt(mean_squared_error(targets, preds))
    return rmse, y_window, probs