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


def evaluate_next_day_probability(model, train_times, test_times, horizon_hours=24):
    """
    Проверяем, насколько хорошо модель предсказывает вероятность покупки
    в окне horizon_hours после последнего события трейна.
    """
    t_last = train_times[-1]
    history = list(train_times)
    
    has_event = False
    if len(test_times) > 0:
        if (test_times[0] - t_last).total_seconds() / 3600 <= horizon_hours:
            has_event = True

    lam_now = model.get_intensity(t_last, history)
    prob_est = 1 - np.exp(-lam_now * horizon_hours)
    
    y_true = 1 if has_event else 0
    epsilon = 1e-15
    prob_est = np.clip(prob_est, epsilon, 1 - epsilon)
    log_loss = -(y_true * np.log(prob_est) + (1 - y_true) * np.log(1 - prob_est))
    
    return log_loss, prob_est, has_event