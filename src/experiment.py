import pandas as pd
import numpy as np
from tqdm import tqdm
from .models import GlobalSeasonalProfile, GlobalSeasonalPoisson, HawkesExp
from .utils import temporal_train_test_split
from .metrics import calc_test_ll, evaluate_forecast

def run_validation_loop(sequences, min_events=30, train_ratio=0.8):
    # 1. Filter
    active_users = sequences[sequences.apply(len) >= min_events].index
    print(f"Running experiment on {len(active_users)} users (min_events={min_events})")
    
    # 2. Global Profile
    print("Fitting Global Profile...")
    all_train_times = []
    for uid in active_users:
        tr, _ = temporal_train_test_split(sequences[uid], train_ratio)
        all_train_times.append(tr)
        
    global_prof = GlobalSeasonalProfile().fit(all_train_times)
    
    # 3. Validation Loop
    results = []
    for uid in tqdm(active_users):
        times = sequences[uid]
        train, test = temporal_train_test_split(times, train_ratio)
        
        if len(test) < 1: continue
            
        try:
            # Models
            poisson = GlobalSeasonalPoisson(global_prof, train)
            hawkes = HawkesExp(baseline_model=poisson).fit(train)
            
            # Extract Params (Alpha, Beta)
            # params = [alpha, beta] т.к. baseline есть
            alpha_val, beta_val = hawkes.params
            
            # Metrics
            ll_p = calc_test_ll(poisson, train, test)
            ll_h = calc_test_ll(hawkes, train, test)
            
            rmse_p, _, _ = evaluate_forecast(poisson, train, test)
            rmse_h, _, _ = evaluate_forecast(hawkes, train, test)
            
            results.append({
                'user_id': uid,
                'events_total': len(times),
                'alpha': alpha_val, 
                'beta': beta_val, 
                'branching_ratio': alpha_val / beta_val, 
                'll_poisson': ll_p,
                'll_hawkes': ll_h,
                'rmse_poisson': rmse_p,
                'rmse_hawkes': rmse_h,
                'imp_rmse': (1 - rmse_h/rmse_p) * 100,
                'hawkes_win': ll_h > ll_p
            })
            
        except Exception as e:
            continue
            
    return pd.DataFrame(results)