import pandas as pd
import numpy as np
from tqdm import tqdm
from .models import GlobalSeasonalProfile, GlobalSeasonalPoisson
from .utils import temporal_train_test_split
from .metrics import calc_test_ll, evaluate_forecast, evaluate_next_day_probability

def run_validation_loop(sequences, model_factory, label="Exp", min_events=30, train_ratio=0.8):
    active_users = sequences[sequences.apply(len) >= min_events].index
    print(f"Running '{label}': N={len(active_users)}")
    
    train_times_all = [temporal_train_test_split(sequences[u], train_ratio)[0] for u in active_users]
    global_prof = GlobalSeasonalProfile().fit(train_times_all)
    
    results = []
    
    for uid in tqdm(active_users):
        times = sequences[uid]
        train, test = temporal_train_test_split(times, train_ratio)
        if len(test) < 1: continue
        
        baseline = GlobalSeasonalPoisson(global_prof, train)
        model = model_factory(baseline)
        model.fit(train)
        
        kernel_type = getattr(model, 'kernel', 'exp')
        p = model.params
        
        mu_scale, alpha, beta, delta = 1.0, 0.0, 1.0, 1.0
        
        if kernel_type == 'power':
            mu_scale, alpha, beta, delta = p
        elif kernel_type == 'exp':
            mu_scale, alpha, beta = p
        
        br_ratio = alpha / beta if beta > 1e-9 else 0.0

        ll_base = calc_test_ll(baseline, train, test)
        ll_model = calc_test_ll(model, train, test)
        
        rmse_base, _, _ = evaluate_forecast(baseline, train, test)
        rmse_model, _, _ = evaluate_forecast(model, train, test)
        
        loss_base, _, _ = evaluate_next_day_probability(baseline, train, test)
        loss_model, _, _ = evaluate_next_day_probability(model, train, test)

        results.append({
            'user_id': uid,
            'experiment': label,
            'kernel': kernel_type,
            'penalty': getattr(model, 'penalty_weight', 0),
            'events': len(times),
            
            'mu_scale': mu_scale,
            'alpha': alpha,
            'beta': beta,
            'delta': delta,
            'branching_ratio': br_ratio,
            
            'll_base': ll_base,
            'll_model': ll_model,
            'imp_ll': ll_model - ll_base,
            'win_ll': ll_model > ll_base,
            
            'rmse_base': rmse_base,
            'rmse_model': rmse_model,
            'imp_rmse': (1 - rmse_model/rmse_base) * 100,
            
            'logloss_base': loss_base,
            'logloss_model': loss_model,
            'imp_logloss': (loss_base - loss_model) * 100
        })
            
    return pd.DataFrame(results)