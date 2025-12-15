import pandas as pd
from tqdm import tqdm
from .models import GlobalSeasonalProfile, GlobalSeasonalPoisson
from .utils import temporal_train_test_split
from .metrics import calc_test_ll, evaluate_forecast, evaluate_next_day_probability

def run_validation_loop(sequences, model_factory, label="Exp", min_events=30, train_ratio=0.8):
    active_users = sequences[sequences.apply(len) >= min_events].index
    print(f"Running '{label}': N={len(active_users)} users")
    
    print("Fitting Global Profile...")
    all_train_times = []
    for uid in active_users:
        tr, _ = temporal_train_test_split(sequences[uid], train_ratio)
        all_train_times.append(tr)
    global_prof = GlobalSeasonalProfile().fit(all_train_times)
    
    results = []
    for uid in tqdm(active_users):
        times = sequences[uid]
        train, test = temporal_train_test_split(times, train_ratio)
        if len(test) < 1: continue
            
        try:
            poisson = GlobalSeasonalPoisson(global_prof, train)
            
            challenger = model_factory(poisson)
            challenger.fit(train)
            
            alpha_val, beta_val, mu_scale = (0.0, 1.0, 1.0) 
            
            if hasattr(challenger, 'params'):
                params = challenger.params
                if len(params) == 3:
                    mu_scale, alpha_val, beta_val = params
                elif len(params) == 2:
                    mu_scale = 1.0
                    alpha_val, beta_val = params
            
            br_ratio = alpha_val / beta_val if beta_val > 1e-9 else 0.0

            ll_p = calc_test_ll(poisson, train, test)
            ll_h = calc_test_ll(challenger, train, test)
            
            rmse_p, _, _ = evaluate_forecast(poisson, train, test)
            rmse_h, _, _ = evaluate_forecast(challenger, train, test)

            loss_p, _, _ = evaluate_next_day_probability(poisson, train, test)
            loss_h, _, _ = evaluate_next_day_probability(challenger, train, test)
            
            results.append({
                'user_id': uid,
                'experiment': label,
                'events_total': len(times),

                'mu_scale': mu_scale, 'alpha': alpha_val, 'beta': beta_val, 'branching_ratio': br_ratio,

                'll_baseline': ll_p,
                'll_challenger': ll_h,
                'rmse_baseline': rmse_p,
                'rmse_challenger': rmse_h,

                'imp_ll': ll_h - ll_p,
                'imp_rmse': (1 - rmse_h/rmse_p) * 100,
                'imp_logloss': (loss_p - loss_h) * 100,
                'hawkes_win_ll': ll_h > ll_p 
            })
            
        except Exception as e:
            # Раскомментируй, если опять будет пусто, чтобы видеть ошибку!
            # print(f"Error {uid}: {e}") 
            continue
            
    return pd.DataFrame(results)