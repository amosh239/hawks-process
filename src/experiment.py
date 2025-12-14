import pandas as pd
from tqdm import tqdm
from .models import GlobalSeasonalProfile, GlobalSeasonalPoisson
from .utils import temporal_train_test_split
from .metrics import calc_test_ll, evaluate_forecast

def run_validation_loop(sequences, model_factory, label="Exp", min_events=30, train_ratio=0.8):
    """
    model_factory: Функция (lambda), которая принимает baseline_model и возвращает HawkesExp.
    label: Метка для логов
    """
    # 1. Filter
    active_users = sequences[sequences.apply(len) >= min_events].index
    print(f"Running '{label}': N={len(active_users)} users")
    
    # 2. Global Profile
    print("Fitting Global Profile...")
    all_train_times = []
    for uid in active_users:
        tr, _ = temporal_train_test_split(sequences[uid], train_ratio)
        all_train_times.append(tr)
    global_prof = GlobalSeasonalProfile().fit(all_train_times)
    
    # 3. Loop
    results = []
    for uid in tqdm(active_users):
        times = sequences[uid]
        train, test = temporal_train_test_split(times, train_ratio)
        if len(test) < 1: continue
            
        try:
            # A. Baseline
            poisson = GlobalSeasonalPoisson(global_prof, train)
            
            # B. Challenger (Injected Model)
            challenger = model_factory(poisson)
            challenger.fit(train)
            
            # C. Metrics
            # Extract params safely
            alpha_val, beta_val = (0.0, 1.0) # Значения по умолчанию
            if hasattr(challenger, 'params'):
                alpha_val, beta_val = challenger.params
            
            # Считаем Branching Ratio (Сила возбуждения)
            # Если beta очень маленькая, может быть деление на ноль, защитимся
            br_ratio = alpha_val / beta_val if beta_val > 1e-9 else 0.0

            # Calculate LL
            ll_p = calc_test_ll(poisson, train, test)
            ll_h = calc_test_ll(challenger, train, test)
            
            # Calculate RMSE
            rmse_p, _, _ = evaluate_forecast(poisson, train, test)
            rmse_h, _, _ = evaluate_forecast(challenger, train, test)
            
            results.append({
                'user_id': uid,
                'experiment': label,
                'events_total': len(times),
                'alpha': alpha_val,
                'beta': beta_val,
                'branching_ratio': br_ratio,
                'll_baseline': ll_p,
                'll_challenger': ll_h,
                'rmse_baseline': rmse_p,
                'rmse_challenger': rmse_h,
                'imp_rmse': (1 - rmse_h/rmse_p) * 100,
                'hawkes_win': ll_h > ll_p 
            })
            
        except Exception as e:
            # print(f"Error {uid}: {e}")
            continue
            
    return pd.DataFrame(results)