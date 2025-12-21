import numpy as np
import pandas as pd
from tqdm import tqdm

class HawkesSimulator:
    def __init__(self, t_max=24*14): # 2 недели
        self.t_max = t_max
        
    def _get_baseline_intensity(self, t, mu_level):
        # Сезональность: синусоида с периодом 24ч (пик в 12:00)
        # mu_level - средний уровень активности юзера
        cycle = np.sin(2 * np.pi * (t % 24) / 24 - np.pi/2) + 1.0 # 0..2
        return mu_level * (0.5 + 0.25 * cycle) # mu +/- вариации

    def simulate_user(self, alpha, beta, mu_level):
        t = 0
        history = []
        # Текущее значение рекурсивной суммы ядра (для скорости)
        # R(t) = sum(alpha * exp(-beta*(t-ti)))
        R = 0 
        last_t = 0
        
        # Upper bound для thinning (mu_max + R_current)
        # Так как R затухает, текущее R - это максимум R на ближайшее будущее
        mu_max = mu_level * 1.5 
        
        while True:
            # 1. Верхняя граница интенсивности (Majorant)
            lambda_bar = mu_max + R
            
            # 2. Кандидат на следующее событие
            step = np.random.exponential(1.0 / lambda_bar)
            t += step
            if t >= self.t_max: break
            
            # 3. Обновляем R до момента t (затухание)
            R *= np.exp(-beta * step)
            
            # 4. Считаем реальную интенсивность в точке t
            lambda_true = self._get_baseline_intensity(t, mu_level) + R
            
            # 5. Thinning (принимаем с вероятностью true / bar)
            if np.random.rand() < (lambda_true / lambda_bar):
                history.append(t)
                # Добавляем прыжок alpha
                R += alpha
                
        # Конвертируем в Datetime (старт 2024-01-01)
        start_date = pd.Timestamp("2024-01-01")
        return [start_date + pd.Timedelta(hours=h) for h in history]

    def generate_dataset(self, n_users=50, seed=42):
        np.random.seed(seed)
        data = {}
        truth = []
        
        print(f"Generating synthetic data for {n_users} users...")
        for i in tqdm(range(n_users)):
            uid = f"user_{i}"
            
            # Генерируем параметры
            # beta: скорость забывания (1.0 - 5.0)
            beta = np.random.uniform(1.0, 5.0)
            
            # alpha: сила возбуждения. 
            # Чтобы процесс не взрывался, branching ratio (alpha/beta) должен быть < 1
            # Возьмем ratio от 0 (пуассон) до 0.9 (сильный хоукс)
            ratio = np.random.uniform(0.0, 0.9)
            alpha = beta * ratio
            
            # mu: базовая активность (0.1 - 0.5 событий в час)
            mu = np.random.uniform(0.05, 0.3)
            
            events = self.simulate_user(alpha, beta, mu)
            
            if len(events) > 10: # Оставляем только живых
                data[uid] = events
                truth.append({
                    'user_id': uid,
                    'true_mu': mu,
                    'true_alpha': alpha,
                    'true_beta': beta,
                    'true_ratio': ratio,
                    'events': len(events)
                })
                
        return pd.Series(data), pd.DataFrame(truth)