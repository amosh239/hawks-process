import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .utils import temporal_train_test_split
from .models import GlobalSeasonalProfile, GlobalSeasonalPoisson

def plot_user_timeline(sequences, uid, ax=None, title="User"):
    if ax is None: fig, ax = plt.subplots(figsize=(12, 2))
    times = sequences[uid]
    ax.scatter(times, [1]*len(times), marker='|', s=500, color='black', alpha=0.6)
    ax.set_yticks([])
    ax.set_title(f"{title}: {uid} ({len(times)})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def analyze_user_intensity(sequences, uid, factory, ratio=0.8):
    times = sequences[uid]
    train, test = temporal_train_test_split(times, ratio)
    
    prof = GlobalSeasonalProfile().fit([train])
    base = GlobalSeasonalPoisson(prof, train)
    model = factory(base).fit(train)
    
    p_names = ['Mu', 'Alpha', 'Beta', 'Delta']
    p_str = ", ".join([f"{n}:{v:.2f}" for n, v in zip(p_names, model.params)]) if hasattr(model, 'params') else ""
    
    grid = pd.date_range(train[0], test[-1], freq='1H')
    hist, idx, lams, bases = [], 0, [], []
    events = sorted(list(train) + list(test))
    mu_scale = model.params[0] if hasattr(model, 'params') and len(model.params) > 2 else 1.0

    for t in grid:
        while idx < len(events) and events[idx] <= t:
            hist.append(events[idx]); idx += 1
        lams.append(model.get_intensity(t, hist))
        bases.append(base.get_intensity(t) * mu_scale)

    plt.figure(figsize=(14, 6))
    plt.scatter(train, [0]*len(train), c='k', label='Train', zorder=5)
    plt.scatter(test, [0]*len(test), marker='x', c='r', label='Test', zorder=5)
    plt.plot(grid, lams, c='b', alpha=0.8, label='Hawkes')
    plt.plot(grid, bases, c='g', ls='--', alpha=0.5, label='Base')
    plt.title(f"User {uid} | {p_str}"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()


def plot_parameter_distributions(df):
    has_delta = 'delta' in df and df['delta'].std() > 0
    cols = 4 if has_delta else 3
    fig, ax = plt.subplots(2, cols, figsize=(5*cols, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    params = ['mu_scale', 'alpha', 'beta'] + (['delta'] if has_delta else [])
    titles = ['Mu Scale', 'Alpha', 'Beta', 'Delta']
    colors = ['purple', 'orange', 'green', 'gray']
    
    y_metric = 'imp_ll' if 'imp_ll' in df else 'imp_rmse'

    for i, (col, title, color) in enumerate(zip(params, titles, colors)):
        sns.histplot(df[col], ax=ax[0, i], color=color, kde=True)
        ax[0, i].set_title(title)
        if col == 'mu_scale': ax[0, i].axvline(1.0, c='k', ls='--')
        
        sns.scatterplot(data=df, x=col, y=y_metric, hue='win_ll', 
                        ax=ax[1, i], palette={True:'b', False:'r'}, alpha=0.6)
        ax[1, i].set_title(f'{title} vs {y_metric}')
        ax[1, i].axhline(0, c='k', ls='--')
        if col == 'mu_scale': ax[1, i].axvline(1.0, c='k', ls='--')

    plt.show()
    print(df[params].describe().T[['mean', 'std', 'min', '50%', 'max']])