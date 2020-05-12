import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

def spectrum_plotter(ax, phi, eddyco, *args, **kwargs):

    tke = eddyco['tke']
    epsilon = eddyco['epsilon']

    k = phi.index.to_numpy()
    kol_spectrum = compute_kolmogov_spectrum(k, epsilon)

    phi.plot(ax=ax, logx=True, logy=True)
    line, = ax.plot(k, kol_spectrum, '--k', label='-5/3 line')
    ax_kwargs = {'xlabel': "2 pi / wavelength",
         'title' : f"E(k), tke={tke:4.3f}, dissipation={epsilon:4.3f}"}
    ax_kwargs.update(kwargs)
    ax.set(**ax_kwargs)
    ax.legend(loc='lower left')
    ax.grid(True, 'both', 'both')
    return ax

def sonic_plotter(ax_00, ax_01, sonic_df):
    sonic_df[['u','v','w']].plot(ax=ax_00)
    ax_00.legend(loc='upper left')
    ax_00.grid(True,'both','both')
#    sonic_df['Ts_K'].plot(ax=ax_01, label='Ts_K')
#    sonic_df['H2O'].plot(ax=ax_01, secondary_y=True, mark_right=True)
    ax_01 = sonic_df[['Ts_K', 'H2O']].plot(ax=ax_01, secondary_y=['H2O'])
    ax_01.set_ylabel('Ts_K');
    ax_01.right_ax.set_ylabel("H2O")
    ax_01.legend(loc='upper left')
    ax_01.grid(True,'both','both')
                                        
    return ax_00, ax_01

def add_viz(ax, viz_ser):
    viz_ser.plot(secondary_y=True, ax=ax, color='k', ls='--', alpha=0.3)
    ax.grid(True, 'both', 'both')
    ax.right_ax.set(yscale='log')
    ax.right_ax.grid(True, 'both', 'both', color='k', ls='--',
                     alpha=0.1)
    return ax

# Section: For radiosonde plots (verticla profile) 
def vertical_subplotter(var_df, axes=None, 
                        y_spines=True, ylabel=None,
                        **kwargs):
    N = var_df.shape[1]
    if axes is None:
        fig, axes = plt.subplots(ncols=N)
    for i in range(N):
        axes[i].plot(var_df.iloc[:, i].values,
                     var_df.index,
                     **kwargs)
        axes[i].set(title=var_df.columns[i])
    if ylabel:
        axes[0].set(ylabel=ylabel)
    adjust_spines(axes[0], ['left', 'bottom'])
    spine_option = {True: ['left', 'bottom'],
                    False: ['bottom']}
    for i in range(1, N):
        adjust_spines(axes[i], spine_option[y_spines])
    return axes

def vertical_subplot_set(axes, grid=None, **kwargs):
    N = len(axes)
    for i in range(N):
        axes[i].set(**kwargs)
    for i in range(N):
        axes[i].set(**kwargs)
        if grid:
            axes[i].grid(True, 'both', 'both')
    return axes
            


