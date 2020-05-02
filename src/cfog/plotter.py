import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import compute_kolmogov_spectrum

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

