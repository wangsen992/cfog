'''Tool functions making use of the package for direct applications'''

# Computation of eddy-covariance and dissipation data for a longer period of
# time. This function should take in a dataframe of u, v, w, T, q with pandas
# datetimeIndex. Thermodyanmics calculations should be completed before
# entering data into the function.Note there should also be a summary of the
# statoinarity measures along with the computation of eddy-covariance data. 

import numpy as np
import pandas as pd

from pyqc import *
from .utils import *

def summarize_sonic(df, P=1.01e5, block = 50):
    """Compute everything needed from sonic. 

    Returns: Dict with keys defined as, 
        qc : quality control values for each variable
        stationarity_measure: as name suggests
        eddyco : eddy covariance values
        phi : turbulent spectrum 
        meta : useful values
    """
 
    result = {}
    time = df.index[0]
    # Prepare pressure
    if isinstance(P, (float, int)):
        P = pd.Series(np.full(df.shape[0], P), index=df.index)

    if not all(df[['v','w']].mean().abs() < 1e-3):
        raise ValueError("Input sonic df must be rotated first, use "
                         "df.sonic.rotate_uvw()")

    # QC 
    df_adjusted = df.qc.despike()
    result['qc'] = df_adjusted.qc.describe()
    result['stationarity_measure'] = df_adjusted.sonic.mean_stationarity_values
    df_thermo = df_adjusted.eddyco.compute_thermo_properties(P)
    result['eddyco'] = df_thermo.eddyco.cov_ra
    U = np.abs(df_adjusted['u'].mean())

    # Spectrum
    phi_dict = {}
    for E_ii, u_i in zip(['E_11','E_22','E_33'], list('uvw')):
       phi_dict[E_ii] = compute_cross_spectra(df_adjusted, u_i, u_i,
                                                         block)
    phi = pd.DataFrame(phi_dict)
    phi['E_k'] = phi[['E_11','E_22', 'E_33']].sum(axis=1)
    result['phi'] = phi

    # Compute dissipation
    k_max = 2 * np.pi / (U * 0.04)
    k_min = k_max * 0.5
    alpha = 1.7
    result['eddyco']['epsilon'] = fit_epsilon(np.abs(phi['E_k'].values),
                                              phi.index.to_numpy(),
                                              k_range=(k_min, k_max))
    # Taylor Microscale
    nu = 1.48e-5
    u_rms = np.sqrt(result['eddyco']['uu'])
    result['eddyco']['lam_g'] = np.sqrt(15 * nu * u_rms**2 / \
                                        result['eddyco']['epsilon'])
    result['meta'] = dict(time=time, U=U,
                          tke=result['eddyco']['tke'],
                          epsilon=result['eddyco']['epsilon'])

    return result 





# A wavelet computation function. 
def wavelet_analysis(df):
    raise NotImplementedError
