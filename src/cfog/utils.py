import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft
import ipdb as debugger

def compute_thermo_from_sonic(Ts_K, P, H2O):
    '''Calculate other thermodynamics properties from sonic temperature,
    pressure and vapor content.

    Parameters: 
        Ts_K:       Sonic temperature [Kelvin]
        P:          Pressure [Pa]
        H2O:        Vapor content [kg/m-3] 

    Notes:
        Inputs are all np.arrays without nan values.

        This method solves for virtual temperature from sonic temperature by
        iteratively calculating dry air density, vapor pressure.
        '''

    # Enforce the no null data requirements of the function. 
#    if np.isnan(Ts_K).sum() or \
#       np.isnan(P).sum() or \
#       np.isnan(H2O).sum():

#        raise ValueError("Does not accept ndarrays with NaN values.")

    # Required constants
    Rd = 287.04 # Gas constant for dry air [JK-1kg-1]

    # Iteratively solve for the other variables
    Tv_K_old = Ts_K.copy()
    Tv_K_new = np.zeros(Tv_K_old.shape)
    err_thres = 1.
    err = 10
    iter_count = 0
    max_iter = 100
    while err > err_thres:
        rho_d = P / (Rd * Tv_K_old)
        r = H2O / rho_d
        e = P * r / 0.622
        Tv_K_new = Ts_K + 0.02 * e / 100
        iter_count += 1
        err = np.nansum(np.abs(Tv_K_new - Tv_K_old))
        print("Iteration {i}: Current residual = {res}"\
              .format(i=iter_count, res=err))
        Tv_K_old = Tv_K_new.copy()
        if iter_count > max_iter:
            print("Max iteration number exceeded.  Break.")
            break
    # Output the required variables after completion of while-loop 
    else:
        T_K = Ts_K - 0.1 * e / 100
        T = T_K - 273.15
        es = 6.11 * 10 ** (7.5*T / (T + 237.3)) * 100
        q = r / (1+r)
        return {'rho_d':rho_d,
                    'r' : r,
                    'q' : q,
                    'e' : e,
                    'es':es,
                    'RH':e/es,
                    'Tv_K' : Tv_K_new,
                    'T_K' : T_K,
                    'T' : T}

def wind_dir(u, v):
    return np.degrees(np.atan2(-v, -u)) % 360

def rotate_uvw(u, v, w): 
    '''Set crosswind(v) and vertical (w) velocity mean to zero through double
    rotation.

    Parameters:
        u, v, w:    ndarrays(n,) velocity components [m/s]
    '''

    # Set up double rotation matrix.
    # R_01: Rotation around z-axis
    R_01 = lambda a: np.array([[np.cos(a), np.sin(a), 0],
                              [-np.sin(a),np.cos(a), 0],
                              [0,0,1]])
    # R_12: Rotation around y-axis
    R_12 = lambda b: np.array([[np.cos(b), 0, np.sin(b)],
                              [0, 1, 0],
                              [-np.sin(b), 0, np.cos(b)]])

    u_mean0, v_mean0 = np.nanmean(u), np.nanmean(v)
    alpha_dr = np.arctan(v_mean0 / u_mean0)
    u_1, v_1, w_1 = R_01(alpha_dr) @ np.array([u,v,w])
    u_mean1, w_mean1 = np.nanmean(u_1), np.nanmean(w_1)
    beta_dr = np.arctan(w_mean1 / u_mean1)
    u_2, v_2, w_2 = R_12(beta_dr) @ np.array([u_1,v_1,w_1])
    return u_2, v_2, w_2

def test_mean_horizontal_stationarity(u, v, w, dti, window='2T',q=0.95):
    '''Test stationarity of horizontal mean veloicty. Compute the speed
    reduction factor, alongwind/crosswind/vector wind relative nonstationarity. 
    Parameters:
        u, v, w:    ndarrays(n,) 
                    Velocity components already adjusted for
                    rotation.

        dti:        pandas.DatetimeIndex
                    Corresponding indice for the velocity values. 

        window:     pandas DateOffet strings.
                    window size used for rolling mean. 

        q:          float.0 < q < 1
                    Quantile for computing delta in u,v,w
    '''
    window = pd.Timedelta(window)
    # More comments here 
    u_arr = np.array([u,v,w]).T
    u_df = pd.DataFrame(u_arr, columns=['u','v','w'], index=dti)
    result = {} 
    result['U_reduction'] =  np.sqrt((np.nanmean(u_arr, axis=0)**2).sum()) \
            / np.nanmean(np.sqrt((u_arr**2).sum(axis=1)))
    u_mean = u_df.mean(axis=0)
    u_mean_rolling = u_df.rolling(window).mean()
    u_mean_delta = u_mean_rolling.quantile(q=q,axis=0) \
                 - u_mean_rolling.quantile(q=1-q,axis=0)

    result['RNu'] = u_mean_delta['u'] / np.abs(u_mean['u'])
    result['RNv'] = u_mean_delta['v'] / np.abs(u_mean['u'])
    result['RNS'] = np.sqrt(u_mean_delta['u']**2 + u_mean_delta['v']**2) \
                            / np.abs(u_mean['u'])

    return result

def series_rolling(x, window, stride):
    window_size = math.floor(window / x.index.freq)
    stride_size = math.floor(stride / x.index.freq) 
    end_index = x.shape[0] - window_size

    if stride_size == 0: 
        stride_size =1

    for i in np.arange(0, end_index, stride_size):
        yield x.iloc[i: i + window_size]

def compute_cross_spectra(input_df, x_name, y_name, block=None,
                          interpolate='zero'):
# Translate timestamp to distance with mean wind speed
    try:
        U = np.abs(input_df['u'].mean())
    except KeyError:
        raise KeyError("along wind component u must be present")
    
    if input_df.shape[0] < 100:
        return None
        
    t = ((input_df.index - input_df.index[0]) / pd.Timedelta('1s')).to_numpy()
    r = t * U

    # Extract to numpy values and fill in the nan data with interpolation
    x = np.copy(input_df[x_name].values)
    y = np.copy(input_df[y_name].values)

    idx_nan_x = np.isnan(x)
    idx_nan_y = np.isnan(y)
    if interpolate == 'zero':
        x[idx_nan_x] = 0
        y[idx_nan_y] = 0
    elif interpolate == 'hermite':
        x[idx_nan_x] = interp.pchip_interpolate(r[~idx_nan_x], x[~idx_nan_x], r[idx_nan_x], der=1)
        y[idx_nan_y] = interp.pchip_interpolate(r[~idx_nan_y], x[~idx_nan_y], r[idx_nan_y], der=1)
    
    # Produce fluctuation value
    x = x - x.mean()
    y = y - y.mean()

    # Compute covariance
    N = x.size
    L = r.max() / (2 * np.pi)
    if block is None:
        print("block fft not init...")
        # Compute fourier transform
        x_hat, y_hat = 2 * np.pi * L / N * fft.fft(np.vstack([x,y]))
        Phi_xy = x_hat * np.conjugate(y_hat) / (4 * np.pi**2 * L)  # 2piL to adjust for deconvolution, same below
        k = np.arange(N) / L
    else:
        print("block fft init...")
        rmd = N%block
        if rmd == 0:
            x_tmp = x.reshape(block, -1)
            y_tmp = y.reshape(block, -1)
        else:
            x_tmp = x[:-(N%block)].reshape(block, -1)
            y_tmp = y[:-(N%block)].reshape(block, -1)
        NN = x_tmp.shape[1]
        L_block = L / block
        xy_hat = (2 * np.pi * L) / NN * fft.fft(np.vstack([x_tmp, y_tmp]))
        Phi_xy = np.mean(xy_hat[:block, :] * np.conjugate(xy_hat[block:, :]), axis=0) / block/ (4 * np.pi**2 * L)
        k = np.arange(NN) / L_block

    return pd.Series(Phi_xy[1:], index=k[1:], name=x_name+y_name+' spectra')


def fit_epsilon(Ek, k, alpha=1.7, k_range=(1,50)):
    '''Fit for epsilon based on -5/3 power law'''
    epsilon = np.exp(3/2 * (np.log(Ek) + 5/3 * np.log(k) - np.log(alpha)))
    return epsilon[(k > k_range[0]) & (k < k_range[1])].mean()
