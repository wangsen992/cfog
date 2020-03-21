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


def mean_ptp_ratio(x, window='2T'):
    '''Compute the ratio of running mean range to record mean

    Parameters:
        x:      pd.Series with DatetimeIndex. Record to measure
        window: pd.DateOffset string. Running window size
    '''

    x_normed = (x - x.min()) / (x.max() - x.min())
    x_mean_rolling = x_normed.rolling(window).mean()
    return (x_mean_rolling.max() - x_mean_rolling.min())/x_normed.mean()

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

def spike_flags(x,
                window='2T',
                stride='10s',
                factor=3.5):
    '''Flag data entry indice identified with spikes using running mean and std
    method.'''

    window = pd.Timedelta(window)
    stride = pd.Timedelta(stride)

    flag_indice = set()
    for x_sub in series_rolling(x, window, stride):
        x_mean = x_sub.mean()
        x_std = x_sub.std()
        sub_flagged_indice =\
                x_sub[np.abs(x_sub-x_mean) > (factor *x_std)].index.to_list()
        flag_indice = flag_indice.union(sub_flagged_indice)

    return flag_indice

def hist_based_flags(x,
                     window='2T',
                     bins=100,
                     pct_thres=0.5):

    window = pd.Timedelta(window)
    stride = window / 2

    flag_indice = set()
    for x_sub in series_rolling(x, window, stride):
        x_sub_no_nan = x_sub.dropna() 
        
        if x_sub_no_nan.size/x_sub.size < 0.2:
            hist, bins = np.histogram(x_sub.dropna().values,
                                     bins=bins,
                                     range=[x_sub.min(), x_sub.max()])
            if ((hist == 0).sum() / hist.size) >= pct_thres:
               flag_indice = flag_indice.union(x_sub.index.to_list())

    return flag_indice

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
#===========*==========*========Accessors===========*============*===========*============*

@pd.api.extensions.register_dataframe_accessor("sonic")
class SonicAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._init_sonic_options()

    @staticmethod
    def _validate(obj):
        if not all([name in obj for name in ['u','v','w']]):
            raise KeyError("Must contain columns 'u', 'v', 'w'")

        if not isinstance(obj.index, 
                          (pd.core.indexes.datetimes.DatetimeIndex,
                           pd.core.indexes.timedeltas.TimedeltaIndex)):
            raise IndexError("DatetimeIndex/TimedeltaIndex must be used."\
                            +"Current index type is {}".format(type(obj.index)))

    def _init_sonic_options(self):
        self._options = dict()
        # compute original data info
        duration = self._obj.index[-1] - self._obj.index[0]

        # assign init options
        self._options['mean_stationarity_window'] = duration / 20
        self._options['mean_stationarity_q'] = 0.95


    @property
    def adjusted_uvw(self):
        if not hasattr(self, '_adjusted_uvw'):
            self._rotate_uvw(inplace=True)
        return self._adjusted_uvw

    def rotate_uvw(self, inplace=False):
        if inplace == True:
            self._obj[['u','v','w']] = self.adjusted_uvw
        else:
            new_df = self._obj.copy()
            new_df[['u','v','w']] = self.adjusted_uvw.values
            return new_df

    def _rotate_uvw(self,
                    method='double rotation',
                    inplace=False):
        u = self._obj['u'].values
        v = self._obj['v'].values
        w = self._obj['w'].values
 
        if method == 'double rotation':
            u, v, w = rotate_uvw(u, v, w)

        adjusted_uvw =  pd.DataFrame(np.array([u,v,w]).T,
                                     columns=['u','v','w'],
                                     index=self._obj.index)
        if inplace == False:
            return adjusted_uvw
        else:
            self._adjusted_uvw = adjusted_uvw

    # Obtain mean stationarity test values
    @property
    def mean_stationarity_values(self):
        if not hasattr(self, '_mean_stationarity_values'):
            self._test_mean_horizontal_stationarity(inplace=True)
        return self._mean_stationarity_values

    def _test_mean_horizontal_stationarity(self,
                                           inplace=False):

        u = self._obj['u'].values
        v = self._obj['v'].values
        w = self._obj['w'].values
 
        mean_stationarity_values = \
            test_mean_horizontal_stationarity(u,v,w,self._obj.index,
                                              window=self._options['mean_stationarity_window'],
                                              q=self._options['mean_stationarity_q'])

        if inplace == False:
            return mean_stationarity_values
        else:
            self._mean_stationarity_values = mean_stationarity_values

 
@pd.api.extensions.register_series_accessor("qc")
class QualityControlAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._init_qc_options()

    @staticmethod
    def _validate(obj):
        # Validate on what? 
        if not isinstance(obj.index, 
                          (pd.core.indexes.datetimes.DatetimeIndex,
                           pd.core.indexes.timedeltas.TimedeltaIndex)):
            raise IndexError("DatetimeIndex/TimedeltaIndex must be used."\
                            +"Current index type is {}".format(type(obj.index)))

    def _init_qc_options(self):
        self._options = dict()
        # compute original data info
        duration = self._obj.index[-1] - self._obj.index[0]

        # assign init options
        self._options['spike_window'] = duration / 20
        self._options['spike_stride'] = self._options['spike_window'] / 20
        self._options['spike_factor'] = 3.5
        self._options['hist_window'] = self._options['spike_window']
        self._options['hist_bins'] = 200
        self._options['hist_pct_thres'] = 0.8
        self._options['stationarity_window'] = self._options['spike_window']

    # Spike detection
    @property
    def spike_indice(self):
        if not hasattr(self, '_spike_indice'):
            self._compute_spike_indice(inplace=True)
        return self._spike_indice

    def despike(self, inplace=False):
        if inplace == True:
            self._obj[self.spike_indice] = np.nan
        else:
            new_ser = self._obj.copy()
            new_ser[self.spike_indice] = np.nan
            return new_ser

    def _compute_spike_indice(self,
                              inplace=False):

        spike_indice = spike_flags(self._obj, 
                                   window=self._options['spike_window'], 
                                   stride=self._options['spike_stride'], 
                                   factor=self._options['spike_factor'])

        if inplace == True:
            self._spike_indice = spike_indice
        else:
            return spike_indice

    # Amplitude resolution and dropouts detection
    @property
    def hist_indice(self):
        if not hasattr(self, '_hist_indice'):
            self._compute_hist_indice(inplace=True)
        return self._hist_indice

    def _compute_hist_indice(self,
                          inplace=False):

        hist_indice = hist_based_flags(self._obj, 
                                       window=self._options['hist_window'], 
                                       bins=self._options['hist_bins'], 
                                       pct_thres=self._options['hist_pct_thres'])

        if inplace == True:
            self._hist_indice = hist_indice
        else:
            return hist_indice


    # Measure single variable nonstationarity by comparing normalized rolling
    # mean range over record mean
    @property
    def stationarity_measure(self):
        if not hasattr(self, '_stationarity_measure'):
            self._compute_stationarity_measure(inplace=True)
        return self._stationarity_measure

    def _compute_stationarity_measure(self, inplace=False):

        stationarity_measure = mean_ptp_ratio(self._obj,
                                              window=self._options['stationarity_window'])

        if inplace == True:
            self._stationarity_measure = stationarity_measure
        else:
            return stationarity_measure

    def describe(self):
        return dict(mean = self._obj.mean(),
                    std = self._obj.std(),
                    skew= self._obj.skew(),
                    kurt= self._obj.kurtosis(),
                    pct_null = self._obj.isna().sum()/self._obj.size,
                    stationarity_measure = self.stationarity_measure,
                    pct_spike_flag = len(self.spike_indice)/self._obj.size,
                    pct_hist_flag = len(self.hist_indice)/self._obj.size)

@pd.api.extensions.register_dataframe_accessor("eddyco")
class EddyCovarianceAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._init_eddyco_options()

    @staticmethod
    def _validate(obj):
        # Validate on what? 
        if not all([name in obj for name in ['u','v','w', 'Ts_K', 'H2O']]):
            raise KeyError("Must contain columns 'u', 'v', 'w', 'Ts_K', 'H2O")

        if not isinstance(obj.index, 
                          (pd.core.indexes.datetimes.DatetimeIndex,
                           pd.core.indexes.timedeltas.TimedeltaIndex)):
            raise IndexError("DatetimeIndex/TimedeltaIndex must be used."\
                            +"Current index type is {}".format(type(obj.index)))

        if obj['Ts_K'].mean() < 200:
            raise ValueError("Sonic temperature must be in kelvin")

        if obj['H2O'].mean() > 1:
            raise ValueError("Vapor content must be in kgm-3")

    def _init_eddyco_options(self):
        self._options = {}
        
        duration  = self._obj.index[-1] - self._obj.index[0]

        self._options['flux_stationarity_window_count'] = 5

    def compute_thermo_properties(self, P, inplace=False):
        '''Compute the required thermodynamics properties from existing
        variables. Pressure is supplied from other data sources.'''

        # Make sure pressure series has matching datetimeIndex
        P = P[self._obj.index[0]:self._obj.index[-1]]
        P_upsample = \
            P.resample(self._obj.index.freq).interpolate().reindex(self._obj.index)
       
        result_dict = compute_thermo_from_sonic(Ts_K=self._obj['Ts_K'].values,
                                                     P=P_upsample.values,
                                                     H2O=self._obj['H2O'].values)
        result= pd.DataFrame(result_dict, index=self._obj.index)
        
        if inplace == True:
            self._obj = pd.concat([self._obj, result], axis=1)
        else:
            new_df = self._obj.copy()
            return pd.concat([new_df, result], axis=1)
            
    @property
    def cov_ra(self):
        '''Covariance from Reynolds averaging assuming stationary'''

        if not hasattr(self, '_cov_ra'):
            self._compute_cov_ra(inplace=True)
        return self._cov_ra

    def _compute_cov_ra(self, inplace=False):
        
        if not all([name in self._obj for name in ['T','q']]):
            raise ValueError("Thermodynamic properties have not been calculated.")

        raw_df = self._obj[['u','v','w','T','q']]
        fluc_df = raw_df.sub(raw_df.mean(axis=0), axis=1)
        cov_df = fluc_df.cov()
        cov_results = pd.Series(dict(uu=cov_df.loc['u','u'],
                                     vv=cov_df.loc['v','v'],
                                     ww=cov_df.loc['w','w'],
                                     uv=cov_df.loc['u','v'],
                                     uw=cov_df.loc['u','w'],
                                     vw=cov_df.loc['v','w'],
                                     Tw=cov_df.loc['T','w'],
                                     qw=cov_df.loc['q','w'],
                                     tke=0.5 * \
                                      np.mean((fluc_df[['u','v','w']] ** 2).sum(axis=1))))
        if inplace == False:
            return cov_results
        else:
            self._cov_ra = cov_results

    @property
    def flux_stationarity_measure(self):
        if not hasattr(self, '_flux_stationarity_measure'):
            self._compute_flux_stationarity_measure(inplace=True)
        return self._flux_stationarity_measure
    
    def _compute_flux_stationarity_measure(self, inplace=False, debug=False):
        
        # First check if record flux is calculated
        if not hasattr(self, '_cov_ra'):
            self._compute_cov_ra(inplace=True)

        # Then compute the average windowed cov values
        duration = self._obj.index[-1] - self._obj.index[0]
        cov_list = []
        for name, df in self._obj.resample(duration \
                                           /self._options['flux_stationarity_window_count']):
            cov_list.append(df.eddyco.cov_ra)

        cov_df = pd.DataFrame(cov_list)
        flux_stationarity_measure = np.abs((cov_df.mean(axis=0) - \
                                            self.cov_ra) / self.cov_ra)

        if inplace == False:
            if debug == False:
                return flux_stationarity_measure
            else:
                return flux_stationarity_measure, cov_df
        else:
            self._flux_stationarity_measure = flux_stationarity_measure
