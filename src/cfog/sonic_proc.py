import math
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft
import ipdb as debugger

from .utils import *
from .qc_accessor import QualityControlAccessor


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
