import math
import copy
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.fftpack as fft
import ipdb as debugger

from .utils import *

#=============================General Accessors==============================#
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

        self._default_options = copy.copy(self._options)
        self._old_options = copy.copy(self._options)

    def set_options(self, **kwargs):
        self._old_options.update(self._options)
        self._options.update(kwargs)

    def reset_default_options(self):
        self._old_options.update(self._options)
        self._options.update(self._default_options)

    @property
    def options(self):
        return self._options

    @property
    def option_is_updated(self):
        return self._options != self._old_options

    # Spike detection
    @property
    def spike_indice(self):
        if (not hasattr(self, '_spike_indice')) or self.option_is_updated:
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
        if not hasattr(self, '_hist_indice') or self.option_is_updated:
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
        out_dict =  dict(mean = self._obj.mean(),
                        std = self._obj.std(),
                        skew= self._obj.skew(),
                        kurt= self._obj.kurtosis(),
                        pct_null = self._obj.isna().sum()/self._obj.size,
                        stationarity_measure = self.stationarity_measure,
                        pct_spike_flag = len(self.spike_indice)/self._obj.size,
                        pct_hist_flag = len(self.hist_indice)/self._obj.size)

        return pd.Series(out_dict, name=self._obj.name)

@pd.api.extensions.register_dataframe_accessor("qc")
class QualityControlDataFrameAccessor(QualityControlAccessor):
    pass
