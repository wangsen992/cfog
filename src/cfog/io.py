# -*- coding: utf-8 -*-
"""
Functions to facilitate all sorts of data loading and saving from files to
memory and visa versa.
"""

import os
import pickle
import numpy as np
import pandas as pd
from google.cloud import bigquery
from .data import TimeSeries

table_names = ['buoy_MTE','buoy_SJB','sharp_ceilometer','sharp_aps',
               'sharp_fast','sharp_opc','sharp_platform',
               'sharp_pwd','sharp_rosr','sharp_slow','sharp_smps',
               'sharp_sms','sharp_sonde']

# Functions for loading data from gcp bigquery 
def get_sharp_data(client, start, end, date_part=None):
    """Wrapper to get all of sharp data using functions below"""

    print("Not loading fast/slow/platform data.")    
    data_out = {}
    data_out['sharp_sonde'] = get_sondes(client, start, end)
    data_out['sharp_ceilometer'] = get_ceilometer(client, start, end)
    data_out['sharp_opc'] = get_opc(client, start, end)
    data_out['sharp_aps'] = get_aps(client, start, end)
    data_out['sharp_smps'] = get_smps(client, start, end)

    tnames = ['buoy_MTE', 'buoy_SJB', 'sharp_pwd', 'sharp_rosr',
              'sharp_sms']
    for table_id in tnames:
        data_out[table_id] = get_table(client, start, end, table_id)

#    for table_id in ['sharp_fast', 'sharp_slow', 'sharp_platform']:
#        data_out[table_id] = get_table(client, start, end, table_id, date_part)

    return data_out

def get_sondes(client, start, end):
    """Getting radiosonde data from bigquery client

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]

    Returns:
      dict  :  a dictionary of radiosonde data, with key as "mm-dd-HH",
               and the value being another dictionary of the radiosonde data
               with the actual measurements in a dataframe.
    """

    sonde_query_str = "SELECT * FROM cfog.sharp_radiosonde " + \
                  f"WHERE LaunchTime BETWEEN '{start}' AND '{end}' " + \
                  "ORDER BY LaunchTime ASC"

    print(f"Executing bigquery query string: ")
    print(sonde_query_str + '\n')

    sonde_data = {f"{s['LaunchTime'].strftime('%m-%d_%H')}":s for s in client.query(query=sonde_query_str)}

    print("Radiosondes obtained within the queried time bounds: ")
    print(list(sonde_data))

    sonde_data_out = {}
    for t in sonde_data:
        # ignored col: SoundingIdPk, RadioRxTimePk, PtuStatus
        sonde_data_out[t] = {}
        sonde_data_out[t]['df'] = pd.DataFrame({
            'DataSrvTime'       : sonde_data[t]['DataSrvTime'],
            'Pressure'          : sonde_data[t]['Pressure'],
            'Temperature'       : sonde_data[t]['Temperature'],
            'Humidity'          : sonde_data[t]['Humidity'],
            'WindDir'           : sonde_data[t]['WindDir'],
            'WindSpeed'         : sonde_data[t]['WindSpeed'],
            'WindNorth'         : sonde_data[t]['WindNorth'],
            'WindEast'          : sonde_data[t]['WindEast'],
            'Height'            : sonde_data[t]['Height'],
            'WindInterpolated'  : sonde_data[t]['WindInterpolated'],
            'Latitude'          : sonde_data[t]['Latitude'],
            'Longitude'        : sonde_data[t]['Longitude'],
            'North'             : sonde_data[t]['North'],
            'East'              : sonde_data[t]['East'],
            'Up'                : sonde_data[t]['Up'],
            'Altitude'          : sonde_data[t]['Altitude'],
            'Dropping'          : sonde_data[t]['Dropping']
        }
        )
        sonde_data_out[t]['LaunchTime'] = sonde_data[t]['LaunchTime']
        sonde_data_out[t]['LaunchLatitude'] = sonde_data[t]['LaunchLatitude']
        sonde_data_out[t]['LaunchLongitude'] = sonde_data[t]['LaunchLongitude']

    print(f"Query complete. Total number of data entries: {len(sonde_data_out)}.\n\n")

    del sonde_data
    return sonde_data_out

def get_ceilometer(client, start, end):
    """Getting ceilometer data from bigquery client

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]

    Returns:
      dict  :  a dictionary of ceilometer data, with keys includings,
                * backscatter: np.array of backscatter profile (heights x time)
                               with sensitivity normalized units
                               (100000·srad·km)-1 unless otherwise scaled with
                               the SCALE parameter.
                * heights : np.array of heights calculated from resolution and
                            num_gates
                * resolution
                * num_gates
                * df : dataframe of other the other data
    """
    # load image or load from bigquery
    ceil_query_str = "SELECT * FROM cfog.sharp_ceilometer " +\
                    f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                     "ORDER BY timestamp ASC"
    print(f"Executing bigquery query string: ")
    print(ceil_query_str + '\n')

    ceil_query_job = client.query(ceil_query_str)
    ceil_query_job.result()
    ceil_data = ceil_query_job.to_dataframe()

    # Check consistency of resolution and num_gates
    if ceil_data['resolution'].unique().size == 1 and ceil_data['num_gates'].unique().size==1:
        print("Consistency check on resolution and num_gates passed.")
        resolution = ceil_data['resolution'].unique()[0]
        num_gates = ceil_data['num_gates'].unique()[0]
    else:
        raise ValueError("Resolutions and num_gates are not consistent")

    scatter_arr = np.array(ceil_data['backscatter_profile'].values.tolist()).T
    ceil_data_df = ceil_data.drop(columns=['backscatter_profile']).set_index('timestamp')
    heights = np.arange(10, 10+resolution*num_gates, resolution)
    ceil_data_out = dict(backscatter = scatter_arr,
                         heights=heights,
                         df=ceil_data_df,
                         resolution = resolution,
                         num_gates = num_gates)

    print(f"Query complete. Total number of data entries: {ceil_data_out['df'].shape[0]}.\n\n")
    return ceil_data_out

def get_opc(client, start, end):
    """Getting ceilometer data from bigquery client

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]

    Returns:
      dict  :  a dictionary of opc data, with keys includings,
                * spectra: np.array of spectra profile (units?)
                * binsize : np.array of heights calculated from resolution and
                            num_gates
                * df : dataframe of other the other data
    """
    # load image or load from bigquery
    opc_query_str = "SELECT * FROM cfog.sharp_OPC " +\
                    f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                     "ORDER BY timestamp ASC"
    print(f"Executing bigquery query string: ")
    print(opc_query_str + '\n')

    opc_query_job = client.query(opc_query_str)
    opc_query_job.result()
    opc_data = opc_query_job.to_dataframe()

    spectra_arr = np.array(opc_data['spectra'].values.tolist()).T
    opc_data_df = opc_data.drop(columns=['spectra']).set_index('timestamp')
    binsize = np.array([0.46010524556604593,0.6606824566769,0.91491746243386907,
                       1.195215726298366,1.4649081758117393,1.8300250375885727,
                       2.5350321387248442,3.4999845389695112,4.50000193575099,
                       5.7499993072082258,7.2499995196838025,8.9999960119985154,
                       11.000000156148959,13.000001845860073,15.000000374490131,
                       16.7500010443006])
    opc_data_out = dict(spectra = spectra_arr,
                         binsize=binsize,
                         df=opc_data_df)

    print(f"Query complete. Total number of data entries: {opc_data_out['df'].shape[0]}.\n\n")
    return opc_data_out

def get_aps(client, start, end):
    """Getting ceilometer data from bigquery client

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]

    Returns:
      dict  :  a dictionary of aps data, with keys includings,
                * values: np.array of spectra profile (units?)
                * lowBouDia: np.array of measurement bounds
                * highBouDia: np.array of measurement bounds
                * midDia: np.array of median diameters
                * df : dataframe of other the other data
    """
    # load image or load from bigquery
    aps_query_str = "SELECT * FROM cfog.sharp_aps " +\
                    f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                     "ORDER BY timestamp ASC"
    print(f"Executing bigquery query string: ")
    print(aps_query_str + '\n')

    aps_query_job = client.query(aps_query_str)
    aps_query_job.result()
    aps_data = aps_query_job.to_dataframe()

    values = np.array(aps_data['values'].values.tolist()).T
    lowBouDia = np.array(aps_data['lowBouDia'].values.tolist()).T
    highBouDia = np.array(aps_data['highBouDia'].values.tolist()).T
    midDia = np.array(aps_data['midDia'].values.tolist()).T
    aps_data_df = aps_data.drop(columns=['values','lowBouDia','highBouDia','midDia']).set_index('timestamp')
    aps_data_out = dict(values=values,
                        lowBouDia=lowBouDia,
                        highBouDia=highBouDia,
                        midDia=midDia,
                        df=aps_data_df)

    print(f"Query complete. Total number of data entries: {aps_data_out['df'].shape[0]}.\n\n")
    return aps_data_out

def get_smps(client, start, end):
    """Getting ceilometer data from bigquery client

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]

    Returns:
      dict  :  a dictionary of smps data, with keys includings,
                * values: np.array of spectra profile (units?)
                * lowBouDia: np.array of measurement bounds
                * highBouDia: np.array of measurement bounds
                * midDia: np.array of median diameters
                * df : dataframe of other the other data
    """
    # load image or load from bigquery
    smps_query_str = "SELECT * FROM cfog.sharp_smps " +\
                    f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                     "ORDER BY timestamp ASC"
    print(f"Executing bigquery query string: ")
    print(smps_query_str + '\n')

    smps_query_job = client.query(smps_query_str)
    smps_query_job.result()
    smps_data = smps_query_job.to_dataframe()

    values = np.array(smps_data['values'].values.tolist()).T
    lowBouDia = np.array(smps_data['lowBouDia'].values.tolist()).T
    highBouDia = np.array(smps_data['highBouDia'].values.tolist()).T
    midDia = np.array(smps_data['midDia'].values.tolist()).T
    smps_data_df = smps_data.drop(columns=['values','lowBouDia','highBouDia','midDia']).set_index('timestamp')
    smps_data_out = dict(values=values,
                        lowBouDia=lowBouDia,
                        highBouDia=highBouDia,
                        midDia=midDia,
                        df=smps_data_df)

    print(f"Query complete. Total number of data entries: {smps_data_out['df'].shape[0]}.\n\n")
    return smps_data_out

def get_table(client,start,end,table_id,date_part = None):
    """Getting table data from bigquery client.

    Args:
      client (bigquery.Client) : Configured client to access bigquery
      start (str)  :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      end (str)    :  time str in the format of yy-mm:dd [HH-MM-SS.FFFFFF]
      table_id (str)  :  table name of any bigquery table without array column.
      date_part (str) : date_part param for SQL TIMESTAMP_TRUNC() function.
    Returns:
      pd.DataFrame : with index being the timestamp of the data
    """
    # enable the ability to obtain averaged data.
    if date_part is None:
        table_query_str = f"SELECT * FROM cfog.{table_id} " +\
                          f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                           "ORDER BY timestamp ASC"
    else:
        # first obtain a list of field names
        table_ref = client.dataset('cfog').table(table_id)
        table = client.get_table(table_ref)
        schemas = [s for s in table.schema if s.field_type in ['INT', 'FLOAT']]
        field_names = [s.name for s in schemas]
        field_name_strs = ','.join([f"AVG({name}) as {name}" for name  in field_names])
        trunc_exp = f"TIMESTAMP_TRUNC(timestamp, {date_part}) AS timestamp"
        table_query_str = f"SELECT {trunc_exp}, {field_name_strs} FROM cfog.{table_id} " +\
                          f"WHERE timestamp BETWEEN '{start}' AND '{end}' " +\
                           "GROUP BY timestamp ORDER BY timestamp"

    print(f"Executing bigquery query string: ")
    print(table_query_str + '\n')

    table_query_job = client.query(table_query_str)
    table_query_job.result()
    print("Query job complete. Start Loading Data. ")
    table_data = table_query_job.to_dataframe().set_index('timestamp')
    print(f"Query complete. Total number of data entries: {table_data.shape[0]}.\n\n")

    return table_data

