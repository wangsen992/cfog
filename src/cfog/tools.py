'''Tool functions making use of the package for direct applications'''

# Computation of eddy-covariance and dissipation data for a longer period of
# time. This function should take in a dataframe of u, v, w, T, q with pandas
# datetimeIndex. Thermodyanmics calculations should be completed before
# entering data into the function.Note there should also be a summary of the
# statoinarity measures along with the computation of eddy-covariance data. 
def summarize_eddyco(df, window):
    # Initiate containers for the eddyco data, qc measure and spectrum data

    # start looping
        # perform qc and double rotation

        # perform eddyco calculation

        # compute cross spectra and dissipation

    raise NotImplementedError()

# A wavelet computation function. 
def wavelet_analysis(df):
    raise NotImplementedError
