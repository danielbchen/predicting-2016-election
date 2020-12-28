import pandas as pd
import numpy as np


def personal_income_data_loader():
    '''
    Loads in personal income data.
    '''
    fname = 'Personal Income by State (BEA).csv'
    df = pd.read_csv(
        fname,
        skiprows=4,
        usecols=['GeoName', '2012', '2016']
    )

    df = df.copy()

    df = df[1:52]
    df['GeoName'] = df['GeoName'].str.replace('*', '')
    df['GeoName'] = [state[:-1] if state.endswith(' ') else state for state in df['GeoName']]

    df.columns = ['STATE', '2012_PERSONAL_INCOME', '2016_PERSONAL_INCOME']

    return df