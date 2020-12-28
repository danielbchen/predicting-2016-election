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


def pivoter(dataframe, year):
    '''
    Pivots GDP data and creates new column reporting manufacturing in a given
    year in a given state as a share of its total GDP for that year.
    '''

    df = dataframe.copy()

    df = (df
          .pivot(index='GeoName', columns='Description', values=year)
          .reset_index())

    df = (df
          .rename_axis(None, axis=1)
          .rename_axis('row_num', axis=0)
          .reset_index())

    cols = ['Manufacturing', 'All industry total']
    df[cols] = df[cols].astype(float)

    df['MANUFACTURING_SHARE_' + year] = df['Manufacturing'] / \
        df['All industry total']

    df = df[['GeoName', 'MANUFACTURING_SHARE_' + year]]
    df.columns = ['STATE', 'MANUFACTURING_SHARE_' + year]

    return df
