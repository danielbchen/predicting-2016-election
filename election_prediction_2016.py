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

    df['MANUFACTURING_SHARE_' + year] = df['Manufacturing'] / df['All industry total']

    df = df[['GeoName', 'MANUFACTURING_SHARE_' + year]]
    df.columns = ['STATE', 'MANUFACTURING_SHARE_' + year]

    return df


def gdp_by_sector_data_loader():
    '''
    Loads in GDP data.
    '''

    fname = 'GDP by Sector (BEA).csv'
    df = pd.read_csv(
        fname,
        skiprows=4,
        usecols=['GeoName', 'Description', '2012', '2016']
    )

    drop_regions = [
        'United States *',
        'New England',
        'Mideast',
        'Great Lakes',
        'Plains',
        'Southeast',
        'Southwest',
        'Rocky Mountain',
        'Far West'
    ]
    df = df[~df['GeoName'].isin(drop_regions)]
    df = df[df['GeoName'].notna()]

    df['Description'] = df['Description'].str.replace('    Manufacturing', 'Manufacturing')
    keep_industries = ['All industry total', 'Manufacturing']
    df = df[df['Description'].isin(keep_industries)]

    df_2012 = pivoter(df, '2012')
    df_2016 = pivoter(df, '2016')

    df = pd.merge(df_2012, df_2016, on='STATE')

    return df


def nces_names_cleaner(dataframe):
    '''
    Subsets dataframes produced by the National Center for Education
    Statistics and cleans up the names of states.
    '''

    df = dataframe.copy()

    df = df[10:70]
    df = df[df['STATE'].notna()]

    df['STATE'] = df['STATE'].str.replace('.', '')
    df['STATE'] = [state[:-4] if len(state) == 26 else state for state in df['STATE']]
    df['STATE'] = [state[:-1] if state.endswith(' ') else state for state in df['STATE']]
    df['STATE'] = [state[2:] for state in df['STATE']]

    return df


def high_school_grad_data_loader():
    '''
    Loads in the public high school graduation data.
    '''

    fname = 'High School Graduation by State (NCES).xls'
    df = pd.read_excel(
        fname,
        skiprows=2,
        usecols=['Unnamed: 0', '2011-12', '2015-16']
    )

    df.columns = ['STATE', 'HS_GRAD_2012', 'HS_GRAD_2016']

    df = nces_names_cleaner(df)

    return df


def high_school_enroll_data_loader():
    '''
    Loads in the public high school enrollment data.
    '''

    fname = 'High School Enrollment (NCES).xls'
    df = pd.read_excel(
        fname,
        skiprows=2,
        usecols=['Unnamed: 0', 'Fall 2012', 'Fall 2016']
    )

    df.columns = ['STATE', 'HS_ENROLLMENT_2012', 'HS_ENROLLMENT_2016']

    df = nces_names_cleaner(df)

    return df
