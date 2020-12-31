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


def nces_data_merger(dataframe1, dataframe2):
    '''
    Merges the files from the National Center for Education Statistics and
    returns a dataframe with HS graduation rates in 2012 and 2016.
    '''

    df = pd.merge(dataframe1, dataframe2, on='STATE')

    cols = [column for column in df.columns if column.startswith('HS')]
    df[cols] = df[cols].astype(int)

    df['HS_GRAD_RATE_2012'] = df['HS_GRAD_2012'] / df['HS_ENROLLMENT_2012']
    df['HS_GRAD_RATE_2016'] = df['HS_GRAD_2016'] / df['HS_ENROLLMENT_2016']

    df = df[['STATE', 'HS_GRAD_RATE_2012', 'HS_GRAD_RATE_2016']]

    return df


def census_cleaner(dataframe, year):
    '''
    Cleans a census data file and returns annual data on the ratio of male to
    female voters along with the ratio of white to non-white voters.

    A greater male to female ratio means that there are more males.
    A greater white to non-white ratio means that there are more white voters.
    '''

    df = dataframe.copy()

    rows_to_cols = [
        'Male',
        'Female',
        '.White non-Hispanic alone',
        'Black alone',
        'Asian alone',
        'Hispanic (of any race)'
    ]
    df = df[df['Race and Hispanic origin'].isin(rows_to_cols)]

    df['Total voted'] = df['Total voted'].replace('-', '0').astype(int)

    df = (df.pivot(index='State', columns='Race and Hispanic origin', values='Total voted')
            .reset_index())

    df = (df.rename_axis(None, axis=1)
            .rename_axis('row_num', axis=0)
            .reset_index()
            .drop('row_num', axis=1))

    df['MALE_FEMALE_RATIO'] = df['Male'] / df['Female']
    df['NON-WHITE'] = (df['Asian alone']
                       + df['Black alone']
                       + df['Hispanic (of any race)'])
    df['WHITE_NON_WHITE_RATIO'] = (df['.White non-Hispanic alone']
                                   / df['NON-WHITE'])

    df = df[['State', 'MALE_FEMALE_RATIO', 'WHITE_NON_WHITE_RATIO']]

    df.columns = [
        'STATE',
        'MALE_FEMALE_RATIO_' + year,
        'WHITE_NON_WHITE_RATIO_' + year
    ]

    return df


def census_voting_2012_loader():
    '''
    Loads in the 2012 Cencus voting data.
    '''

    fname = 'Voting Registration by Race 2012 (Census).xls'
    df = pd.read_excel(
        fname,
        skiprows=3,
        usecols=['State', 'Race and Hispanic origin', 'Total voted']
    )

    df['State'].fillna(method='ffill', inplace=True)
    df = df[11:572]

    df = census_cleaner(df, '2012')

    return df


def census_voting_2016_loader():
    '''
    Loads in the 2016 Census voting data.
    '''

    fname = 'Voting Registration by Race 2016 (Census).xlsx'
    df = pd.read_excel(
        fname,
        skiprows=3,
        usecols=['STATE', 'Sex, Race and Hispanic-Origin', 'Voted']
    )

    df = df[12:573]
    df['STATE'].fillna(method='ffill', inplace=True)

    df.columns = ['State', 'Race and Hispanic origin', 'Total voted']
    replacements = {'White non-Hispanic alone': '.White non-Hispanic alone'}
    df['Race and Hispanic origin'] = df['Race and Hispanic origin'].replace(replacements)

    df = census_cleaner(df, '2016')

    return df


def census_merger(dataframe1, dataframe2):
    '''
    Merges census data from 2012 and 2016.
    '''

    df = pd.merge(dataframe1, dataframe2, on='STATE')

    return df


def cook_pvi_loader():
    '''
    Loads in Cook Partisan Voting Index (PVI) data for 2012 and 2016.

    The Cook PVI measures how strongly a state leans towards the Democratic
    or Republican Party compared to the nation as a whole.

    Usually Cook PVI is reported as D+12 or R+5. Here D+12 would be -12 and
    R+5 would be +5. More negative values indicate a more Democratic state.
    More positive values indicate a more Republican state.
    '''

    fname = 'Cook PVI (Cook).csv'
    df = pd.read_csv(
        fname,
        skiprows=1,
        usecols=['State', 'PVI', 'Unnamed: 5', 'PVI.1', 'Unnamed: 9']
    )
    df.columns = [
        'STATE',
        'PARTY_2016',
        'PVI_SCORE_2016',
        'PARTY_2012',
        'PVI_SCORE_2012'
    ]

    df = df[df['STATE'].notna()]
    drop_state_values = [
        'Nationwide',
        'Region',
        'The Midwest',
        'The Northeast',
        'The South',
        'The West'
    ]
    df = df[~df['STATE'].isin(drop_state_values)]

    replacements = {
        'R+': 1,
        'D+': -1
    }
    cols = ['PARTY_2016', 'PARTY_2012']
    df[cols] = (df[cols].replace(replacements)
                        .astype(int))

    df['PVI_2012'] = df['PARTY_2012'] * df['PVI_SCORE_2012']
    df['PVI_2016'] = df['PARTY_2016'] * df['PVI_SCORE_2016']

    df = df[['STATE', 'PVI_2012', 'PVI_2016']]

    df['STATE'] = df['STATE'].replace('Washington DC', 'District of Columbia')

    return df


def winner_calculator(dataframe, year):
    '''
    Calculates the winning party in all states in a given year.
    '''

    df = dataframe.copy()

    df = df[df['year'] == year]
    df = (df.pivot_table(index='state', columns='party', values='candidatevotes')
            .reset_index())

    df = (df.rename_axis(None, axis=1)
            .rename_axis('row_num', axis=0)
            .reset_index()
            .drop('row_num', axis=1))

    df['WINNER'] = np.where(df['democrat'] > df['republican'],
                            'Democrat',
                            'Republican')

    df = df[['state', 'WINNER']]
    df.columns = ['STATE', 'WINNER_' + year]

    return df