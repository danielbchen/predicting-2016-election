import numpy as np
import os
import pandas as pd
import plotly.express as px  # pip install plotly==4.14.1
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import us  # pip install us


def main():
    """Combines data from various sources into a dataframe and uses a
    Random Forest to predict the winner of the 2016 election by party
    by state based on the 2012 data.

    Prints out the following:
    1) A dataframe containing results from a test harness.
    2) Statements regarding which machine algorithm to use.
    3) The accuracy score of using a Random Forest.
    4) The states where the model predicted the 2016 winner incorrectly.
    """

    census_2012 = census_voting_2012_loader()
    census_2016 = census_voting_2016_loader()
    hs_enrollment = high_school_enroll_data_loader()
    hs_grad = high_school_grad_data_loader()

    income = personal_income_data_loader()
    manufacturing_gdp = gdp_by_sector_data_loader()
    hs_edu = nces_data_merger(hs_enrollment, hs_grad)
    cook_pvi = cook_pvi_loader()
    election_winners = harvard_election_winner_loader()
    census_merged = census_merger(census_2012, census_2016)

    df = all_data_merger(income, 
                         manufacturing_gdp, 
                         hs_edu, cook_pvi, 
                         election_winners, 
                         census_merged)

    harness_df = test_harness(df)

    print('Test Harness Results:', harness_df, sep='\n\n')
    print('\n')
    algorithm_decider(df)
    print('\n')

    predictions_df = random_forest_modeler(df)
    mapper(predictions_df, '2016_PREDICTED_WINNER',
           'Predicted 2016 U.S. Presidential Election Results')
    mapper(predictions_df, 'WINNER_2016',
           '2016 U.S. Presidential Election Results')
    print('Maps showing actual vs. predicted presidential election results '
          'have been saved.')
    print('\n')

    results_summarizer(predictions_df)


def personal_income_data_loader():
    """Loads in personal income data."""
    
    fname = 'Personal Income by State (BEA).csv'
    df = pd.read_csv(fname,
                     skiprows=4,
                     usecols=['GeoName', '2012', '2016'])

    df = df.copy()

    df = df[1:52]
    df['GeoName'] = df['GeoName'].str.replace('*', '')
    df['GeoName'] = [state[:-1] if state.endswith(' ') else state for state in df['GeoName']]

    df.columns = ['STATE', '2012_PERSONAL_INCOME', '2016_PERSONAL_INCOME']

    return df


def pivoter(dataframe, year):
    """Pivots GDP data and creates new column reporting manufacturing in a given
    year in a given state as a share of its total GDP for that year.
    """

    df = dataframe.copy()

    df = (df.pivot(index='GeoName', columns='Description', values=year)
            .reset_index())

    df = (df.rename_axis(None, axis=1)
            .rename_axis('row_num', axis=0)
            .reset_index())

    cols = ['Manufacturing', 'All industry total']
    df[cols] = df[cols].astype(float)

    df['MANUFACTURING_SHARE_' + year] = df['Manufacturing'] / df['All industry total']

    df = df[['GeoName', 'MANUFACTURING_SHARE_' + year]]
    df.columns = ['STATE', 'MANUFACTURING_SHARE_' + year]

    return df


def gdp_by_sector_data_loader():
    """Loads in GDP data."""

    fname = 'GDP by Sector (BEA).csv'
    df = pd.read_csv(fname,
                     skiprows=4,
                     usecols=['GeoName', 'Description', '2012', '2016'])

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
    """Subsets dataframes produced by the National Center for Education
    Statistics and cleans up the names of states.
    """

    df = dataframe.copy()

    df = df[10:70]
    df = df[df['STATE'].notna()]

    df['STATE'] = df['STATE'].str.replace('.', '')
    df['STATE'] = [state[:-4] if len(state) == 26 else state for state in df['STATE']]
    df['STATE'] = [state[:-1] if state.endswith(' ') else state for state in df['STATE']]
    df['STATE'] = [state[2:] for state in df['STATE']]

    return df


def high_school_grad_data_loader():
    """Loads in the public high school graduation data."""

    fname = 'High School Graduation by State (NCES).xls'
    df = pd.read_excel(fname,
                       skiprows=2,
                       usecols=['Unnamed: 0', '2011-12', '2015-16'])

    df.columns = ['STATE', 'HS_GRAD_2012', 'HS_GRAD_2016']

    df = nces_names_cleaner(df)

    return df


def high_school_enroll_data_loader():
    """Loads in the public high school enrollment data."""

    fname = 'High School Enrollment (NCES).xls'
    df = pd.read_excel(fname,
                       skiprows=2,
                       usecols=['Unnamed: 0', 'Fall 2012', 'Fall 2016'])

    df.columns = ['STATE', 'HS_ENROLLMENT_2012', 'HS_ENROLLMENT_2016']

    df = nces_names_cleaner(df)

    return df


def nces_data_merger(dataframe1, dataframe2):
    """Merges the files from the National Center for Education Statistics and
    returns a dataframe with HS graduation rates in 2012 and 2016.
    """

    df = pd.merge(dataframe1, dataframe2, on='STATE')

    cols = [column for column in df.columns if column.startswith('HS')]
    df[cols] = df[cols].astype(int)

    df['HS_GRAD_RATE_2012'] = df['HS_GRAD_2012'] / df['HS_ENROLLMENT_2012']
    df['HS_GRAD_RATE_2016'] = df['HS_GRAD_2016'] / df['HS_ENROLLMENT_2016']

    df = df[['STATE', 'HS_GRAD_RATE_2012', 'HS_GRAD_RATE_2016']]

    return df


def census_cleaner(dataframe, year):
    """Cleans a census data file and returns annual data on the ratio of male 
    to female voters along with the ratio of white to non-white voters.

    A greater male to female ratio means that there are more males.
    A greater white to non-white ratio means that there are more white voters.
    """

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
    df['WHITE_NON_WHITE_RATIO'] = df['.White non-Hispanic alone'] / df['NON-WHITE']

    df = df[['State', 'MALE_FEMALE_RATIO', 'WHITE_NON_WHITE_RATIO']]

    df.columns = [
        'STATE',
        'MALE_FEMALE_RATIO_' + year,
        'WHITE_NON_WHITE_RATIO_' + year
    ]

    return df


def census_voting_2012_loader():
    """Loads in the 2012 Cencus voting data."""

    fname = 'Voting Registration by Demographics 2012 (Census).xls'
    df = pd.read_excel(fname,
                       skiprows=3,
                       usecols=['State', 'Race and Hispanic origin', 'Total voted'])

    df['State'].fillna(method='ffill', inplace=True)
    df = df[11:572]

    df = census_cleaner(df, '2012')

    return df


def census_voting_2016_loader():
    """Loads in the 2016 Census voting data."""

    fname = 'Voting Registration by Demographics 2016 (Census).xlsx'
    df = pd.read_excel(fname,
                       skiprows=3,
                       usecols=['STATE', 'Sex, Race and Hispanic-Origin', 'Voted'])

    df = df[12:573]
    df['STATE'].fillna(method='ffill', inplace=True)

    df.columns = ['State', 'Race and Hispanic origin', 'Total voted']
    replacements = {'White non-Hispanic alone': '.White non-Hispanic alone'}
    df['Race and Hispanic origin'] = df['Race and Hispanic origin'].replace(replacements)

    df = census_cleaner(df, '2016')

    return df


def census_merger(dataframe1, dataframe2):
    """Merges census data from 2012 and 2016."""

    df = pd.merge(dataframe1, dataframe2, on='STATE')

    return df


def cook_pvi_loader():
    """ Loads in Cook Partisan Voting Index (PVI) data for 2012 and 2016.

    The Cook PVI measures how strongly a state leans towards the Democratic
    or Republican Party compared to the nation as a whole.

    Usually Cook PVI is reported as D+12 or R+5. Here D+12 would be -12 and
    R+5 would be +5. More negative values indicate a more Democratic state.
    More positive values indicate a more Republican state.
    """

    fname = 'Cook PVI (Cook).csv'
    df = pd.read_csv(fname,
                     skiprows=1,
                     usecols=['State', 'PVI', 'Unnamed: 5', 'PVI.1', 'Unnamed: 9'])
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
    df[cols] = df[cols].replace(replacements).astype(int)

    df['PVI_2012'] = df['PARTY_2012'] * df['PVI_SCORE_2012']
    df['PVI_2016'] = df['PARTY_2016'] * df['PVI_SCORE_2016']

    df = df[['STATE', 'PVI_2012', 'PVI_2016']]

    df['STATE'] = df['STATE'].replace('Washington DC', 'District of Columbia')

    return df


def winner_calculator(dataframe, year):
    """Calculates the winning party in all states in a given year."""

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


def harvard_election_winner_loader():
    """ Loads in election winners by state in the 2012 and 2016 U.S. 
    Presidential Elections.
    """

    fname = '1976-2016-president.csv'
    df = pd.read_csv(fname,
                     usecols=['year', 'state', 'party', 'candidatevotes'])

    df = df[(df['year'] == 2012) | (df['year'] == 2016)]
    df = df[(df['party'] == 'republican') | (df['party'] == 'democrat')]

    df['year'] = df['year'].astype(str)

    winners_2012 = winner_calculator(df, '2012')
    winners_2016 = winner_calculator(df, '2016')

    df = pd.merge(winners_2012, winners_2016, on='STATE')

    return df


def all_data_merger(df1, df2, df3, df4, df5, df6):
    """ Merges all datasets together. Then returns a binary column with rows
    equal to 1 if the Democrat is the winner in a given state in a given
    year, and a 0 if the Republican is the victor.
    """

    df = (df1.merge(df2, on='STATE')
             .merge(df3, on='STATE')
             .merge(df4, on='STATE')
             .merge(df5, on='STATE'))

    df['STATE'] = [state.upper() for state in df['STATE']]

    df = pd.merge(df, df6, on='STATE')

    df['WINNER_2012_BINARY'] = [1 if winner == 'Democrat' else 0 for winner in df['WINNER_2012']]
    df['WINNER_2016_BINARY'] = [1 if winner == 'Democrat' else 0 for winner in df['WINNER_2016']]

    return df


def split_data(dataframe, year):
    """Splits the election data into a training set or testing set."""

    df = dataframe.copy()

    columns = []
    for column in df.columns:
        if year in column or column == 'STATE':
            columns.append(column)

    df = df[columns]

    return df


def test_harness(dataframe):
    """Takes a dataframe, splits it into a training dataset, and returns a new
    dataframe summarizing the performance of different machine learning models
    on the dataframe.
    """

    df = dataframe.copy()

    training_data = split_data(df, '2012')
    x_train = training_data.drop(['STATE', 'WINNER_2012', 'WINNER_2012_BINARY'], 1)
    y_train = training_data['WINNER_2012_BINARY']

    algorithms = []
    algorithms.append(('Decision Tree', DecisionTreeClassifier()))
    algorithms.append(('Linear Discriminant', LinearDiscriminantAnalysis()))
    algorithms.append(('Support Vector Classifier', SVC(gamma='auto')))
    algorithms.append(('Naive Bayes', GaussianNB()))
    algorithms.append(('Logistic Regression', LogisticRegression(max_iter=1000)))
    algorithms.append(('K-Nearest Neighbor', KNeighborsClassifier()))
    algorithms.append(('Random Forest', RandomForestClassifier()))

    measures = ['accuracy', 'recall', 'precision', 'f1']

    test_harness_results = pd.DataFrame(columns=['ALGORITHM',
                                                 'ACCURACY',
                                                 'STANDARD DEVIATION',
                                                 'PRECISION',
                                                 'RECALL',
                                                 'F1'])

    for name, algorithm in algorithms:
        results = cross_validate(algorithm, x_train, y_train, cv=10, scoring=measures)
        test_harness_results = test_harness_results.append(
            {
                'ALGORITHM': name,
                'ACCURACY': results['test_accuracy'].mean(),
                'STANDARD DEVIATION': results['test_accuracy'].std(),
                'PRECISION': results['test_precision'].mean(),
                'RECALL': results['test_recall'].mean(),
                'F1': results['test_f1'].mean()
            },
            ignore_index=True
        )

    return test_harness_results


def algorithm_decider(dataframe):
    """Prints four sentences that report the optimal ML model to use given
    a certain metric.
    """

    df = dataframe.copy()

    df = test_harness(df)

    optimal_accuracy = df[df['ACCURACY'] == df['ACCURACY'].max()]
    optimal_precision = df[df['PRECISION'] == df['PRECISION'].max()]
    optimal_recall = df[df['RECALL'] == df['RECALL'].max()]
    optimal_F1 = df[df['F1'] == df['F1'].max()]

    print('Optimal model(s) for ACCURACY:', 
          *optimal_accuracy['ALGORITHM'].values)
    print('Optimal model(s) for PRECISION:', 
          ', '.join(optimal_precision['ALGORITHM'].values))
    print('Optimal model(s) for RECALL:',
          ', '.join(optimal_recall['ALGORITHM'].values))
    print('Optimal model(s) for F1:', *optimal_F1['ALGORITHM'].values)


def random_forest_modeler(dataframe):
    '''
    Runs a Random Forest algorithm on the data. Returns the model's accuracy,
    and the incorrect predictions.
    '''

    df = dataframe.copy()

    training_data = split_data(df, '2012')
    test_data = split_data(df, '2016')

    x_train = training_data.drop(['STATE', 'WINNER_2012', 'WINNER_2012_BINARY'], 1)
    y_train = training_data['WINNER_2012']

    x_test = test_data.drop(['STATE', 'WINNER_2016', 'WINNER_2016_BINARY'], 1)
    y_test = test_data['WINNER_2016']

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    df['2016_PREDICTED_WINNER'] = predict

    abbr_state_xwalk = us.states.mapping('name', 'abbr')
    abbr_state_xwalk = {state.upper(): abbreviation for state,
                        abbreviation in abbr_state_xwalk.items()}
    df['ABBREVIATION'] = df['STATE'].replace(abbr_state_xwalk)

    return df


def mapper(dataframe, election_year, plot_title):
    """Base function to map election results."""

    df = dataframe.copy()

    fig = px.choropleth(df,
                        locations=df['ABBREVIATION'],
                        locationmode='USA-states',
                        scope='usa',
                        color=election_year,
                        labels={election_year: 'WINNER'},
                        color_discrete_map={
                            'Democrat'  : '#0015BC',
                            'Republican': '#FF0000'},
                        title=plot_title)

    fig.write_image('{}.png'.format(plot_title))


def results_summarizer(dataframe):
    """Prints out statements summarizing how the model performed."""

    df = dataframe.copy()

    incorrect_predicitions = df[df['WINNER_2016'] != df['2016_PREDICTED_WINNER']]

    accuracy = 1 - (len(incorrect_predicitions) / len(predictions_df))
    accuracy = round(accuracy * 100, 2)

    print('Based on data from 2012, the Random Forest algorithm predicts the '
          'winners of the 2016 Presidential Election with an accuracy score of',
          '{}%'.format(accuracy))
    print('\n')
    print('The following states were predicted incorrectly:',
          incorrect_predicitions[['STATE', 'WINNER_2016', '2016_PREDICTED_WINNER']],
          sep='\n\n')


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath("__file__"))
    main()
