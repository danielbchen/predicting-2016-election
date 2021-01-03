# Predicting the 2016 United States Presidential Election with 2012 Data

### 1. Overview 

I use a Random Forest Classifier to predict who a state voted for in the 2016 United States presidental election based on state level data from 2012. In this model, the label predicted will not be Hillary Clinton nor Donald Trump, instead the label predicted will be the candidate's party. 

### 2. Understanding the Features 

For this supervised machine learning model, I retreive the following statewide data from a variety of sources:

1. [Personal Income Data](https://github.com/danielbchen/predicting-2016-election/blob/main/Personal%20Income%20by%20State%20(BEA).csv) from the [Bureau of Economic Analysis](https://www.bea.gov/data/income-saving/personal-income). Various studies have tied economic status to political party identification, and I hypothesize that states with larger incomes will be more likely to vote for a Republican candidate. 
