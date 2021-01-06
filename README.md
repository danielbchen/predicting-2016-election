# Predicting the 2016 United States Presidential Election with 2012 Data

### 1. Overview 

I use a Random Forest Classifier to predict who a state voted for in the 2016 United States presidental election based on state level data from 2012. In this model, the label predicted will not be Hillary Clinton nor Donald Trump, instead the label predicted will be the candidate's party. 

### 2. Understanding the Features 

For this supervised machine learning model, I retreive the following statewide data from a variety of sources:

1. [Personal Income Data - Personal Income by State (BEA).csv](https://github.com/danielbchen/predicting-2016-election/blob/main/Personal%20Income%20by%20State%20(BEA).csv) from the [Bureau of Economic Analysis](https://www.bea.gov/data/income-saving/personal-income). Various studies, such as [Powdthavee and Oswald (2014)](http://www.andrewoswald.com/docs/SentVotingLottery12014PowdthaveeOs.pdf) and [Doherty, Gerber, and Green (2006)](https://www.jstor.org/stable/3792457?seq=1) have tied economic status to political party identification, and I hypothesize that states with larger incomes will be more likely to vote for a Republican candidate. Moreover, based on Republican platform rhetoric of small government and fewer taxes, it is in the best interest of higher income earners to vote for a GOP candidate who will keep more money in thier pockets. 

2. [State GDP in Manufacturing Sector as a Share of Total GDP - GDP by Sector (BEA).csv](https://github.com/danielbchen/predicting-2016-election/blob/main/GDP%20by%20Sector%20(BEA).csv) from the [Bureau of Economic Analysis](https://apps.bea.gov/iTable/iTable.cfm?reqid=70&step=1&isuri=1). With the rise of China as a global economic power in conjunction with fears of immigrants "stealing jobs" from Americans, Republicans have focused on American industry to revitalize the economy. With [The Washington Post](https://www.washingtonpost.com/business/2020/08/26/republicans-favor-industrial-policy/) reporting on Republican enthusiasm for government intervention in the industrial sector and an [NPR breaking down votes in the 2016 election](https://www.npr.org/2016/11/12/501848636/7-reasons-donald-trump-won-the-presidential-election), the value of a state's manufacturing sector may predict which party the state favors. Indeed, in 2016, those without a college degree in the industrial north broke for Trump as the Republican party made double digit gains from 2012 to 2016.

3. Public High School Graduation Rates derived from [High School Graduation by State (NCES).xls](https://github.com/danielbchen/predicting-2016-election/blob/main/High%20School%20Graduation%20by%20State%20(NCES).xls) and [High School Enrollment (NCES).xls](https://github.com/danielbchen/predicting-2016-election/blob/main/High%20School%20Enrollment%20(NCES).xls) provided by [the National Center for Education Statistics](https://nces.ed.gov/programs/digest/d16/tables/dt16_219.20.asp). Lower education voters shifted from Obama in 2012 to Trump in 2016. [FiveThirtyEight](https://fivethirtyeight.com/features/education-not-income-predicted-who-would-vote-for-trump/) provides a thorough breakdown of education and income in counties that voted for Clinton and those that voted for Trump.

4. Ratio of Male to Female Voters who Actually Voted in [2012 (Voting Registration by Demographics 2012 (Census).xls)](https://github.com/danielbchen/predicting-2016-election/blob/main/Voting%20Registration%20by%20Demographics%202012%20(Census).xls) and [2016 (Voting Registration by Demographics 2016 (Census).xlsx)](https://github.com/danielbchen/predicting-2016-election/blob/main/Voting%20Registration%20by%20Demographics%202016%20(Census).xlsx) from [the U.S. Census](https://www.census.gov/topics/public-sector/voting/data/tables.html). According to [Pew Research](https://www.pewresearch.org/fact-tank/2020/08/18/men-and-women-in-the-u-s-continue-to-differ-in-voter-turnout-rate-party-identification/), women -- on the whole -- are more likely to identify as Democrats or lean Democratic. The opposite is true for men. A higher ratio of male to female voters in a state may be indicative of a Republican tilt. Though, of course I have the luxury of writing in 2021, so we know that pluralities of white women have supported Trump in both elections.  

5. Ratio of White to Non-White Voters who Actually Voted in [2012 (Voting Registration by Demographics 2012 (Census).xls)](https://github.com/danielbchen/predicting-2016-election/blob/main/Voting%20Registration%20by%20Demographics%202012%20(Census).xls) and [2016 (Voting Registration by Demographics 2016 (Census).xlsx)](https://github.com/danielbchen/predicting-2016-election/blob/main/Voting%20Registration%20by%20Demographics%202016%20(Census).xlsx) from [the U.S. Census](https://www.census.gov/topics/public-sector/voting/data/tables.html). [Pew Research](https://www.pewresearch.org/politics/2018/03/20/1-trends-in-party-affiliation-among-demographic-groups/) reveals that non-white voters are overwhelmingly Democratic. A lower ratio of white to non-white voters suggests that a state may vote for the Democrat. Again, because I am writing four years after the election, it's worth highlighting that [exit polls](https://www.brookings.edu/research/2020-exit-polls-show-a-scrambling-of-democrats-and-republicans-traditional-bases/) have shown that the Democratic margin among Black, Latino or Hispanic, and Asian voters slightly diminished from 2016 to 2020. 

6. [Cook Partisan Voter Index Score - Cook PVI (Cook).csv](https://github.com/danielbchen/predicting-2016-election/blob/main/Cook%20PVI%20(Cook).csv) from [The Cook Political Report](https://docs.google.com/spreadsheets/d/1D-edaVHTnZNhVU840EPUhz3Cgd7m39Urx7HM8Pq6Pus/edit#gid=29622862) The Cook PVI score measures how Democratic or Republican a state is relative to the entire country. A higher score, such as R+5 (Republican +5) is likely a strong indication that the state will vote for the Republican candidate in the presidential election).

### Model Predictions

The [script](https://github.com/danielbchen/predicting-2016-election/blob/main/main.py) loads, cleans, and produces a test harness to determine the optimal algorithm to use based on accuracy. It ultimately returns results from the test harness, a dataframe containing the incorrect predictions, and saves two maps showing the predicted winners from 2016 and the actual winners from 2016. 

Using a Random Forest Classifier, the model predicts the final 2016 electoral map to look like the below:

![Map of Election Predictions](https://github.com/danielbchen/predicting-2016-election/blob/main/Predicted%202016%20U.S.%20Presidential%20Election%20Results.png)

The actual map is shown below:

![Actual Map of Election Outcomes](https://github.com/danielbchen/predicting-2016-election/blob/main/2016%20U.S.%20Presidential%20Election%20Results.png)

Arizona, Florida, Minnesota, Oregon, and Pennsylvania were predicted incorrectly. In terms of accuracy, 45 out of 50 states were predicted correctly for an accuracy score of 90%. 

### Model Discussion

The model was early in predicting a blue Arizona and a blue Pennsylvania. Though they went red in 2016 before flipping four years down the road. The model's incorrect prediction isn't surprising. Pennsylvania is a battleground state after all, and strategists and pollsters have been optimistic of the turning tide in Arizona. 

Out of context 90% may appear impressive. However, this would assume that each state is weighted equal on the path to the Oval Office. This is not true. States such as Pennsylvania and Florida are always under watch because they are purple states. Moreover, they carry the pair of states carries 20 and 29 electoral votes respectively. An incorrect forecast in these places can not only cost a candidate/party a total of 49 electoral votes, but more importantly, it can cost them the path to the executive branch.
