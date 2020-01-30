import pandas as pd
from pprint import pprint
import pickle
from DATE_NUMBER_TO_WEEK_NUMBER_data import *

def check_for_pk(V_or_H_row):
    if V_or_H_row['Close'] == 'pk' and V_or_H_row['Open'] == 'pk':
        print('ERROR - Both Open and Close == pk')
    elif V_or_H_row['Close'] == 'pk':
        V_or_H_row['Close'] = V_or_H_row['Open']
    elif V_or_H_row['Open'] == 'pk':
        V_or_H_row['Open'] = V_or_H_row['Close']

def add_to_running_dict_df(running_dict, game):
    if len(running_dict) == 0:
        for key in game:
            running_dict[key] = []
    for key, val in game.items():
        running_dict[key].append(val)

NAME_TO_SHORTHAND = {'GreenBay': 'GB',
                     'Chicago': 'CHI',
                     'Atlanta': 'ATL',
                     'Minnesota': 'MIN',
                     'Cleveland': 'CLE',
                     'SanFrancisco': 'SF',
                     'Philadelphia': 'PHI',
                     'Pittsburgh': 'PIT',
                     'Indianapolis': 'IND',
                     'Buffalo': 'BUF',
                     'Baltimore': 'BAL',
                     'Jacksonville': 'JAX',
                     'NYGiants': 'NYG',
                     'TampaBay': 'TB',
                     'NewOrleans': 'NO',
                     'Houston': 'HOU',
                     'NewEngland': 'NE',
                     'Tennessee': 'TEN',
                     'Miami': 'MIA',
                     'KansasCity': 'KC',
                     'LAChargers': 'LAC',
                     'Seattle': 'SEA',
                     'Denver': 'DEN',
                     'Dallas': 'DAL',
                     'Carolina': 'CAR',
                     'Washington': 'WAS',
                     'Arizona': 'ARI',
                     'NYJets': 'NYJ',
                     'Detroit': 'DET',
                     'Oakland': 'OAK',
                     'LARams': 'LA',
                     'Cincinnati': 'CIN'}


DATE_NUMBER_TO_WEEK_NUMBER = DATE_NUMBER_TO_WEEK_NUMBER_2019

YEAR = 2019

data_path = 'E:\\PycharmProjects\\FootballSim\\historical_odds\\' + str(YEAR) + '.xlsx'
if YEAR == 2019:
    data_path = 'E:\\PycharmProjects\\FootballSim\\historical_odds\\' + str(YEAR) + '_w_17.xlsx'

df = pd.read_excel(data_path)
running_dict = {}
for idx in range(0, len(df), 2):
    V_row = df.loc[idx]
    H_row = df.loc[idx + 1]
    if V_row['Team'] not in NAME_TO_SHORTHAND:
        print('Team not in shorthand mapping:', V_row['Team'])
        print('v idx:', idx)
        continue
    if H_row['Team'] not in NAME_TO_SHORTHAND:
        print('Team not in shorthand mapping:', H_row['Team'])
        print('h idx:', idx)

        continue
    V_team = NAME_TO_SHORTHAND[V_row['Team']]
    H_team = NAME_TO_SHORTHAND[H_row['Team']]
    for V_or_H_row in [V_row, H_row]:
        check_for_pk(V_or_H_row)

    if V_row['Close'] < H_row['Close']:
        spread_open = V_row['Open']
        spread_close = V_row['Close']
        favorite = V_team
        over_under_open = H_row['Open']
        over_under_close = H_row['Close']
    else:
        spread_open = H_row['Open']
        spread_close = H_row['Close']
        favorite = H_team
        over_under_open = V_row['Open']
        over_under_close = V_row['Close']

    ml = (abs(V_row['ML']) + abs(H_row['ML'])) / 2

    date_key = str(int(V_row['Date']))
    if date_key not in DATE_NUMBER_TO_WEEK_NUMBER:
        print('Date key not in:', date_key)
        continue
    else:
        week = DATE_NUMBER_TO_WEEK_NUMBER[date_key]

    game = {'home': H_team, 'away': V_team, 'favorite': favorite, 'spread_open': spread_open, 'spread_close': spread_close,
            'over_under_open': over_under_open, 'over_under_close': over_under_close, 'ML': ml, 'year': YEAR, 'week': week,
            'final_real_home_score': H_row['Final'], 'final_real_away_score': V_row['Final'],
            'final_real_home_minus_away':  H_row['Final']-V_row['Final']}
    #print('Game:', game)
    add_to_running_dict_df(running_dict, game)

df_output = pd.DataFrame.from_dict(running_dict)
print('df output:')
#print(df_output)
with open('historical_odds\\compiled' + str(YEAR) + '.pkl', 'wb') as file:
    pickle.dump(df_output, file)