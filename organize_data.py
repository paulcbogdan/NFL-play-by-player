import pandas as pd
from utils import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
from random import random
from statsmodels.sandbox.distributions.extras import pdf_mvsk
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import datetime
from copy import deepcopy
from time import time
from DATE_NUMBER_TO_WEEK_NUMBER_data import *


DATE_NUMBER_TO_WEEK_NUMBER = {2019: DATE_NUMBER_TO_WEEK_NUMBER_2019,
                              2018: DATE_NUMBER_TO_WEEK_NUMBER_2018}

class Game:
    def __init__(self, home_team, away_team):
        self.home = home_team
        self.away = away_team
        self.plays = {}

    def is_same_game(self, row):
        if row['home_team'] == self.home and row['away_team'] == self.away:
            return True
        return False

    def process_row(self, row, next_row=None):
        def add_half_details():
            if row['game_half'] == 'Half1':
                play['half'] = 1
                play['overtime'] = 0
            elif row['game_half'] == 'Half2':
                play['half'] = 2
                play['overtime'] = 0
            elif row['game_half'] == 'Overtime':
                play['half'] = None
                play['overtime'] = 1
            else:
                print('ERROR #0 - row[\'game_half\'] != \'Half1\' or \'Half2\' or \'Overtime\'')
                play['half'] = None

        def add_pass_details():
            if  pd.isna(row['pass_length']) and not pd.isna(row['run_location']):
                play['run'] = 1
                play['short_pass'] = None
                play['deep_pass'] = None
                play['run_vs_pass'] = 1
            elif row['pass_length'] == 'short':
                play['run'] = 0
                play['short_pass'] = 1
                play['deep_pass'] = 0
                play['run_vs_pass'] = 0
            elif row['pass_length'] == 'deep':
                play['run'] = 0
                play['short_pass'] = 0
                play['deep_pass'] = 1
                play['run_vs_pass'] = 0
            else:
                # this is reached if its a field goal or so
                play['run'] = 0
                play['short_pass'] = 0
                play['deep_pass'] = 0
                play['run_vs_pass'] = None

            if row['sack'] == 1:
                play['run_vs_pass'] = 0


        def add_type_of_play_details():
            if pd.isna(row['play_type']):
                play['type'] = 'end_of_period'
            elif 'TWO-POINT CONVERSION' in row['desc']:
                play['type'] = '2pt'
                if row['two_point_conv_result'] == 'success':
                    play['2pt_success'] = 1
                elif row['two_point_conv_result'] == 'failure':
                    play['2pt_success'] = 0
            elif 'extra_point' in row['play_type']:
                play['type'] = '1pt'
                if row['extra_point_result'] == 'good':
                    play['1pt_success'] = 1
                else:
                    play['1pt_success'] = 0
            else:
                play['type'] = row['play_type']

        def add_home_away_score_details():
            play['home_score'] = row['total_home_score']
            play['away_score'] = row['total_away_score']
            if play['home_score'] > play['away_score']:
                play['home_winning'] = 1
                play['away_winning'] = 0
                play['scoreless_game'] = 0
                play['tie'] = 0
            elif play['away_score'] > play['home_score']:
                play['home_winning'] = 0
                play['away_winning'] = 1
                play['scoreless_game'] = 0
                play['tie'] = 0
            elif play['away_score'] == 0:
                play['home_winning'] = None
                play['away_winning'] = None
                play['scoreless_game'] = 1
                play['tie'] = 1
            else:
                play['home_winning'] = None
                play['away_winning'] = None
                play['scoreless_game'] = 0
                play['tie'] = 1

        def add_pos_def_score_details():
            play['pos_score'] = row['posteam_score']
            play['def_score'] = row['defteam_score']
            play['pos_minus_def_score'] = play['pos_score'] - play['def_score']
            if play['pos_score'] > play['def_score']:
                play['pos_winning'] = 1
                play['def_winning'] = 0
                play['tie'] = 0
            elif play['away_score'] > play['home_score']:
                play['pos_winning'] = 0
                play['def_winning'] = 1
                play['tie'] = 0
            else:
                play['pos_winning'] = 0
                play['def_winning'] = 0
                play['tie'] = 1

        def add_down_converted():
            if row['ydstogo'] <= row['yards_gained']:
                play['converted'] = True
            else:
                play['converted'] = False

        def add_td():
            if row['posteam_score_post'] == row['posteam_score'] + 6:
                play['pos_td'] = 1
            else:
                play['pos_td'] = 0
            if row['defteam_score_post'] == row['defteam_score'] + 6:
                play['def_td'] = 1
            else:
                play['def_td'] = 0

        def add_time_of_play():
            if next_row is None:
                play['play_t_length'] = play['qt_left']
                play['last_play_of_game'] = 1
            else:
                play['play_t_length'] = row['game_seconds_remaining'] - next_row['game_seconds_remaining']
                play['last_play_of_game'] = 0
                if play['play_t_length'] > 65 or play['play_t_length'] < 0:
                    play['play_t_length'] = None

        def add_penalty():
            if row['penalty']:
                play['penalty'] = 1
                play['penalty_1st'] = row['first_down_penalty']
                play['penalty_team'] = row['penalty_team']
                if play['penalty_team'] == play['pos_team']:
                    play['penalty_pos'] = 1
                else:
                    play['penalty_pos'] = 0
                play['penalty_type'] = row['penalty_type']
                play['penalty_yd'] = row['penalty_yards']
                if 'declined' in row['desc']:
                    play['penalty_acpt'] = 0
                else:
                    play['penalty_acpt'] = 1
            else:
                play['penalty'] = 0

        def add_turnover():
            play['fumble'] = row['fumble']
            #play['fumble_out'] = row['fumble_out_of_bounds']
            if play['fumble']:
                play['fumble_lost'] = row['fumble_lost']
                play['fumble_spot_yds'] = row['yards_gained']
                if play['fumble_lost']:
                    play['fumble_spot_yds'] -= row['fumble_recovery_1_yards']
                else:
                    play['fumble_spot_yds'] += row['fumble_recovery_1_yards']
            else:
                set_none(['fumble_lost', 'fumble_spot_yds'])

            play['interception'] = row['interception']
            if play['interception']:
                play['int_spot_yds'] = row['air_yards'] - row['return_yards']
            else:
                play['int_spot_yds'] = None

        def add_punt():
            play['punt'] = row['punt_attempt']
            if play['punt']:
                play['punt_blocked'] = row['punt_blocked']
                if 'Touchback' in row['desc']:
                    play['punt_yds'] = play['goal_yd'] - 20
                else:
                    play['punt_yds'] = row['kick_distance'] - row['return_yards']
            else:
                set_none(['punt_blocked', 'punt_yds'])

        def add_FG():
            play['pos_down_three_or_less'] = 1 if -3 <= play['pos_minus_def_score'] <= 0 else 0
            if isinstance(row['field_goal_result'], str):
                play['FG_attempted'] = 1
                if row['field_goal_result'] == 'made':
                    play['FG_made'] = 1
                else:
                    play['FG_made'] = 0
            else:
                play['FG_attempted'] = 0
                play['FG_made'] = None

        def set_none(list_of_keys):
            for key in list_of_keys:
                play[key] = None

        def add_week():
            play['year'] = row['game_date'].year
            play['month'] = row['game_date'].month
            play['day'] = row['game_date'].day
            if play['day'] < 10:
                week_str = str(play['month']) + '0' + str(play['day'])
            else:
                week_str = str(play['month']) + str(play['day'])
            play['week'] = DATE_NUMBER_TO_WEEK_NUMBER[play['year']][week_str]

        play = {}
        play['time_left'] = row['game_seconds_remaining']
        play['half_left'] = row['half_seconds_remaining']
        play['qt_left'] = row['quarter_seconds_remaining']

        play['home'] = self.home
        play['away'] = self.away
        play['down'] = row['down']
        play['yd_gain'] = row['yards_gained']
        play['ydstogo'] = row['ydstogo']
        play['ydsnet'] = row['ydsnet']
        play['id'] = 'g' +  str(row['game_id']) + 'p' + str(row['play_id'])
        play['pos_team'] = row['posteam']
        play['def_team'] = row['defteam']
        play['goal_yd'] = row['yardline_100']
        play['qtr'] = row['qtr']
        add_half_details()
        add_pass_details()
        add_type_of_play_details()
        add_home_away_score_details()
        add_pos_def_score_details()
        add_down_converted()
        add_td()
        add_time_of_play()
        add_penalty()
        add_turnover()
        add_punt()
        add_FG()
        add_week()
        self.plays[play['id']] = play

def iterate_df(df):
    print('Running iterate_df...')
    print('Number of rows:', df.shape[0])
    games = []
    plays = {}
    i = 0
    df.index = np.arange(0, len(df)) # index will start at some high number if not year 2009, the first year of data xl
    for idx, row in tqdm(df.iterrows()):
        if not i:
            game = Game(row['home_team'], row['away_team'])
            i += 1
        if not game.is_same_game(row):
            games.append(game)
            game = Game(row['home_team'], row['away_team'])
        elif idx + 1 < df.shape[0]:
            game.process_row(row, next_row = df.loc[idx+1])
        else:
            game.process_row(row)
        plays.update(game.plays)

    plays_df = pd.DataFrame.from_dict(plays, 'index')

    return plays_df

def load_big(year=2012, small=True):
    if year == 2019:
        df1 = pickle_wrap('pbp_data\\pkl\\2019_pbp.pkl', lambda: pd.read_excel('pbp_data\\2019_wk_first12.xlsx'))
        df2 = pickle_wrap('pbp_data\\pkl\\2019_pbp_wk13141516.pkl', lambda: pd.read_excel('pbp_data\\2019_wk13141516.xlsx'))
        df = pd.concat([df1, df2], axis=0)
        return df
    print('Loading...')
    print('Year:', year)
    print('Small:', small)
    if small:
        df = pickle_wrap('pbp_small.pkl', lambda: pd.read_excel('pbp_small.xlsx'), easy_override=True)
    else:
        df = pickle_wrap('pbp.pkl', lambda: pd.read_excel('pbp.xlsx'), easy_override=False)
    if year is not None and not small:
        return df[df['game_date'].apply(lambda x: x.year)==year]
    else:
        return df
    #df_ = pickle_wrap('pbp_small.pkl', lambda: pd.read_excel('pbp_small.xlsx'), easy_override=True)

def make_weeks_from_df(year):
    print('Making weeks from df...')
    for week in range(1, 18):
        start = time()
        print()
        print('Week:', week)

        if True:
            file_path = 'pbp_data\\y_' + str(year) + '_wk_excl_' + str(week) + '.pkl'
            week_df = plays_df[plays_df['week'] != week]
            week_df = week_df[week_df['week'] != 17] # FUGGGGGGGGGGGGGGGG # TODO !!!!!!
        else:
            file_path = 'pbp_data\\y_' + str(year) + '_wk_' + str(week) + '.pkl'
            week_df = plays_df[plays_df['week'] <= week]
        # TODO !!!!!!
        # TODO !!!!!!
        # TODO !!!!!!# TODO !!!!!!
        # error detected at 1.18.2018 at 2 am
        # shouldn't apply to new 2019 adta
        print('Time needed to do week:', round(time()-start,4),'s') # for my curiosity
        start = time()
        with open(file_path, 'wb') as file:
            pickle.dump(week_df, file)
        print('Time needed to do dump:', round(time()-start,4),'s') # for my curiosity

if __name__ == '__main__':
    #iterate_df
    YEAR = 2019
    SMALL = False
    df = pickle_wrap('pbp_' + str(YEAR) + str(SMALL) + '.pkl', lambda: load_big(year=YEAR, small=SMALL), easy_override=False)
    plays_df = pickle_wrap('plays_' + str(YEAR) + str(SMALL) + '.pkl', lambda: iterate_df(df), easy_override=False)
    print('go:', plays_df['week'].unique())
    make_weeks_from_df(YEAR)
    #plot_3d(plays_df)
    #create_pass_predictor(plays_df)
