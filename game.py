from random import random
import pandas as pd
from predictors import Distribution_Manager, Predictor_Manager, Heirarchical_DM, Basic_Predictor
from utils import *
import statistics as stat
from termcolor import colored


class Game():
    def __init__(self, home_team, away_team, df, print=True):
        self.home = home_team
        self.away = away_team
        if random() > 0.5:
            self.pos_team = self.home
            self.def_team = self.away
        else:
            self.pos_team = self.away
            self.def_team = self.home
        self.is_kickoff = True
        self.is_extra_point = True
        self.scores = {self.home: 0, self.away: 0}
        self.ydline_100_pos = None
        self.first_down = None
        self.DMs = {'run_pass': Heirarchical_DM(df, ['run_vs_pass', 'ydstogo'], 'yd_gain'),
                    'play_t_length': Distribution_Manager(df, 'run_vs_pass', 'play_t_length')}
        #self.DMs = {'run_pass': Distribution_Manager(df, 'ydstogo', 'yd_gain')}
        self.PMs = {'run_pass': Predictor_Manager(df, 'yd_gain ~ 1 + time_left * ydstogo + pos_team + run_vs_pass'),
                    'play_t_length': Predictor_Manager(df, 'play_t_length ~ 1 + time_left * run_vs_pass * pos_team')}
        self.time_left = 3600
        self.first_down_100 = None
        self.down = None
        self.print = print
        self.basic_predictors = {'run_vs_pass': Basic_Predictor(df, 'run_vs_pass ~ pos_winning*pos_minus_def_score * time_left * ydstogo + pos_winning*pos_minus_def_score * qtr * ydstogo',
                                                                logistic=True),
                                 'punt': Basic_Predictor(df[df['down'] == 4],
                                                         'punt ~ time_left*pos_minus_def_score*ydstogo*goal_yd',
                                                         logistic=True),
                                 'punt_yds': Basic_Predictor(df[df['punt'] == 1], 'punt_yds ~ goal_yd'),
                                 'FG_attempted': Basic_Predictor(df[(df['down'] == 4) & (df['punt'] == 0)],
                                                                 'FG_attempted ~ time_left*pos_down_three_or_less*goal_yd*ydstogo + time_left*pos_minus_def_score*goal_yd*ydstogo',
                                                                 logistic=True),
                                 'FG_made': Basic_Predictor(df[df['FG_attempted'] == 1],
                                                                 'FG_made ~ goal_yd',
                                                                 logistic=True),
                                 'fumble': Basic_Predictor(df, 'fumble ~ pos_winning*time_left*run_vs_pass', logistic=True),
                                 'fumble_lost': Basic_Predictor(df[df['fumble'] == 1], 'fumble_lost ~ pos_winning', logistic=True),
                                 'fumble_yds': Basic_Predictor(df[df['fumble'] == 1], 'fumble_spot_yds ~ fumble_lost'),
                                 'interception': Basic_Predictor(df[df['run_vs_pass'] == 0], 'interception ~ pos_winning*time_left', logistic=True),
                                 'int_yds': Basic_Predictor(df[df['interception'] == 1], 'int_spot_yds ~ def_team')}
        self.pos_minus_def_score = None
        self.yd_to_go = None
        self.qtr = 1

    def update_basics(self):
        self.pos_minus_def_score = self.scores[self.pos_team] - self.scores[self.def_team]
        self.pos_down_three_or_less = 1 if -3 <= self.pos_minus_def_score <= 0 else 0
        self.pos_winning = 1 if self.pos_minus_def_score > 0 else 0
        self.ydstogo = self.ydline_100_pos - self.first_down_100
        self.qtr = 4 - int(self.time_left / 900)

    def simulate_play(self): # self.pos_team is receiving
        if self.check_end_game():
            return True
        if self.is_kickoff:
            self.kickoff()
            self.is_kickoff = False
            return
        self.update_basics()
        is_FG = False
        is_punt = self.check_and_do_punt()
        if is_punt:
            run_pass = False
        else:
            is_FG, FG_made = self.check_and_do_FG()
            run_pass = not is_FG

        if run_pass:
            is_run = self.predict_run_vs_pass()
            self.penalty_check()
            fumble_or_int, ball_lost, yd_change = self.turnover_check(is_run)
            if fumble_or_int:
                self.move_yds(yd_change, ball_lost, move_yd_str='Fumbled into a 1st down:')
            else:
                self.run_pass_play(is_run)
            t_play = self.get_play_t_length(is_run)
        else:
            t_play = 10

        if not is_FG:
            self.check_TD()

        #print('Play length:', self.get_play_t_length(is_run))
        if t_play < 0 or t_play > 60:
            for i in range(0, 400):
                print(colored('ERROR IN T_PLAY', 'green'))
        self.time_left = self.time_left - t_play
        self.check_4th_down()

    def check_punt(self):
        if self.down != 4: # catches ~99.999% of plays I imagine
            return False, None
        else:
            is_punt, real_p = self.basic_predictors['punt'].predict(pd.DataFrame.from_dict({'punt': [None],
                                                                             'pos_minus_def_score': [self.pos_minus_def_score],
                                                                             'time_left': [self.time_left],
                                                                             'ydstogo': [self.ydstogo],
                                                                             'goal_yd': [self.ydline_100_pos]}),
                                                                    get_real_p=True)
            if self.print:
                print('p(punt) =', real_p)
            # TODO: implement blocked punts
            if is_punt:
                punt_yds = self.basic_predictors['punt_yds'].predict(pd.DataFrame.from_dict({'punt_yds': [None],
                                                                             'goal_yd': [self.ydline_100_pos]}))
                print('Punting the ball:', punt_yds)
                return is_punt, punt_yds
            else:
                return is_punt, None

    def check_and_do_punt(self):
        is_punt, punt_yds = self.check_punt()
        if is_punt:
            self.move_yds(punt_yds, True, 'Punting the ball. Possession of')
        return is_punt

    def check_FG(self):
        if self.down != 4: # catches ~99.999% of plays I imagine
            return False, None
        else:
            is_FG, real_p = self.basic_predictors['FG_attempted'].predict(pd.DataFrame.from_dict({'FG_attempted': [None],
                                                                             'pos_minus_def_score': [self.pos_minus_def_score],
                                                                             'pos_down_three_or_less': [self.pos_down_three_or_less],
                                                                             'time_left': [self.time_left],
                                                                             'ydstogo': [self.ydstogo],
                                                                             'goal_yd': [self.ydline_100_pos]}),
                                                                    get_real_p=True)
            if self.print:
                print('p(FG_attempt) =', real_p)
            if is_FG:
                FG_made = self.basic_predictors['punt_yds'].predict(pd.DataFrame.from_dict({'FG_made': [None],
                                                                             'goal_yd': [self.ydline_100_pos]}))
                print('FG is:', FG_made)
                return is_FG, FG_made
            else:
                return is_FG, None

    def check_and_do_FG(self):
        is_FG, FG_made = self.check_FG()
        if is_FG:
            if FG_made:
                self.is_kickoff = True
                self.scores[self.pos_team] += 3
                print(colored('FG is good from', str(self.ydline_100_pos) + '. Score:', self.scores), 'yellow')
            else:
                print(colored('FG failed from', str(self.ydline_100_pos) + '. Score:', self.scores), 'orange')
                self.ydline_100_pos += 10
            self.transfer_possession(dont_print=True)
        return is_FG, FG_made

    def move_yds(self, yd_change, ball_lost, move_yd_str=''):
        self.ydline_100_pos = float(int(self.ydline_100_pos - yd_change))
        if ball_lost:
            self.transfer_possession()
        else:
            if self.is_first_down():
                self.reset_downs()
                print(move_yd_str, self.pos_team, 'ball. 1st and 10 to go.')
            else:
                self.down += 1

    def check_TD(self):
        if self.ydline_100_pos < 0:
            self.scores[self.pos_team] += 7
            if self.print:
                print()
                print(colored('Touchdown:', self.pos_team,'Score:', self.get_score_str()), 'red')
            self.transfer_possession(dont_print=True)
            self.kickoff()

    def penalty_check(self):
        pass

    def turnover_check(self, is_run):
        is_fumble = self.basic_predictors['fumble'].predict(pd.DataFrame.from_dict({'fumble': [None],
                                                                       'pos_winning': [self.pos_winning],
                                                                       'time_left': [self.time_left],
                                                                       'run_vs_pass': [is_run]}))
        if is_fumble:
            is_fumble_lost = self.basic_predictors['fumble_lost'].predict(pd.DataFrame.from_dict({'fumble_lost': [None],
                                                                                             'pos_winning': [self.pos_winning]}))
            fumble_spot_yds = self.basic_predictors['fumble_yds'].predict(pd.DataFrame.from_dict({'fumble_spot_yds': [None],
                                                                                                'fumble_lost': [is_fumble_lost]}))
            print('Fumbled!!!')
            print('Lost:', is_fumble_lost)
            return True, is_fumble_lost, int(fumble_spot_yds + 0.5)

        if not is_run:
            return None, None, None
        is_int = self.basic_predictors['interception'].predict(pd.DataFrame.from_dict({'interception': [None],
                                                                       'pos_winning': [self.pos_winning],
                                                                       'time_left': [self.time_left]}))
        if is_int:
            int_spot_yds = self.basic_predictors['int_yds'].predict(pd.DataFrame.from_dict({'int_spot_yds': [None],
                                                                                           'def_team': [self.def_team]}))
            print('Intercepted!!!')
            return True, True, int(int_spot_yds + 0.5)
        else:
            return None, None, None

    def kickoff(self):
        self.ydline_100_pos = 75
        self.first_down_100 = 65
        self.down = 1
        if self.print:
            print('Kickoff:', self.pos_team,'gets the ball at their', self.ydline_100_pos,'yard line. 1st down.')

    def predict_run_vs_pass(self):
        return self.basic_predictors['run_vs_pass'].predict(pd.DataFrame.from_dict({'run_vs_pass': [-999],
                                                                             'pos_minus_def_score': [self.pos_minus_def_score],
                                                                             'time_left': [self.time_left],
                                                                             'ydstogo': [self.ydstogo],
                                                                             'qtr': [self.qtr],
                                                                             'pos_winning': [self.pos_winning]}))

    def run_pass_play(self, is_run):
        current_play_df = pd.DataFrame.from_dict({'time_left': [self.time_left],
                                                  'ydstogo': self.ydstogo,
                                                  'pos_team': self.pos_team,
                                                  'yd_gain': -999,
                                                  'run_vs_pass': is_run})
        tile = self.PMs['run_pass'].predict(current_play_df)
        yd_gain_pred = self.DMs['run_pass'].get_val([is_run, self.ydstogo], tile)
        self.ydline_100_pos -= yd_gain_pred
        if self.is_first_down():
            self.reset_downs()
        else:
            self.down += 1
        if self.print:
            if self.down <= 4:
                print(self.is_run_to_str(is_run, yd_gain_pred), self.pos_team,'gains',yd_gain_pred,'to',
                      self.ydline_100_pos, 'yd line.', self.get_togo_str())
            else:
                print(self.is_run_to_str(is_run, yd_gain_pred), self.pos_team,'gains',yd_gain_pred,'to',
                      self.ydline_100_pos, 'yd line. Transferring possession!' )
                print()

    def check_4th_down(self):
        if self.down == 5:
            self.transfer_possession()

    def reset_downs(self):
        self.first_down_100 = self.ydline_100_pos - 10
        self.down = 1

    def transfer_possession(self, dont_print=False):
        self.ydline_100_pos = 100 - self.ydline_100_pos
        temp = self.def_team
        self.def_team = self.pos_team
        self.pos_team = temp
        self.reset_downs()
        if self.print:
            if not dont_print:
                print(self.pos_team,'gets the ball at',str(self.ydline_100_pos) + '.', self.get_togo_str())

    def is_first_down(self):
        if self.ydline_100_pos <= self.first_down_100:
            return True
        else:
            return False

    def get_togo_str(self):
        return str(self.down) +  ' down ' + str(self.ydline_100_pos - self.first_down_100) +  ' to go. Time: ' + str(self.time_left)

    def get_score_str(self):
        return self.home + ': ' +  str(self.scores[self.home]) + ', ' + self.away + ': ' + str(self.scores[self.away])

    def check_end_game(self):
        if self.time_left <= 0:
            if self.print:
                print()
                print('Game over. Final score:', self.get_score_str())
            return True

    def get_play_t_length(self, is_run):
        current_play_df = pd.DataFrame.from_dict({'time_left': [self.time_left],
                                                  'pos_team': self.pos_team,
                                                  'play_t_length': -999,
                                                  'run_vs_pass': is_run})
        tile = self.PMs['play_t_length'].predict(current_play_df)
        return self.DMs['play_t_length'].get_val(is_run, tile)

    def is_run_to_str(self, is_run, yd_gain_pred):
        if is_run:
            return 'Run play.'
        else:
            if yd_gain_pred < 0:
                return 'Sack.'
            else:
                return 'Pass play.'

def simulate_game(df, print=True):
    game = Game(TEAM_A, TEAM_B, df, print=print)
    final_score = None
    for i in range(0, 200):
        is_game_ended = game.simulate_play()
        if is_game_ended:
            final_score = game.scores
            break
    return final_score[TEAM_A], final_score[TEAM_B]


# TODO: add FGs
# TODO: add pos_team and def_team as predictors
# TODO: analyze season of play

TEAM_A = 'KC'
TEAM_B = 'CHI'

if __name__ == '__main__':
    YEAR = 2018
    SMALL = False
    df = pickle_wrap('plays_' + str(YEAR) + str(SMALL) + '.pkl', lambda: iterate_df(df), easy_override=False)
    A_scores = []
    B_scores = []
    A_wins = []
    for j in range(0, 100):
        A, B = simulate_game(df, print=True)
        A_scores.append(A)
        B_scores.append(B)
        if A > B:
            A_wins.append(1)
        elif A == B:
            A_wins.append(0.5)
        else:
            A_wins.append(0)
        print()
        print('--*---*---')
        print('Average', TEAM_A,'scores:', stat.mean(A_scores), ',',TEAM_B,'scores:', stat.mean(B_scores))
        print('Proportion of A wins:', stat.mean(A_wins))
        print('iteration:', j)

# can calculate expected interceptions for different players' seasons
