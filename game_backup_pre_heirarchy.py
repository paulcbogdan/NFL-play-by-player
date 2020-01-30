from random import random
import pandas as pd
from predictors import Distribution_Manager, Predictor_Manager
from utils import *
import statistics as stat

class Game():
    def __init__(self, home_team, away_team, df, print=True):
        self.home = home_team
        self.away = away_team
        if random() > 0.5:
            self.pos_team = self.home
        else:
            self.pos_team = self.away
        self.is_kickoff = True
        self.is_extra_point = True
        self.scores = {self.home: 0, self.away: 0}
        self.ydline_100_pos = None
        self.first_down = None
        self.DMs = {'run_pass': Distribution_Manager(df, 'ydstogo', 'yd_gain')}
        self.PMs = {'run_pass': Predictor_Manager(df, 'yd_gain ~ 1 + time_left * ydstogo + pos_team')}
        self.time_left = 3600
        self.first_down_100 = None
        self.down = None
        self.print = print

    def simulate_play(self): # self.pos_team is receiving
        if self.check_end_game():
            return True
        if self.is_kickoff:
            self.kickoff()
            self.is_kickoff = False
            return
        self.run_pass_play()
        is_TD = self.check_TD()
        self.check_4th_down()

    def check_TD(self):
        if self.ydline_100_pos < 0:
            self.scores[self.pos_team] += 7
            if self.print:
                print()
                print('Touchdown:', self.pos_team,'Score:', self.get_score_str())
            self.transfer_possession(dont_print=True)
            self.kickoff()

    def kickoff(self):
        self.ydline_100_pos = 75
        self.first_down_100 = 65
        self.down = 1
        if self.print:
            print('Kickoff:', self.pos_team,'gets the ball at their', self.ydline_100_pos,'yard line. 1st down.')

    def predict_run_vs_pass(self):


    def run_pass_play(self):
        self.time_left = self.time_left - 25
        yd_to_go = self.ydline_100_pos - self.first_down_100
        current_play_df = pd.DataFrame.from_dict({'time_left': [self.time_left],
                                                  'ydstogo': yd_to_go,
                                                  'pos_team': self.pos_team,
                                                  'yd_gain': -999})
        tile = self.PMs['run_pass'].predict(current_play_df)
        #print('yd_to_go:', yd_to_go)
        #print('tile:', tile)
        yd_gain_pred = self.DMs['run_pass'].get_val(yd_to_go, tile)
        self.ydline_100_pos -= yd_gain_pred
        if self.is_first_down():
            self.reset_downs()
        else:
            self.down += 1
        if self.print:
            if self.down <= 4:
                print(self.pos_team,'gains',yd_gain_pred,'to', self.ydline_100_pos, 'yd line.', self.get_togo_str())
            else:
                print(self.pos_team,'gains',yd_gain_pred,'to', self.ydline_100_pos, 'yd line. Transferring possession!' )
                print()

    def check_4th_down(self):
        if self.down == 5:
            self.transfer_possession()

    def reset_downs(self):
        self.first_down_100 = self.ydline_100_pos - 10
        self.down = 1

    def transfer_possession(self, dont_print=False):
        self.ydline_100_pos = 100 - self.ydline_100_pos
        if self.pos_team == self.home:
            self.pos_team = self.away
        else:
            self.pos_team = self.home
        self.reset_downs()
        if self.print:
            if dont_print:
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

    def predict_time_of_play

def simulate_game(df, print=True):
    game = Game(TEAM_A, TEAM_B, df, print=print)
    final_score = None
    for i in range(0, 200):
        is_game_ended = game.simulate_play()
        if is_game_ended:
            final_score = game.scores
            break
    return final_score[TEAM_A], final_score[TEAM_B]

TEAM_A = 'KC'
TEAM_B = 'ARI'

if __name__ == '__main__':
    YEAR = 2018
    SMALL = False
    df = pickle_wrap('plays_' + str(YEAR) + str(SMALL) + '.pkl', lambda: iterate_df(df), easy_override=False)
    A_scores = []
    B_scores = []
    A_wins = []
    for j in range(0, 100):
        A, B = simulate_game(df, print=False)
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


