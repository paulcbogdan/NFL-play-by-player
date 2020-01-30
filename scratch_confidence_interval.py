from game_interaction import * # uh this is importing something needed for statsmodels but idk what
import statsmodels as sm
from random import randint


def print_ci(count_home_wins, count_home_losses):
    propo_win = count_home_wins / (count_home_wins + count_home_losses)
    for method in ['normal', 'agresti_coull', 'beta', 'wilson', 'jeffreys']:
        lower_bound, upper_bound = sm.stats.proportion.proportion_confint(count_home_wins,
                                                                          count_home_wins + count_home_losses,
                                                                          alpha=.025, method=method.lower())
        print(method[0:6], '\tSim W/L %:', round(propo_win, 3), '[', round(lower_bound, 3), ',', round(upper_bound, 3)
              ,'] n =', count_home_wins + count_home_losses)

for num_sim in range(30):
    cw = randint(0, 426)
    cl = 426 - cw
    cw = int(426*.23)
    cl = int(426*.77)
    #cl = randint(0, 500)
    print_ci(cw, cl)
    print()