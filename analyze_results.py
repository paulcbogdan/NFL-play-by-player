from os import listdir
from os.path import isfile, join
import pickle
from mass_simulate_compare_odds import add_to_running_dict_df, extract_info, get_predicted_win_loss, load_odds_history_df
import statistics as stat
import statsmodels as sm
from math import sqrt

def get_all_file_paths(base_path='results_save'):
    return [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f))]

def load_file_for_oddsrow(odds_row, excl_vs_before=True, n_sims=258):
    if excl_vs_before:
        excl_str = '_excl'
    else:
        excl_str = ''
    H_team = odds_row['home']
    A_team = odds_row['away']
    fp_ = 'results_save\\yr.' + str(odds_row['year']) + '_wk.' + str(odds_row['week']) + excl_str + '_H.' + H_team + '_A.' + A_team + '_n.' + str(n_sims) + '_results.pkl'
    with open(fp_, 'rb') as file:
        final_data = pickle.load(file)
    return final_data

def apply_home_team_advantage(odds_row, home_mod, excl_vs_before=False, n_sims=258, percentile_break=.1):
    if excl_vs_before:
        excl_str = '_excl'
    else:
        excl_str = ''
    week = odds_row['week']

    H_team = odds_row['home']
    A_team = odds_row['away']
    fp_ = 'results_save\\yr.' + str(odds_row['year']) + '_wk.' + str(odds_row['week']) + excl_str + '_H.' + H_team + '_A.' + A_team + '_n.' + str(n_sims) + '_results.pkl'
    if not isfile(fp_):
        return
    with open(fp_, 'rb') as file:
        final_data = pickle.load(file)
    H_minus_A_history = final_data['H_minus_A_history']
    H_plus_A_history = final_data['H_plus_A_history']

    print('test:', H_minus_A_history)

    H_minus_A_history = [x + home_mod for x in H_minus_A_history]

    if odds_row['favorite'] == odds_row['home']:
        home_spread_open = odds_row['spread_open']
        home_spread_close = odds_row['spread_close']
    else:
        home_spread_open = -1*odds_row['spread_open']
        home_spread_close = -1*odds_row['spread_close']

    over_under_better_open, over_under_better_close, WL_better_than_vegas, better_ATS_open, better_ATS_close, AST_open_dif, AST_close_dif, OU_open_better_dif, OU_close_better_dif, final_total = \
        extract_info(H_minus_A_history, H_plus_A_history, odds_row, home_spread_open, home_spread_close, percentile_break)

    if odds_row['final_real_home_minus_away'] > 0:
        H_win = 1
    elif odds_row['final_real_home_minus_away'] < 0:
        H_win = 0
    else:
        H_win = .5



    WL_better_than_vegas, p_dif_temp = get_predicted_win_loss(H_minus_A_history, odds_row['ML'], home_spread_close,
                                                              odds_row['final_real_home_minus_away'])

    final_data = {'home': H_team, 'away': A_team,
                  'home_spread_open': home_spread_open, 'home_spread_close': home_spread_close,
                  'pred_spread_median': stat.median(H_minus_A_history), 'pred_spread_avg': stat.mean(H_minus_A_history),
                  'over_under_better_open': over_under_better_open, 'over_under_better_close': over_under_better_close,
                  'WL_better_vegas': WL_better_than_vegas, 'better_ATS_open': better_ATS_open,
                  'better_ATS_close': better_ATS_close, 'n_sims': n_sims, 'excl_vs_before': excl_vs_before,
                  'True_H_win': H_win, 'True_H_minus_A': odds_row['final_real_home_minus_away'], 'True_total_score': final_total,
                  'AST_open_dif': AST_open_dif, 'AST_close_dif': AST_close_dif,
                  'OU_open_better_dif': OU_open_better_dif, 'OU_close_better_dif': OU_close_better_dif, 'H_minus_A_history': H_minus_A_history,
                  'H_plus_A_history': H_plus_A_history}

    fp_out = 'results_save\\yr.' + str(odds_row['year']) + '_wk.' + str(odds_row['week']) + '_hm.' + str(home_mod) + excl_str + '_H.' + H_team + '_A.' + A_team + '_n.' + str(n_sims) + '_results.pkl'
    with open(fp_out, 'wb') as file:
        pickle.dump(final_data, file)

def update_home_mod(home_mod=2):
    print('Upating data. Home_mod =', 2)
    YEAR = 2019
    MAX_EFFECTIVE_TIME = 900
    odds_data_df = load_odds_history_df(YEAR)
    running_dict = {}
    HOME_MOD = 2
    for idx, row in odds_data_df.iterrows():
        apply_home_team_advantage(row, home_mod, n_sims=999)

def sim_season(year=2018, n_sims=258):
    YEAR = year
    odds_data_df = load_odds_history_df(YEAR)
    team_records = {}
    for idx, odds_row in odds_data_df.iterrows():
        week = odds_row['week']
        if False:
            if YEAR == 2019:
                if week < 3 or week == 15 or week == 17:
                    continue
            elif YEAR == 2018:
                if week in [3, 4, 8, 12, 14, 15, 16, 17]:
                    continue
        H_team = odds_row['home']
        A_team = odds_row['away']
        for team in [H_team, A_team]:
            if team not in team_records:
                team_records[team] = 0
        week_data = load_file_for_oddsrow(odds_row, excl_vs_before=True, n_sims=n_sims)
        if YEAR == 2019:
            n_home_wins = 0
            n_home_losses = 0
            for game_plus_minus in week_data['H_minus_A_history']:
                if game_plus_minus > 0:
                    n_home_wins += 1
                elif game_plus_minus < 0:
                    n_home_losses += 1
            p_home_win = n_home_wins / (n_home_wins + n_home_losses)
        else:
            p_home_win = week_data['p_home_win'] / (week_data['p_home_win'] + week_data['p_away_win'])
        team_records[H_team] += p_home_win
        team_records[A_team] += 1 - p_home_win
    print('records:')
    print(team_records)

    sorted_records = {k: v for k, v in sorted(team_records.items(), key=lambda item: item[1])}
    print(sorted_records)
    team_actual_records_2019 = {'CIN': 2, 'MIA': 5, 'WAS': 3, 'NYG': 4, 'CAR': 5, 'JAX': 6, 'NYJ': 7, 'DET': 3, 'ARI': 5,
                           'IND': 7, 'ATL': 7, 'OAK': 7, 'CHI': 8, 'DEN': 7, 'CLE': 6, 'HOU': 10, 'SEA': 11, 'PIT': 8,
                           'PHI': 9, 'GB': 13, 'TEN': 9, 'BUF': 10, 'LAC': 5, 'LA': 9, 'MIN': 10, 'TB': 7, 'DAL': 8,
                           'NO': 13, 'BAL': 14, 'KC': 12, 'SF': 13, 'NE': 12}

    team_actual_records = team_actual_records_2019
    # HOU > BUF: Wrong
    # TIT > NE: Wrong
    # MIN > NO: Wrong
    # SEA > PHI: Wrong

    # 49  > MIN: Correct!
    # TIT > BAL: Wrong
    # KC  > HOU: Correct!
    # GB  > SEA: Wrong

    # KC  > TIT: Correct!
    # 49  >  GB: Correct!

    # HOU was the lowest at 17

    sum = 0
    difs = []
    actual = []
    sim = []
    for i, team in enumerate(sorted_records):
        print(32-i, '\t', team, '\t', round(sorted_records[team], 2), '\t', team_actual_records[team], '\t, dif:\t',  round(team_actual_records[team] - sorted_records[team], 2))
        dif = abs(team_actual_records[team] - sorted_records[team])
        actual.append(team_actual_records[team])
        sim.append(sorted_records[team])
        sum += sorted_records[team]
        difs.append(dif)
    print(stat.mean(difs))
    print(stat.mean(actual))
    print(stat.mean(sim))
    print('Mean dif:', round(stat.mean(difs), 3))
    print('RMSE:', sqrt(stat.mean([dif * dif for dif in difs])))

def create_max_num_fp():
    file_paths = get_all_file_paths()
    fp_inits = {}
    fp_n = {}
    for fp in file_paths:
        fp_initial = fp.split('_n.')[0]
        num_runs = int(fp.split('_n.')[1][0:3])
        if fp_initial not in fp_inits:
            with open(fp, 'rb') as file:
                data = pickle.load(file)
            fp_inits[fp_initial] = data
            fp_n[fp_initial] = num_runs

        if num_runs > fp_n[fp_initial]:
            with open(fp, 'rb') as file:
                data = pickle.load(file)
            fp_inits[fp_initial] = data
            fp_n[fp_initial] = num_runs

    for fp_initial in fp_inits:
        new_fn = fp_initial + '_n.999_results.pkl'
        with open(new_fn, 'wb') as file:
            pickle.dump(fp_inits[fp_initial], file)


def analyze_fp():
    file_paths = get_all_file_paths()
    running_dict = {}
    for fp in file_paths:
        # if 'excl' in fp:
        #    continue
        # if 'wk.12' not in fp:
        #    continue

        skip = True
        for wk in range(1, 11):
           if 'wk.' + str(wk) + '_' in fp:
                skip = False

        #if skip:
        #    continue

        if '2020' in fp:
            continue
        if 'excl' in fp:
            continue
        if 'n.999' not in fp:
            continue

        if 'hm' in fp:
            continue

        if 'wk.17' in fp:
            continue
        with open(fp, 'rb') as file:
            results = pickle.load(file)
        # if 'excl_vs_before' in results:
        #    results.pop('excl_vs_before')
        #    continue
        add_to_running_dict_df(running_dict, results)

    print(running_dict)
    print('n =', len(running_dict['home']))
    for key, list_ in running_dict.items():
        #if key not in ['over_under_better_open', 'over_under_better_close']: # 'better_ATS_open', 'better_ATS_close', 'WL_better_vegas',
        #    continue

        try:
            print(key, ': Average =', round(stat.mean(list_)*100,1), ', median =', round(stat.median(list_),3), ', n =', len(list_))
            method = 'beta'

            good = int(len(list_)*stat.mean(list_))

            lower_bound, upper_bound = sm.stats.proportion.proportion_confint(good,
                                                                              len(list_),
                                                                              alpha=.025, method=method.lower())
            print('[' + str(round(lower_bound*100, 1)) + ', ' + str(round(upper_bound*100, 1)) + ']')
            print('(' + str(round(stat.mean(list_)*100,1)) + ' [' + str(round(lower_bound*100, 1)) + ', ' + str(round(upper_bound*100, 1)) + '])')

            print('abs avg =', round(stat.median([abs(x) for x in list_]), 3))
            print()
        except:
            pass


if __name__ == '__main__':
    #create_max_num_fp()
    sim_season(year=2019, n_sims=999)
    #update_home_mod()
    #analyze_fp()
