class Distribution_Manager_OLD():
    def __init__(self, df, conditions=()):
        self.df = df
        self.predictor_dict = {}
        self.conditions = conditions
        self.create_predictor_dict()
        self.unique_vals = None
        self.df_vals = {}

    def create_predictor_dict(self):
        print('test')
        unique_vals = {}
        for condition in self.conditions:
            unique_vals[condition] = self.df[condition].unique()
            unique_vals[condition].sort()
        level_dict = [self.predictor_dict]
        for condition in self.conditions:
            next_level_dict = []
            for unique_val in unique_vals[condition]:
                for branch in level_dict:
                    branch[unique_val] = {}
                    next_level_dict.append(branch[unique_val])
            level_dict = next_level_dict
        self.unique_vals = unique_vals
        print(self.predictor_dict)
        self.df_vals = deepcopy(self.predictor_dict)
        self.fill()

    def fill(self):
        combos = combinations_lists([self.unique_vals[condition] for condition in self.unique_vals])
        tups = []
        for combo in combos:
            tup = []
            for cond_number, val in enumerate(combo):
                tup.extend([self.conditions[cond_number], val])
            tups.append(tup)

        for tup in tups:
            self.add_to_depths(self.df_vals, tup, 2)
            #current_dict = self.df_vals
            #for i in range(0, len(tup), 2):
            #    current_dict = current_dict[tup[i+1]]
        print(self.df_vals)

    def add_to_depths(self, passed, tup, thing):
        if len(passed) == 0:
            passed = thing
        else:
            self.add_to_depths(passed[tup[1]], tup[2:], thing)

    def train(self):
        pass

class Distribution_Manager():
    def __init__(self, df, condition, DV):
        self.df = df
        self.predictor_dict = {}
        self.condition = condition
        self.df_vals = {}
        self.DV = DV
        self.create_vals_dict()

    def create_vals_dict(self):
        uniques = self.df[self.condition].unique()
        uniques.sort()
        for unique in uniques:
            self.df_vals[unique] = self.df[self.df[self.condition] == unique][self.DV].tolist()
            self.df_vals[unique].sort()
        print(self.df_vals)

    def get_val(self, condition_val, percentile):
        print(percentile)
        if percentile > 0.9999999: # prevents errors due to percentile = 1.0
            percentile = 0.999999
        return self.df_vals[condition_val][int(percentile * len(self.df_vals[condition_val]))]

class Predictor_Manager():
    def __init__(self, df, formula):
        mod = smf.ols(formula=formula, data=df)
        self.res = mod.fit()
        self.predictions_list = self.res.predict()
        #print('pre sort:', self.predictions_list)
        #print(vars(self.res.model.data))
        prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(self.res, exog=transform_exog_to_model(self.res, df))
        #print('std:', prstd_ols)
        self.predictions_list = self.predictions_list + np.random.normal(0, 1, prstd_ols.shape[0]) * prstd_ols
        #print('post std:', self.predictions_list)
        self.predictions_list.sort()
        #print('sorted:', self.predictions_list)

    def predict(self, input_df):
        print('*')
        pred_pre_std = self.res.predict(input_df)
        #prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(self.res, input_df)
        prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(self.res, exog=transform_exog_to_model(self.res, input_df))
        z = np.random.normal(0, 1, prstd_ols.shape[0]) * prstd_ols
        pred_final = (pred_pre_std + z).to_numpy()
        #print('std:', prstd_ols)
        #print('pre', pred_pre_std)
        #print('pred:', pred_final)
        pred_indices = np.searchsorted(self.predictions_list, pred_final)
        pred_percentiles = pred_indices / self.predictions_list.shape[0]
        print('percentiles:', pred_percentiles)
        return pred_percentiles

def combinations_lists(lists):
    first_list = lists[0]
    for list_ in lists[1:]:
        first_list = [(x,y) for x in first_list for y in list_]
    return first_list



def create_pass_predictor(df):
    df = df.dropna(subset=['ydstogo', 'time_left', 'yd_gain'])
    print('--------')
    df['pos_winning'].subtract(df['pos_winning'].mean())
    #mod = smf.ols(formula='yd_gain ~ time_left*pos_winning + half_left*pos_winning + qt_left*pos_winning', data=df)
    dm = Distribution_Manager(df, 'ydstogo', 'yd_gain')
    test = Predictor_Manager(df, 'yd_gain ~ 1 + time_left * ydstogo')
    #df = pd.DataFrame.from_dict({'time_left': [100, 200, 300, 300, 400], 'yd_gain': [3, 4, 1, 3, 2], 'ydstogo': [3, 4, 5, 6, 7]})
    tiles = test.predict(pd.DataFrame.from_dict(df))
    df['tiles'] = tiles
    df['super_pred'] = df.apply(lambda x: dm.get_val(x['ydstogo'], x['tiles']), axis=1)
    print(df['super_pred'])
    plt.title('super sim ' + str(datetime.datetime.now()))
    plt.hist2d(df['super_pred'], df['ydstogo'], bins=100, range=[[-40, 80], [0, 50]], norm=mpl.colors.LogNorm())
    plt.show()
    plt.title('real ' + str(datetime.datetime.now()))
    plt.hist2d(df['yd_gain'], df['ydstogo'], bins=100, range=[[-40, 80], [0, 50]], norm=mpl.colors.LogNorm())
    plt.show()
    #for idx, row in df.iterrows():
    #    print(row['tiles'])
    #    print(dm.get_val(row['ydstogo'], row['tiles']))
    return

def plot_3d(df):
    df = df[df['type'] == 'pass']

    df = df.reset_index()
    df['converted'] = df['converted'].astype(int)
    mod_conv = smf.logit(formula='converted ~ ydstogo + pos_team', data=df)
    res_conv = mod_conv.fit()

    pred_conv = pd.Series(res_conv.predict())
    df['pred_converted'] = pred_conv.apply(lambda x: random() < x)
    df = df.dropna(subset=['pred_converted', 'pos_team'])
    mod_yd = smf.ols(formula='yd_gain ~ converted*ydstogo + pos_team', data=df)
    res_yd = mod_yd.fit()


    predicting_df = pd.DataFrame()
    predicting_df['converted'] = df['pred_converted']
    predicting_df['converted'] = predicting_df['converted'].astype(int)
    predicting_df['ydstogo'] = df['ydstogo']
    predicting_df['pos_team'] = df['pos_team']

    plt.title('sim')
    df['pred_yd_gained'] = res_yd.predict(exog=predicting_df)

    prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(res_yd, exog=transform_exog_to_model(res_yd, predicting_df))
    df.loc[df['pred_yd_gained'] > -1000, 'yd_std'] = prstd_ols
    df = df.dropna(subset=['pred_yd_gained'])
    print('gag')
    from scipy.stats import skewnorm
    from scipy.stats import skew
    print(df[df['converted'] == 1]['yd_gain'])
    skew_dict = {1: skew(df[df['converted'] == 1]['yd_gain']), 0: 0}#skew(df[df['converted'] == 0]['yd_gain'])}
    a = skew(df[df['converted'] == 1]['yd_gain'])
    print('skew:', a)
    b = skew(df[df['converted'] == 0]['yd_gain'])
    print('skew:', b)
    print('shape:', df.loc[df['converted'] == 1, 'yd_std'].shape[0])
    a_k = stats.kurtosis(df[df['converted'] == 1]['yd_gain'])
    a_m = df[df['converted'] == 1]['yd_gain'].mean()
    a_v = df[df['converted'] == 1]['yd_gain'].var()

    f = pdf_mvsk([a_m, a_v, a, a_k])

    for i in range(-20, 80):
        print(i, ':', f(i))

    plt.hist(skewnorm.rvs(skew_dict[1], size=df.loc[df['converted'] == 1, 'yd_std'].shape[0]))
    plt.show()

    for converted in [0, 1]:
        s_dist = skewnorm.rvs(skew_dict[converted], size=df.loc[df['converted'] == converted, 'yd_std'].shape[0])
        print(s_dist - s_dist.mean())

        df.loc[df['converted'] == converted, 'super_pred'] = df.loc[df['converted'] == converted, 'pred_yd_gained'] + \
                                                     df.loc[df['converted'] == converted, 'yd_std'] \
                                                     * (s_dist - s_dist.mean())
                                                     #* np.random.uniform(low=0, high=2, size = df.loc[df['converted'] == converted, 'yd_std'].shape[0])
                                                     #* skewnorm.rvs(skew_dict[converted], size=df.loc[df['converted'] == converted, 'yd_std'].shape[0])

    #df.loc[df['converted'] == 1, 'super_pred'] = df.loc[df['converted'] == 1, 'pred_yd_gained'] + \
    #                                             df.loc[df['converted'] == 1, 'yd_std'] \
    #                                             * skewnorm.rvs(a, size=df.loc[df['converted'] == 1, 'yd_std'].shape[0])


    #df.loc[df['pred_yd_gained'] > -1000, 'super_pred'] = df['pred_yd_gained'] + df['yd_std'] * skewnorm.rvs(a, size=prstd_ols.shape[0])
    print('gucci')
    import datetime
    plt.title('sim ' + str(datetime.datetime.now()))
    plt.hist2d(df['super_pred'], df['ydstogo'], bins=100, range=[[-40, 80], [0, 50]], norm=mpl.colors.LogNorm())
    plt.show()
    plt.title('real ' + str(datetime.datetime.now()))
    plt.hist2d(df['yd_gain'], df['ydstogo'], bins=100, range=[[-40, 80], [0, 50]], norm=mpl.colors.LogNorm())
    plt.show()
    #plt.hist2d(res_yd.predict(exog=df), df['ydstogo'], norm=mpl.colors.LogNorm())
    #plt.show()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(res_yd.predict(), df['pred_converted'], df['ydstogo'])
    #plt.show()

    return

    mod = smf.ols(formula='converted ~ ydstogo', data=df)


    res = mod.fit()
    plt.hist2d(res.predict(), df['ydstogo'])
    plt.show()
    return


    plt.hist2d( df['yd_gain'], df['ydstogo'], bins=50, norm=mpl.colors.LogNorm())
    plt.show()
    return
    plt.scatter( df['yd_gain'], df['ydstogo'])
    plt.show()
    return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['converted'], df['yd_gain'], df['ydstogo'])
    plt.show()
