def train_and_submit(X_train, X_test, y_train, y_test, train_df, test_df, model, best_params,
                     file_sufix= 'tas', opis='', do_plot = False, 
                     save_min=7000, kaggle=False, kaggle_min=6000):
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import r2_score, mean_absolute_error
    from scikitplot.estimators import plot_learning_curve

    time_model_start = datetime.now().strftime("%H:%M:%S")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test[X_test.columns[1:]]) #bez car_id

    y_test_org = from_ln_pln_trans_to_predict(X_test, y_test, train_df)
    y_pred_org = from_ln_pln_trans_to_predict(X_test, pd.DataFrame(y_pred), train_df)

    mae = mean_absolute_error(y_test_org, y_pred_org)
    r2 = r2_score(y_test_org, y_pred_org)
    print(mae)
    print(r2)

    if do_plot: 
        learning_curve = plot_learning_curve(model, X_train, y_train, cv=3, random_state=0, shuffle=True)
        #plt.hist(y_test_org - y_pred_org, bins=50);

    time_model_end = datetime.now().strftime("%H:%M:%S")

    if mae <= save_min:
        save_model(name=file_sufix, train_df=train_df, test_df=test_df, used_feats=X_train.columns, 
                                        my_model=model, mae=mae, r2=r2,
                                        details = {"mean_absolute_error" : mae, "r2_score" : r2, "vars": ' | '.join(list(X_train.columns)), 
                                                   "best_params" : best_params,
                                         #"tuning_start" : time_hyper_tuning_start, "tuning_end" : time_hyper_tuning_end, 
                                         "model_start" : time_model_start, "model_end" : time_model_end, 
                                        "opis" : opis}, 
                                        plot_learning_curve = learning_curve if do_plot else None, 
                                        #  plt_hist = plt if do_plot else None, #nie zadzialaja 2 wykresy :(
                                        kaggle=(kaggle and mae < kaggle_min))
    return mae, r2, model

def my_train_test_split(df_X, df_y, train_N):

    import pandas as pd
    
    df_total = pd.merge(df_X, df_y, left_index=True, right_index=True)
    test_car_ids = train_N['car_id'].values
    X = df_X.columns[1:] #bez car_id
    #print(X)
    y = df_y.name
    #print(y)
    
    df_train = df_total[~df_total['car_id'].isin(test_car_ids)]
    df_test = df_total[df_total['car_id'].isin(test_car_ids)]

    X_train = df_train[X]
    X_test = df_test[df_X.columns[0:]] #z car_id
    y_train = df_train[y]
    y_test = df_test[y]
    
    return X_train, X_test, y_train, y_test

def merge_with_features(csv_file, train_df, test_df):
    
    import pandas as pd
    
    csv_file = csv_file.replace('test', 'REPLACE')
    csv_file = csv_file.replace('train', 'REPLACE')
    try:
        features_train = pd.read_csv('~/pml7/konkurs/output/' + csv_file.replace('REPLACE', 'train'))

        for cl in features_train.columns[1:]:
            if cl != 'car_id':
                train_df.drop([cl], axis=1, errors='ignore', inplace=True)
            
        train_df = pd.merge(train_df, features_train, on=['car_id', 'car_id'])
        print('train') 
        print(features_train.columns)
    except:
        print('FAIL: train')
    try:
        features_test = pd.read_csv('~/pml7/konkurs/output/' + csv_file.replace('REPLACE', 'test'))

        for cl in features_train.columns[1:]:
            if cl != 'car_id':
                test_df.drop([cl], axis=1, errors='ignore', inplace=True)
            
        test_df = pd.merge(test_df, features_test, on=['car_id', 'car_id'])
        print('test')
        print(features_test.columns)
    except:
        print('FAIL: test')
        
    return train_df, test_df
    

def my_train_test_split(df_X, df_y, train_N):

    import pandas as pd
    
    df_total = pd.merge(df_X, df_y, left_index=True, right_index=True)
    test_car_ids = train_N['car_id'].values
    X = df_X.columns[1:] #bez car_id
    #print(X)
    y = df_y.name
    #print(y)
    
    df_train = df_total[~df_total['car_id'].isin(test_car_ids)]
    df_test = df_total[df_total['car_id'].isin(test_car_ids)]

    X_train = df_train[X]
    X_test = df_test[df_X.columns[0:]] #z car_id
    y_train = df_train[y]
    y_test = df_test[y]
    
    return X_train, X_test, y_train, y_test

    
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def save_model(name, train_df, test_df, used_feats, my_model, mae=0.0, r2=0.0, 
               details=None, plot_learning_curve = None, plt_hist = None, 
               kaggle=False):
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    
    import json
    import pandas as pd
    import numpy as np

    y_pred = my_model.predict(test_df[used_feats])

    test_df = test_df.drop(['price_value_pln_log_transl','price_value_pln','price_value'], axis=1, errors='ignore')
    
    test_df['price_value_pln_log_transl'] = y_pred
    test_df['price_value_pln'] = np.exp(test_df['price_value_pln_log_transl'] + 10)
    test_df['price_value'] = test_df.apply(lambda row: 
                                       row.price_value_pln if row.price_currency == 'PLN' 
                                       else row.price_value_pln / row.pln_to_eur, axis=1) 
   
    global_min = train_df['price_value'].min()
    test_df.loc[test_df['price_value'] <= global_min, ['price_value']] = global_min

    
    file_name = 'output_model/' \
        + ('' if mae==0.0 else "mae_{:.5f}_".format(mae).replace('.', '_')) \
        + ('' if r2==0.0 else "r2_{:.5f}_".format(r2).replace('.', '_')) \
        + type(my_model).__name__ + '_' \
        + name + '.csv'
        
    test_df[ ['car_id', 'price_value'] ].to_csv(file_name, index=False)
    test_df = test_df.drop(['price_value_pln_log_transl','price_value_pln','price_value'], axis=1, errors='ignore')

    message = json.dumps(details, ensure_ascii=False, indent=4)
    with open(file_name.replace('.csv','.txt'), 'w') as outfile: 
        outfile.write(message)
    
    if plot_learning_curve is not None:
        plot_learning_curve.get_figure().savefig(file_name.replace('.csv','_lc.png'), bbox_inches='tight')

    if plt_hist is not None:
        plt_hist.savefig(file_name.replace('.csv','_hist.png'), bbox_inches='tight')

    if kaggle and details is not None:
        print(message)
        kaggle_api.competition_submit(file_name.replace('~/pml7/konkurs/', ''), message, 'dw-car-price-prediction')
        leaderboard = kaggle_api.competition_view_leaderboard('dw-car-price-prediction')
        print(pd.DataFrame(leaderboard['submissions'])[['teamName', 'submissionDate', 'score']].head())

def from_ln_pln_trans_to_predict(X_values, y_values, full_df):
    
    import pandas as pd
    import numpy as np
    
    X_values = pd.merge(X_values, full_df[['car_id', 'price_currency', 'pln_to_eur']], on=['car_id'], how='inner')[['car_id', 'price_currency', 'pln_to_eur']]
    X_values['price_value_pln_log_transl'] =  y_values.values

    X_values['price_value_pln'] = np.exp(X_values['price_value_pln_log_transl'] + 10)
    X_values['price_value'] = X_values.apply(lambda row: 
                                       row.price_value_pln if row.price_currency == 'PLN' 
                                       else row.price_value_pln / row.pln_to_eur, axis=1) 

    return X_values['price_value']
        

def how_many_levels(df, skipColumnsStart=-1, skipColumnsStop=-1, nameContains=""):
    
    import collections
    from ftfy import fix_text
    import re
    
    index = -1
    skipColumnsStop = 1_000_000 if skipColumnsStop==-1 else skipColumnsStop
    for column in df.columns[1:]:
        index += 1
        if index >= skipColumnsStart and index <= skipColumnsStop and (nameContains=="" or re.search(nameContains, column)):
            nestedDict = False
            try:
                k = dict(collections.Counter(df[column]).most_common(7))
            except:
                cl = df[column].map(lambda row_item: fix_text(json.dumps(row_item,ensure_ascii=False).replace("  ", "").replace("\\n", "").replace("\xa0", " ")))
                k = dict(collections.Counter(cl).most_common(3))
                nestedDict = True
            rows = len(df.index)
            nas = rows - df[column].isna().sum()
            print(str(index) + ': ' + ('dict-' if nestedDict else '') + column + " [{:.2f}%=".format(nas/rows*100) + str(nas) + '/' + str(rows) + ']: ')
            print(list(k.keys()))
            print('==================================================================================================================')
            
            
def compare_and_combine(ft_pl, ft_ang, df):
    
#    import pandas as pd
    import numpy as np

    print(df[ft_pl].value_counts())    
    print(df[ft_ang].value_counts())
    new_cl = 'new_' + ft_pl
    df[new_cl] = np.where(df[ft_pl].isnull(), df[ft_ang], df[ft_pl] )
    print(df[[ft_pl, ft_ang, new_cl]])
    print(df[new_cl].value_counts())
    print('==========================================================================================================')

def compare_and_combine_2(ft_pl_ft_ang, df):
    ft_pl, ft_ang = ft_pl_ft_ang.split(' / ', 1)
    print(ft_pl)
    print(ft_ang)
    compare_and_combine(ft_pl, ft_ang, df)


def ln_exp_obj(y_true, y_pred):
    x = y_true - y_pred
    exp_2x = np.exp(2*x)
    grad = (exp_2x - 1) / (exp_2x + 1)
    hess = (4 * exp_2x) / (exp_2x + 1)**2
    return grad, hess



def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    #rename cols
    strata2 = []
    
    for cl in strata:                             #df.columns[1:]:
        cl_new = cl.replace('-', '_')
        df.rename({cl: cl_new}, axis=1, inplace=True)
        strata2.append(cl_new)

    strata = strata2
    
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"
            
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    
    return stratified_df



def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling 
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n

def num_to_range_categories(df, var, n_cat=100):
    import numpy as np
    import pandas as pd

    percentiles = np.array(np.linspace(.00, 1, n_cat + 0, 0))

    bins = np.array(df[var].quantile( percentiles ))

    names = []
    for name in percentiles:
        names.append("perc_{:.2f}".format(name).replace('0.', '').replace('.', ''))

    #print(bins)
    #print(names)
    d = dict(enumerate(names, 1))
    #print(d)
    new_column = 'new_cat_' + var.replace('-', '_')
    df[new_column] = np.vectorize(d.get)(np.digitize(df[var], bins))
    df.loc[df[var].isnull(), new_column] = '<NA>'
    print(df[new_column].value_counts())
    print(sorted(df[new_column].unique()))
    print(df[new_column].isna().sum())
