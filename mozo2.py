def add_mpvp_min_max(mpvp_col, total):
    dictm = total[['new_cat_price_value', 'price_value_min', 'price_value_max']].drop_duplicates().to_dict()
    def get_from_dict(dictm, col, perc_val):
        index = [i for i, v in dictm.get('new_cat_price_value').items() if v == perc_val][0]
        return dictm.get(col).get(index)

    total[mpvp_col + '_' + 'price_value_min'] = total.apply(lambda row: get_from_dict(dictm, 'price_value_min', row['mpvp1_']), axis=1)
    total[mpvp_col + '_' + 'price_value_max'] = total.apply(lambda row: get_from_dict(dictm, 'price_value_max', row['mpvp1_']), axis=1)

def train_and_submit(train_70, train_30, test, feats, global_min, 
                     model, params, shift = 0, digitize=None, 
                     file_sufix= '', opis='', subfolder='', 
                     learning_curve = False, add_model_column_min = -1, total = None,  
                     save_min=7000, kaggle_min=-1):
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import r2_score, mean_absolute_error
    from scikitplot.estimators import plot_learning_curve

    train_70[feats]
    train_30[feats]
    test[['car_id', *feats]]

    
    time_model_start = datetime.now().strftime("%H:%M:%S")
    
#     shift = 0 #10
    
    X_train = train_70[ feats ].values
    y_train = train_70['price_value'].values
    y_train_log = np.log(y_train) - shift

    if params is not None: 
        model = model(**params)
    model.fit(X_train, y_train_log)

    ## check 
    y_train_pred_log = model.predict(train_30[ feats ].values)
    y_train_pred = np.exp(y_train_pred_log + shift) 
    if digitize is not None: y_train_pred = digitalize_prediction(y_train_pred, train_30, digitize)
    else: y_train_pred[ y_train_pred < global_min] = global_min

    mae = mean_absolute_error(train_30['price_value'].values, y_train_pred)
    r2 = r2_score(train_30['price_value'].values, y_train_pred)
    print(f'mea: {mae}')
    print(f'r2: {r2}')

    ## add_model_column
    model_cl = 'model_' + type(model).__name__ + "_{:.5f}_".format(mae).replace('.', '_') + file_sufix
    if mae <= add_model_column_min and total is not None: 
        X_total = total[feats].values
        y_pred_log = model.predict(X_total)
        y_pred = np.exp(y_pred_log + shift)
        if digitize is not None: y_pred = digitalize_prediction(y_pred, total, digitize)
        else: y_pred[ y_pred < global_min] = global_min
        total[model_cl] = y_pred

    learning_curve_plot = None
    if learning_curve: 
        print("Learning curve for: " + type(model).__name__)
        learning_curve_plot = plot_learning_curve(model, X_train, y_train, cv=3, random_state=0, shuffle=True)

    time_model_end = datetime.now().strftime("%H:%M:%S")

#     try:
    if mae <= save_min:
        save_model(file_sufix, subfolder, test, feats, global_min, 
                   model, shift, digitize, mae, r2,
                   details = {"mean_absolute_error" : mae, "r2_score" : r2, 
                              "vars_count": len(feats),
                              "vars": ("'"+ "', '".join(list(feats))+"'"), 
                              "best_params" : params,
                              "model_start" : time_model_start, "model_end" : time_model_end, 
                              "opis" : opis}, 
                   learning_curve_plot = learning_curve_plot if learning_curve else None, hist_plt=None,
                   kaggle=(mae < kaggle_min))
#     except: 
#         print('Error on save!')
        
    return mae, r2, model, model_cl, learning_curve_plot

def digitalize_prediction(y_pred, train_all, digitize):
    import pandas as pd

    result = pd.DataFrame(pd.np.column_stack([train_all[[digitize + '_' + 'price_value_min', digitize + '_' + 'price_value_max']], y_pred]))
    result = result.rename(columns={0: "price_value_min", 1: "price_value_max", 2: "__model__"})         

    result.loc[result['__model__'] < result['price_value_min'], '__model__'] = result.loc[result['__model__'] < result['price_value_min'], 'price_value_min']
    result.loc[result['__model__'] > result['price_value_max'], '__model__'] = result.loc[result['__model__'] > result['price_value_max'], 'price_value_max']
    
    return result['__model__'].values


def save_model(file_sufix, subfolder, test, feats, global_min, model, shift = 0, digitize=False, mae=0.0, r2=0.0,
               details='', learning_curve_plot=None, hist_plt=None, kaggle=None):
    
#     from kaggle.api.kaggle_api_extended import KaggleApi
    
    import json
    import pandas as pd
    import numpy as np

    test['car_id']
    
    ## predict test
    X_test = test[feats].values
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log + shift)
    if digitize is not None: y_pred = digitalize_prediction(y_pred, test, digitize)
    else: y_pred[ y_pred < global_min] = global_min
    test['price_value'] = y_pred
    
    file_name = 'output_model/' + subfolder + '/' \
        + ('' if mae==0.0 else "mae_{:.5f}_".format(mae).replace('.', '_')) \
        + ('' if r2==0.0 else "r2_{:.5f}_".format(r2).replace('.', '_')) \
        + type(model).__name__ + '_' \
        + file_sufix + '.csv'
        
    test[['car_id','price_value']].to_csv(file_name, index=False)

    message = json.dumps(details, ensure_ascii=False, indent=4)
    with open(file_name.replace('.csv','.txt'), 'w') as outfile: 
        outfile.write(message)
    
    if learning_curve_plot is not None:
        learning_curve_plot.get_figure().savefig(file_name.replace('.csv','_lc.png'), bbox_inches='tight')

    if hist_plt is not None:
        hist_plt.savefig(file_name.replace('.csv','_hist.png'), bbox_inches='tight')

    if kaggle and details is not None:
        try:
            kaggle_api = KaggleApi()
            kaggle_api.authenticate()

            print(message)
            kaggle_api.competition_submit(file_name.replace('~/pml7/konkurs/', ''), message, 'dw-car-price-prediction')
            leaderboard = kaggle_api.competition_view_leaderboard('dw-car-price-prediction')
            print(pd.DataFrame(leaderboard['submissions'])[['teamName', 'submissionDate', 'score']].head())
        except: print('Kaggle API fail')

            
def plot_cat(df, var):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    plt.title('Histogram')
    sns.countplot(df[var], palette=("cubehelix"))

    plt.subplot(1,2,2)
    plt.title(var + ' vs Price')
    sns.boxplot(x=df[var], y=df['price_value_log'], palette=("cubehelix"))

    plt.show()

import numpy as np
def print_plots_by_type(total, type = [np.object, bool], min_c = 9.0, max_c = 100., max_values=30): 
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    index = 0
    for cl in total.columns[2:]:
        index += 1
        if total[cl].dtype in type:
            rows = len(total.index)
            nas = rows - total[cl].isna().sum()
            nas_perc = nas/rows*100
            values = len(total[cl].value_counts())
            print(str(index) + ': ' + cl + " [{:.2f}%=".format(nas_perc) + str(nas) + '/' + str(rows) + ' #' + str(values) + ']: ')
            if max_c >= nas_perc >= min_c: 
                if 1 < values < max_values:
                    print(total[cl].value_counts())
                    plot_cat(total, cl)
                    plt.clf()
                    plt.cla()
                    plt.close()

def save_dataframe_total(total):
    from datetime import datetime
    
    print(total.info(verbose=True))
    today = datetime.today() 
    cvs_file_name = f'newest_total_{today:%Y%m%d_%H_%M.h5}'
    total.to_csv('~/pml7/konkurs/output/' + cvs_file_name + '.csv', index=False) 

def replace_feature(current_vars, old, new):
    current_vars = np.array(current_vars)
    current_vars = list(current_vars)
    current_vars.remove(old)
    current_vars.append(new)
    return np.sort(current_vars)

# replace_feature(['a', 'b', 'c'], 'b', 'e')

import random

def mutate_rand_feature(current_vars, all_vars, added_so_far, removed_so_far):
    
    def diff(a, b):
        return np.concatenate([np.setdiff1d(a,b), np.setdiff1d(b,a)])
    
#     vars_to_draw_from = diff(current_vars, all_vars)
    vars_to_draw_from = [x for x in all_vars if x not in current_vars] #zmienna ktrej nie ma w obecnych
#     vars_to_draw_from = diff(vars_to_draw_from, added_so_far)
    vars_to_draw_from = [x for x in vars_to_draw_from if x not in added_so_far]
#     vars_to_draw_from = diff(vars_to_draw_from, removed_so_far)
    vars_to_draw_from = [x for x in vars_to_draw_from if x not in removed_so_far]
    
    last_trial = False
    if len(vars_to_draw_from) == 1: last_trial = True

    print(vars_to_draw_from)
    added = np.random.choice(vars_to_draw_from, 1)[0]
    removed = np.random.choice(np.setdiff1d(current_vars, added), 1)[0]
    print(removed)
    print(added)
    current_vars = replace_feature(current_vars, removed, added)
    print(current_vars)
    
    return current_vars, added, np.append(added_so_far, added), removed, np.append(removed_so_far, removed), last_trial

# added_so_far = []
# removed_so_far = []
# current_vars = ['a', 'b', 'c', 'd', 'e']
# all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
# mutate_rand_feature(current_vars, all_vars, added_so_far, removed_so_far)


def mutate_rand_feature_pairs(current_vars, all_vars, tested_pairs):
    
    def check_if_pair_occured(removed, added, tested_pairs):
        for removed_p, added_p in tested_pairs:
            if removed == removed_p and added == added_p: 
                print('bylo: '+ removed_p + ' -> ' + added_p)
                return True
        return False
   
    vars_to_draw_from = [x for x in all_vars if x not in current_vars] #zmienna ktrej nie ma w obecnych

    print('draw next pair')
    added = ''
    removed = ''
    while True:
        added = np.random.choice(vars_to_draw_from, 1)[0]
        removed = np.random.choice(np.setdiff1d(current_vars, added), 1)[0]
        if not check_if_pair_occured(removed, added, tested_pairs): break
    
    tested_pairs.append((removed,added))
    
    set_size = len(current_vars)
    last_trial = True if len(current_vars) >= set_size * (set_size - 1) else False

#     print(removed)
#     print(added)
    current_vars = replace_feature(current_vars, removed, added)
#     print(current_vars)
    
    return current_vars, added, removed, tested_pairs, last_trial


def get_kaggle_board():
    import pandas as pd

    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    leaderboard = kaggle_api.competition_view_leaderboard('dw-car-price-prediction')
    print(pd.DataFrame(leaderboard['submissions'])[['teamName', 'submissionDate', 'score']].head())


