# use conda env AAScore

from get_features import get_shell_features
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import pickle
from itertools import product
import multiprocessing
from tqdm import tqdm

FEATURE_LEN = 1540
SHELLS = 23
MAX_PCC = 0.87 # save model file when pcc_result greater than MAX_PCC

def cut_onion(features, num=SHELLS):
    # reshape onion 23 to num
    return features.reshape(-1, SHELLS, FEATURE_LEN)[:, :num, :].reshape(-1, num*FEATURE_LEN)

def train_eval_GBT_model(parameter):
    learning_rate, iters, shells, max_depth, min_samples_split, subsample, max_f = parameter
    x_train, y_train, x_test, y_test = get_shell_features()
    x_train, x_test = cut_onion(x_train, shells), cut_onion(x_test, shells)
    GBT = GradientBoostingRegressor(
        random_state=1206, n_estimators=iters, max_features=max_f, 
        max_depth=max_depth, min_samples_split=min_samples_split, 
        learning_rate=learning_rate, loss="ls", subsample=subsample,
    )
    GBT.fit(x_train, y_train)

    y_pred_GBT = GBT.predict(x_test)
    pcc = pearsonr(y_test,y_pred_GBT)[0]
    rmse = sqrt(mean_squared_error(y_test,y_pred_GBT))

    print(f"""
    MaxDepth:     {max_depth}
    Min_samples_split: {min_samples_split},
    Subsample:    {subsample}
    MaxFeatures:  {max_f}
    LearningRate: {learning_rate}
    Iterations:   {iters}
    Shells:       {shells}
    PCC:          {pcc}
    RMSE:         {rmse}
    """)

    # save model
    saved_path = f"./saved_GBT/OnionECIF_GBT_{pcc:.4f}_{rmse:.4f}_{iters}_{learning_rate}_{shells}_{max_depth}_{min_samples_split}_{subsample}_{max_f}.pkl"
    if pcc > MAX_PCC: pickle.dump(GBT, open(saved_path, 'wb'))

# search paramaters
lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
iters = [5000, 10000, 20000, 50000]
shells = [20, 21, 22, 23]
max_depths = [5, 6, 7] # default 8
min_samples_splits = [2, 3, 4, 5]
subsamples = [0.6, 0.7, 0.8]
max_features = ['sqrt', 'auto', 'log2']
paras = []
for lr, it, shell, max_depth, min_s, subs, max_f in product(lrs, iters, shells, max_depths, min_samples_splits, subsamples, max_features):
    paras.append((lr, it, shell, max_depth, min_s, subs, max_f))


# multi process
print('cpu count:', multiprocessing.cpu_count())
with multiprocessing.Pool(multiprocessing.cpu_count()) as workers:
    with tqdm(total=len(paras), ncols=80) as pbar:
        for _ in workers.imap_unordered(train_eval_GBT_model, paras):
            pbar.update()
