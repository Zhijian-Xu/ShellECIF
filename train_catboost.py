# use conda env  lightgbm
import time
from get_features import get_shell_features
from catboost import CatBoostRegressor

x_train, y_train, x_test, y_test = get_shell_features()

print('train shape:')
print(x_train.shape[0], x_test.shape[0])
print(x_train.shape[1], x_test.shape[1])

LEARNING_RATE = 0.1
ITERATIONS = 50000

model = CatBoostRegressor(iterations=ITERATIONS,
                          learning_rate=LEARNING_RATE,
                           task_type="GPU",
                           devices='0')

t_start = time.time()
print('LEARNING_RATE', LEARNING_RATE)
print('ITERATIONS', ITERATIONS)
print('start training:', time.asctime( time.localtime(time.time()) ))

model.fit(
    x_train, y_train,
    verbose=False, 
    eval_set=(x_test, y_test), 
    # early_stopping_rounds=300,
    # plot=True,
    plot_file='./catboost_offline_plotfile'
)

print('train finished', end='\t')
print("cost: {:.2f}seconds".format(time.time() - t_start))

ypred = model.predict(x_test)

from math import sqrt
import numpy as np
from scipy import stats

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse
def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse
def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

pcc = pearson(y_test, ypred)

print('rmse:', rmse(y_test, ypred))
print('pearson:', pcc)

# save model
model.save_model(f'./saved_catboost/saved_catboost_{LEARNING_RATE}_{ITERATIONS}_{pcc}.model')
