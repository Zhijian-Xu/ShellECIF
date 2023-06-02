from get_features import get_shell_features
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from math import sqrt
import matplotlib.pyplot as plt

model_file = f'./saved_GBT/saved_GBT.pkl'
with open(model_file, 'rb') as f:
    GBT = pickle.load(f)

SHELLS = 23
FEATURE_LEN = 1540

def cut_onion(features, num=SHELLS):
    # reshape shells from SHELLS to num
    return features.reshape(-1, SHELLS, FEATURE_LEN)[:, :num, :].reshape(-1, num*FEATURE_LEN)

shells_num = GBT.n_features_ // FEATURE_LEN
x_train, y_train, x_test, y_test = get_shell_features()
x_train, x_test = cut_onion(x_train, shells_num), cut_onion(x_test, shells_num)

y_pred_GBT = GBT.predict(x_test)

pcc = pearsonr(y_test,y_pred_GBT)[0]
rmse = sqrt(mean_squared_error(y_test,y_pred_GBT))
mae = mean_absolute_error(y_test,y_pred_GBT)
spear, _ = spearmanr(y_test,y_pred_GBT)

print(f'{pcc:.4f}')
print(f'{rmse:.4f}')
print(f'{mae:.4f}')
print(f'{spear:.4f}')

# 画图
plt.scatter(y_test, y_pred_GBT, marker='o', s=5)
plt.plot([0, 14], [0, 14], color=(0.6,0.6,0.6), linestyle='-')
plt.xlabel('Experimental')
plt.ylabel('Predicted')
plt.title('Performance on CASF-2016 core set')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.xlim(xmax=14)
plt.ylim(ymax=14)
plt.text(0.05, 0.95, f'Pearson = {pcc:.4f}', transform=plt.gca().transAxes, ha='left', va='top')
plt.text(0.05, 0.9, f'RMSE = {rmse:.4f}', transform=plt.gca().transAxes, ha='left', va='top')
plt.text(0.05, 0.85, f'MAE = {mae:.4f}', transform=plt.gca().transAxes, ha='left', va='top')
plt.text(0.05, 0.8, f'Spearman = {spear:.4f}', transform=plt.gca().transAxes, ha='left', va='top')
plt.savefig('./performance.png', dpi=300)
plt.show()