import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict


import wine_actual_evaluation
import model_generator

dataset = pd.read_csv('generated_wine_candidates.csv', index_col=0, header=0)
x_prediction = pd.read_csv('generated_wine_candidates.csv', index_col=0, header=0)

regression_method = "gpr_one_kernel"  # gpr_one_kernel', 'gpr_kernels'
acquisition_function = 'PTR'  # 'PTR', 'PI', 'EI', 'MI'

fold_number = 10  # クロスバリデーションの fold 数
kernel_number = 7  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
target_range = [-40, -130]  # PTR
relaxation = 0.01  # EI, PI
delta = 10 ** -6  # MI

x = dataset.iloc[1000:4000]  # 説明変数
# 標準偏差が 0 の特徴量の削除
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
y = wine_actual_evaluation.wine_evaluatiuon_Y2(x)  # 目的変数
#x_prediction = x_prediction.drop(deleting_variables, axis=1)

autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()
#print(x)
#print(y)
#print(y.max())

model = model_generator.generate_gaussian_process_regressor(
    x, y, 
    regression_method, 
    kernel_number= kernel_number, 
    fold_number=fold_number)
#model = GaussianProcessRegressor(alpha=0, kernel=kernels[0]) 

model.fit(autoscaled_x, autoscaled_y)  # モデル構築

cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)  # y の推定
estimated_y_in_cv = autoscaled_estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
estimated_y_in_cv = pd.DataFrame(estimated_y_in_cv, index=x.index, columns=['estimated_y'])

print('r^2 in cross-validation :', r2_score(y, estimated_y_in_cv))
print('RMSE in cross-validation :', mean_squared_error(y, estimated_y_in_cv, squared=False))
print('MAE in cross-validation :', mean_absolute_error(y, estimated_y_in_cv))

# クロスバリデーションにおける実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.scatter(y, estimated_y_in_cv.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y.max(), estimated_y_in_cv.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y.min(), estimated_y_in_cv.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
        [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.show()  # 以上の設定で描画

