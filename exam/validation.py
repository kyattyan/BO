import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def cross_validation(
        x: pd.DataFrame, 
        actual_y : pd.DataFrame, 
        model,
        fold_number: int,
        graph_plot = False
        )->None:
    
    autoscaled_y = (actual_y - actual_y.mean()) / actual_y.std()
    autoscaled_x = (x - x.mean()) / x.std()
    
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)  # y の推定
    estimated_y_in_cv = autoscaled_estimated_y_in_cv * actual_y.std() + actual_y.mean()  # スケールをもとに戻す
    estimated_y_in_cv = pd.DataFrame(estimated_y_in_cv, index=x.index, columns=['estimated_y'])

    print('r^2 in cross-validation :', r2_score(actual_y, estimated_y_in_cv))
    print('RMSE in cross-validation :', mean_squared_error(actual_y, estimated_y_in_cv, squared=False))
    print('MAE in cross-validation :', mean_absolute_error(actual_y, estimated_y_in_cv))

    if graph_plot == True:
        plt.rcParams['font.size'] = 18
        plt.scatter(actual_y, estimated_y_in_cv.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
        y_max = max(actual_y.max(), estimated_y_in_cv.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
        y_min = min(actual_y.min(), estimated_y_in_cv.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
        plt.xlabel('actual y')  # x 軸の名前
        plt.ylabel('estimated y')  # y 軸の名前
        plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
        plt.show()  # 以上の設定で描画
    return 