
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVR # SVR モデルの構築に使用
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct, ConvergenceWarning, ExpSineSquared
warnings.filterwarnings('ignore')




def generate_gaussian_process_regressor(
		x_data: pd.DataFrame, 
		y_data: pd.DataFrame,
		regression_method: str,
		kernel_number: int = 2,
		fold_number:int =5, 
		)->GaussianProcessRegressor:
	#fold_number = 2  # クロスバリデーションの fold 数
	#kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

	autoscaled_y = (y_data - y_data.mean()) / y_data.std()
	autoscaled_x = (x_data - x_data.mean()) / x_data.std()
	
	# カーネル 11 種類
	kernels = [ConstantKernel() * DotProduct() + WhiteKernel(), #0
			ConstantKernel() * RBF() + WhiteKernel(), 
			ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
			ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel(),
			ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
			ConstantKernel() * Matern(nu=1.5) + WhiteKernel(), #5
			ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
			ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
			ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
			ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
			ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct(), #10
			ConstantKernel() * Matern(nu=1.5) * ExpSineSquared() + WhiteKernel(),
			ConstantKernel() * Matern(nu=1.5) * ExpSineSquared(periodicity=10) + WhiteKernel(),
			ConstantKernel() * Matern(nu=1.5) * ExpSineSquared(periodicity=100) + WhiteKernel(),
			ConstantKernel() * Matern(nu=1.5) * ExpSineSquared(periodicity=0.1) + WhiteKernel(), #14
			ConstantKernel() * Matern(nu=1.5) * ExpSineSquared(length_scale=5) + WhiteKernel(), #15
			ConstantKernel() * Matern(nu=2.5) + ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel(),
			ConstantKernel() * Matern(nu=2.5) + ConstantKernel() * RBF(np.ones(x_data.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
			ConstantKernel() * Matern(nu=3.5) + WhiteKernel(), 
			ConstantKernel() * Matern(nu=3.5) + WhiteKernel() + ConstantKernel() * DotProduct(),]
	
		# モデル構築
	if regression_method == 'gpr_one_kernel':
		selected_kernel = kernels[kernel_number]
		model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)
	elif regression_method == 'gpr_kernels':
		# クロスバリデーションによるカーネル関数の最適化
		cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
		r2cvs = [] # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
		for index, kernel in enumerate(kernels):
			print(index + 1, '/', len(kernels))
			model = GaussianProcessRegressor(alpha=0, kernel=kernel)
			estimated_y_in_cv = np.ndarray.flatten(cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation))
			estimated_y_in_cv = estimated_y_in_cv * y_data.std(ddof=1) + y_data.mean()
			r2cvs.append(r2_score(y_data, estimated_y_in_cv))
			print(r2_score(y_data, estimated_y_in_cv))
		optimal_kernel_number = np.where(r2cvs == np.max(r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
		optimal_kernel = kernels[optimal_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
		print('クロスバリデーションで選択されたカーネル関数の番号 :', optimal_kernel_number)
		print('クロスバリデーションで選択されたカーネル関数 :', optimal_kernel)
	
		# モデル構築
		model = GaussianProcessRegressor(alpha=0, kernel=optimal_kernel) # GPR モデルの宣言

	return model


def generate_gaussian_support_vector_regressor(x: pd.DataFrame, y: pd.DataFrame, number_of_test_samples:float = 5):
	number_of_test_samples = 5  # テストデータのサンプル数
	fold_number = 10  # クロスバリデーションの fold 数
	nonlinear_svr_cs = 2 ** np.arange(-5, 11, dtype=float) # SVR の C の候補
	nonlinear_svr_epsilons = 2 ** np.arange(-10, 1, dtype=float) # SVR の ε の候補
	nonlinear_svr_gammas = 2 ** np.arange(-20, 11, dtype=float) # SVR のガウシアンカーネルの γ の候補
	
	# ランダムにトレーニングデータとテストデータとに分割
	# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
	if number_of_test_samples == 0:
		x_train = x.copy()
		x_test = x.copy()
		y_train = y.copy()
		y_test = y.copy()
	else:
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
															random_state=99)

	# 標準偏差が 0 の特徴量の削除
	deleting_variables = x_train.columns[x_train.std() == 0]
	x_train = x_train.drop(deleting_variables, axis=1)
	x_test = x_test.drop(deleting_variables, axis=1)

	# オートスケーリング
	autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
	autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()

	# C, ε, γの最適化
	# 分散最大化によるガウシアンカーネルのγの最適化
	variance_of_gram_matrix = []
	autoscaled_x_train_array = np.array(autoscaled_x_train)
	for nonlinear_svr_gamma in nonlinear_svr_gammas:
		gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_train_array[:, np.newaxis] - autoscaled_x_train_array) ** 2).sum(axis=2))
		variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
	optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]

	cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
	# CV による ε の最適化
	r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
	for nonlinear_svr_epsilon in nonlinear_svr_epsilons:
		model = SVR(kernel='rbf', C=3, epsilon=nonlinear_svr_epsilon, gamma=optimal_nonlinear_gamma)
		autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=cross_validation)
		r2cvs.append(r2_score(y_train, autoscaled_estimated_y_in_cv * y_train.std() + y_train.mean()))
	optimal_nonlinear_epsilon = nonlinear_svr_epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

	# CV による C の最適化
	r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
	for nonlinear_svr_c in nonlinear_svr_cs:
		model = SVR(kernel='rbf', C=nonlinear_svr_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)
		autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=cross_validation)
		r2cvs.append(r2_score(y_train, autoscaled_estimated_y_in_cv * y_train.std() + y_train.mean()))
	optimal_nonlinear_c = nonlinear_svr_cs[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

	# CV による γ の最適化
	r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
	for nonlinear_svr_gamma in nonlinear_svr_gammas:
		model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=nonlinear_svr_gamma)
		autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=cross_validation)
		r2cvs.append(r2_score(y_train, autoscaled_estimated_y_in_cv * y_train.std() + y_train.mean()))
	optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

	return SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)  # SVR モデルの宣言