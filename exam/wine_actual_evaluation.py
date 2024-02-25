#%%
import pandas as pd
import numpy as np

wine_data = pd.read_csv("wine_with_evaluation.csv", index_col=0, header=0).iloc[:, 0:13]

def wine_evaluatiuon_Y1(
	wine_property: pd.DataFrame
)->np.ndarray:
	"""
	ワイン評価関数（Y1）
	"""
	
	#wine_data = pd.read_csv("wine_with_evaluation.csv", index_col=0, header=0).iloc[:, 0:13]
	
	scaled_property = (wine_property - wine_data.mean())/wine_data.std()

	score = - (scaled_property.values**2).sum(axis=1)

	return score




def wine_evaluatiuon_Y2(
	wine_property: pd.DataFrame
)->np.ndarray:

	#wine_data = pd.read_csv("wine_with_evaluation.csv", index_col=0, header=0).iloc[:, 0:13]

	scaled_property = (wine_property - wine_data.mean())/wine_data.std()
	n = wine_data.shape[1]

	score = -10*n - (scaled_property.values**2 - 10*np.cos(2*np.pi*scaled_property.values)).sum(axis=1)

	return score