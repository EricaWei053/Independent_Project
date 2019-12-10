import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
#repace it with MulticoreTSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as mp
import openTSNE


def _visual(categories, finalDf, colors, fn, title):

	#Visualize 2D Projection
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Component 1', fontsize = 30)
	ax.set_ylabel('Component 2', fontsize = 30)
	ax.set_title(title, fontsize = 20)
	
	for target, color in zip(categories, colors):
	    indicesToKeep = finalDf['target'] == target
	    ax.scatter(finalDf.loc[indicesToKeep, 'component 1']
	               , finalDf.loc[indicesToKeep, 'component 2']
	               , c = color
	               , s = 3)
	ax.legend(categories)
	ax.grid()
	mp.savefig(fn)


def plot_PCA(data, targets, categories, colors, fn): 
	#Standarlize data 
	# Separating out the features
	x = data
	# Separating out the target
	y = targets
	# Standardizing the features
	x = StandardScaler().fit_transform(x)

	#Projection to 2D
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(x)

	principalDf = pd.DataFrame(data = principalComponents
							, columns = ['component 1', 'component 2'])

	targets = pd.DataFrame(data = y, columns = ['target'])
	finalDf = pd.concat([principalDf, targets], axis = 1)


	_visual(categories, finalDf, colors, fn, "linear PCA")


def kernal_PCA(data, targets, categories, colors, fn): 
	# Separating out the features
	x = data
	# Separating out the target
	y = targets
	# Standardizing the featuress
	x = StandardScaler().fit_transform(x)
	#Gamma need to be obtained via hyperparameter tuning techniques like Grid Search. 
	gamma = 10

	kernel = "sigmoid"

	kpca = KernelPCA(n_components=2, n_jobs=4, kernel=kernel, gamma=gamma)
	X_kpca = kpca.fit_transform(x)
	principalDf = pd.DataFrame(data = X_kpca
							, columns = ['component 1', 'component 2'])

	targets = pd.DataFrame(data = y, columns = ['target'])
	finalDf = pd.concat([principalDf, targets], axis = 1)
	
	_visual(categories, finalDf, colors, fn, "kernel: "  + kernel +  " PCA")



def tSNE(data, targets, categories, colors, fn): 
	# Separating out the features
	x = data
	# Separating out the target
	y = targets
	# Standardizing the features
	x = StandardScaler().fit_transform(x)

	#default n_components=2
	X_2d = TSNE(n_components=2).fit_transform(x)
	principalDf = pd.DataFrame(data = X_2d
					  ,columns = ['component 1', 'component 2'])

	targets = pd.DataFrame(data = y, columns = ['target'])
	finalDf = pd.concat([principalDf, targets], axis = 1)


	_visual(categories, finalDf, colors, fn, " tSNE ")


	
"""
Use OPENTSNE

@article {Poli{\v c}ar731877,
    author = {Poli{\v c}ar, Pavlin G. and Stra{\v z}ar, Martin and Zupan, Bla{\v z}},
    title = {openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding},
    year = {2019},
    doi = {10.1101/731877},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2019/08/13/731877},
    eprint = {https://www.biorxiv.org/content/early/2019/08/13/731877.full.pdf},
    journal = {bioRxiv}
}

"""

def scale_data(vis_x, vis_y): 
	"""
	Scale data to 0-1 range on both axis 
	"""
	scaler1 = MinMaxScaler()
	scaler1.fit(vis_x.reshape(-1, 1))
	norm_x = scaler1.transform(vis_x.reshape(-1, 1)).reshape(vis_y.shape) 

	scaler2 = MinMaxScaler()
	scaler2.fit(vis_y.reshape(-1, 1))
	norm_y = scaler2.transform(vis_y.reshape(-1, 1)).reshape(vis_y.shape) 

	return norm_x, norm_y


def open_tSNE(X, grad_X, Y, cate_0, fn):
	#model.embedding_ = model.embedding_.astype(np.float32, order='A')
	#Embedding original data.
	embedding = openTSNE.TSNE().fit(X) 
	vis_x = embedding[:, 0]
	vis_y = embedding[:, 1]
	#print(vis_x.shape)
	#print(vis_y.shape)

	"""
	Scale data to 0-1 range on both axis 
	"""
	norm_x, norm_y = scale_data(vis_x, vis_y)
	print(norm_x.shape)
	print(norm_y.shape)
	

	#Plot tsne embedding. 
	n = len(cate_0)
	#plt.scatter(vis_x, vis_y, c=Y, cmap=plt.cm.get_cmap("hot"), s = 3)
	#normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
	plt.scatter(norm_x, norm_y, c=Y ,cmap=plt.cm.get_cmap("hot"), s = 3)
	plt.colorbar()

	'''  
	#Embed new points(grad vectors) into the existing embedding.
	grad_embedding = embedding.transform(grad_X)
	vis_gx = grad_embedding[:, 0]
	vis_gy = grad_embedding[:, 1]

	#Plot new points from tsne. 
	plt.quiver(vis_x, vis_y, vis_gx, vis_gy, angles='xy') #, scale_units='xy', scale=1) 
	'''
	mp.savefig(fn)
	plt.close()

