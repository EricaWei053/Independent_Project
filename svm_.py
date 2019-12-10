import matplotlib
import numpy as np 
from sklearn.metrics import pairwise 
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import scipy.stats as ss
from sklearn import metrics
from sklearn.svm import libsvm
import sys
import copy
import plot
np.set_printoptions(threshold=sys.maxsize)

#1. avergage kernel --> kernel matrix 
#2. svm using the kernel matrix 
#[n_samples, n_features]

####
# use pre-computed kernel  (train, train)
# (test, train)
###

bn_path = "./panel1_Data23_bf/"
file_names = []
group1 = ['base', 'tx'] 
group2 = ['HD', 'NR', 'R']



def my_kernel(X1, X2, kernel, gamma): # X1, X2 are the lists of file names  
	n = len(X1)
	m = len(X2)
	K = np.zeros([n, m])

	for i in range(n):
		fn_x1 = X1[i] 
		x1 = load_from_bn(fn_x1) #load matrix of this sample 
		for j in range(i,m): # only do triangle matrix
			fn_x2 = X2[j]

			x2 = load_from_bn(fn_x2)
			#pairwise_kernels of two individuals 
			temp = pairwise.pairwise_kernels(x1, x2, kernel, gamma=gamma)
			K[i, j] = np.mean(temp)

			if i != j:
				K[j, i] = K[i, j]
	return K


def svm_classifier(X, y, kernel, pos_label, classes, gamma): 

	y_preds = [] 
	y_scores = [] 

	#1a. Calculate kernel 
	#K = my_kernel(X, X, kernel=kernel, gamma=gamma)
	
	#backup the matrix 
	#np.save("k_matrix_" + kernel + "_" + str(gamma) + "_" + pos_label \
	#		 + "_backup", np.vstack(K))
	
	#1b. load K matrix from backup file. 
	K = np.load("k_matrix_" + kernel + "_" + str(gamma) + "_" + classes \
			 + "_backup.npy")
	
	#2. Fit whole dataset into svm to get alpha and vectors. (Don't do step 3.)
	clf = svm.SVC(kernel = "precomputed")
	clf.fit(K, y)
	print( "\n parameters of clf: " , flush=True)
	print(clf.get_params(deep=True))
	print("\n dual coef: " + str(clf.dual_coef_))
	print("\n sup indices: " + str(clf.support_))
	print("\n sup vectors: " + str(clf.support_vectors_))
	print(clf.support_vectors_)
	print("\n intercept: " + str(clf.intercept_) )


	'''	
	3. Slice matrix to do prediction (Do not do step 2.)

	train_indices = range(len(X))

	for test_index in range(len(X)):
		#K_test: is the row by test_index in the whole matrix 
		K_test = K[test_index]
		K_test = np.delete(K_test, test_index, 0)

		#K_train: Delete row and column by test_index 
		K_train = K 
		K_train = np.delete(K_train, test_index, 0)
		K_train = np.delete(K_train, test_index, 1)

		y_test = y[test_index]
		y_train = copy.copy(y) #Need use cp.copy() here. 
		y_train.pop(test_index)

		#Training...
		clf = svm.SVC(kernel = "precomputed")
		clf.fit(K_train, y_train)

		print( "\n Test index " + str(test_index) + " parameters of clf: " , flush=True)
		print(clf.get_params(deep=True))
		print("\n dual coef: " + str(clf.dual_coef_))

		#Testing... 
		y_pred = clf.predict(K_test.reshape(1, -1))
		y_preds.append(y_pred[0]) #Only one sample 

		#Score of decision_fucntion to do roc_curve 
		y_s = clf.decision_function(K_test.reshape(1, -1))
		y_scores.append(y_s[0]) #Only one sample 

	
	print("\nTrue y: " + str(y), flush=True)
	print("Predicted y: " + str(y_preds), flush=True)
	print("y_scores : " + str(y_scores), flush=True)

	evaluation(y, y_preds, y_scores, pos_label)
	'''


def evaluation(true_y, pred_y, y_scores, pos_label):
	t_y = np.array(true_y)
	p_y = np.array(y_scores)
	fpr, tpr, thresholds = metrics.roc_curve(true_y, y_scores, pos_label=pos_label)
	auc = metrics.auc(fpr, tpr)
	print("AUC: ", auc)

	#C[i, j] is equal to the number of observations known to be in group i but predicted to be in group j.
	CM = metrics.confusion_matrix(true_y, pred_y)
	#May not need, but I write it in advance. 
	TN = CM[0][0]
	FN = CM[1][0]
	TP = CM[1][1]
	FP = CM[0][1]

	accu = metrics.accuracy_score(true_y, pred_y)
	print("Accuracy: ", accu)
	precision = metrics.precision_score(true_y, pred_y, pos_label=pos_label)
	print("Precision : ", precision)


def load_from_bn(file_name): 
	data = np.load(file_name + '.npy')
	return data 


def under_sample(big_fn, size): 
	#undersample the biggest file 
	data = load_from_bn(big_fn)
	under_idx = np.random.choice(len(data), size=size)
	under_data = [] 
	for idx in under_idx: 
		under_data.append(data[idx])
	np.save(big_fn+"_under", np.vstack(under_data))


def main(): 

	# a list of path+filenames to load data 
	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = bn_path + 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i) + ".csv"
				file_names.append(fn)

	big_fn = "./panel1_Data23_bf/Data23_Panel1_tx_HD3_Patient3.csv"	
	#under_sample(big_fn, 10000)
	file_names.remove(big_fn)
	file_names.append(big_fn+"_under")


	y_1 = [] 

	for fn in file_names:
		if 'base' in fn: 
			y_1.append('base') # baseline
		else:
			y_1.append('tx') # post treatment

	'''
	###Binary classes prediction on base, tx

	gamma = [1.0/(2000^2), 1.0/(1500^2)]# try two 
	# use filenames as train list, load each file each time when doing pairwise kernel 
	kerne_list = ['rbf', 'laplacian' , 'poly', 'cosine', 'linear']

	for k in kerne_list[:3]:
		for g in gamma: 
			print("kernel: " + k + " gamma: " + str(g), flush=True)
			svm_classifier(file_names, y_1, kernel=k, pos_label= 'base', gamma=g)
	'''

	### Multi classes prediction on NR, HD, R 
	### Only for tx samples 

	fn_nr_r = []
	fn_nr_hd = []
	fn_hd_r = []
	
	y_nr_r = []
	y_nr_hd = []
	y_hd_r = []
	
	for fn in file_names:
		if 'tx' in fn: 
			if 'NR' in fn: 
				y_nr_r.append('NR') # non-responder
				fn_nr_r.append(fn)
				y_nr_hd.append('NR')
				fn_nr_hd.append(fn)

			elif 'HD' in fn: 
				y_nr_hd.append('HD') # healthy donor
				fn_nr_hd.append(fn)
				y_hd_r.append('HD') 
				fn_hd_r.append(fn)

			else:
				y_nr_r.append('R') # responder
				fn_nr_r.append(fn)
				y_hd_r.append('R')
				fn_hd_r.append(fn)

	k_list  = ['poly', 'rbf', 'laplacian'] #,'cosine'] 
	g_list = [1, 1.0/(2000^2)] 


	# pos_label set --> R for R vs. NR   (was NR)
	# 					NR for HD vs. NR  (was HD)
	# 					R for HD vs. R 

	for k in k_list[:1]: 
		for g in g_list: 
	
			print("\nKernel: " + k + "  Gamma: " + str(g),  flush=True)
			print("NR vs R: ", flush=True)
			svm_classifier(fn_nr_r, y_nr_r, kernel=k, pos_label= 'R', classes= 'RvNR',  gamma=g)
			print("\nNR vs HD: ", flush=True)
			svm_classifier(fn_nr_hd, y_nr_hd, kernel=k, pos_label= 'NR', classes ='NRvHD', gamma=g)
			print("\nHD vs R: ", flush=True)
			svm_classifier(fn_hd_r, y_hd_r, kernel=k, pos_label= 'R', classes ='RvHD',  gamma=g)

			##base vs. tx 
			print("\nbase vs tx: ", flush=True)
			svm_classifier(file_names, y_1, kernel=k, pos_label= 'tx', classes ='TXvBASE', gamma=g)

#main()


def hist_cells():
	"""
	This function plots a histogram of pairwise_kernel dists of cells in one individual
	"""
	#Take a sample with samll cellls. 
	small_fn = "./panel1_Data23_bf/Data23_Panel1_base_NR2_Patient7.csv" 
	#small_fn2 = "./panel1_Data23_bf/Data23_Panel1_tx_R4_Patient14.csv" 

	metrics = ['euclidean', 'manhattan' ,'cosine']

	for k in metrics[1:]: 

		cells = load_from_bn(small_fn)
		#cells = ss.zscore(cells) #transform data to z-score 
		print(cells.shape)
		print(k)

		num = cells.shape[0]
		dist_list = [] 

		for i in range(num):
			x1 = cells[i]
		
			for j in range(i,num):
				x2 = cells[j]
				temp = pairwise.pairwise_distances(np.array([x1, x2]), metric=k)
				dist_list.append(np.mean(temp))

		np.save("dist_list_" + k, dist_list)

		plt.figure()
		plt.hist(dist_list, bins='auto')
		plt.title("Histogram of distance by " + k + " kernel.")
		plt.show()
		plt.savefig('hist_' + k + '.png')
		plt.clf() 
		print('plot finished. ')

#hist_cells()


def poly_hist(): 
	k = 'poly'
	dist_list = np.load("dist_list_poly_z.npy")

	plt.hist(dist_list, bins='auto')
	plt.title("Histogram of distance by " + k + " kernel.")
	plt.show()
	plt.savefig('hist_' + k + '_z_transformed.png')

#poly_hist()



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

    ###
    # Plot heatmaps of K matrix 
    ###
def t(): 

	# a list of path+filenames to load data 
	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = bn_path + 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i) + ".csv"
				file_names.append(fn)

	big_fn = "./panel1_Data23_bf/Data23_Panel1_tx_HD3_Patient3.csv"	
	#under_sample(big_fn, 10000)
	file_names.remove(big_fn)
	file_names.append(big_fn+"_under")


	y_1 = [] 
	y_2 = [] 

	for fn in file_names:
		if 'base' in fn: 
			y_1.append('base') # baseline
		else:
			y_1.append('tx') # post treatment


	y_nr_r = []
	y_nr_hd = []
	y_hd_r = []

	for fn in file_names:
		if 'tx' in fn: 
			if 'NR' in fn: 
				y_nr_r.append('NR') # non-responder
				y_nr_hd.append('NR')

			elif 'HD' in fn: 
				y_nr_hd.append('HD') # healthy donor
				y_hd_r.append('HD') 

			else:
				y_nr_r.append('R') # responder
				y_hd_r.append('R')
	
	NR = y_nr_r
	HD = y_nr_hd
	R = y_hd_r

	matrix_list = np.load('k_matrix_poly_0.0004995004995004995_backup.npy')
	np.fill_diagonal(matrix_list, 0)
	fn = open("k_matrix_poly_0.0004995004995004995_backup.txt", "w")
	fn.write(str(matrix_list))
	#fig, ax = plt.subplots()
	#im = ax.imshow(matrix_list)
	fig, ax = plt.subplots()
	im, cbar = heatmap(matrix_list, y_1, y_1, ax=ax,
	                   cmap="YlGn", cbarlabel="Samples")
	#texts = annotate_heatmap(im, valfmt="{x:.1f} t")

	fig.tight_layout()
	#plt.show()
	mp.savefig('poly_0.0005_base_tx.png')

#t()


def plot_k_matrix():
	###
	#Plot by PCA
	###
	
	data = np.load('k_matrix_poly_1_NR_backup.npy')
	np.fill_diagonal(data, 0)
	print("No. of data: ", len(data))
	#trans_data = list(ss.zscore(data)) #transform data to z-score 
	
	# a list of path+filenames to load data 
	y_1 = ['base'] * 15
	y_1 += ['tx'] * 15
	# split into 2 groups 
	cate1 = group1 
	colors = ['r', 'b']

	plot.plot_PCA(data, y_1, cate1, colors, "base_tx_k_poly_1.png")
	print("Plot PCA Done ")
	plot.kernal_PCA(data, y_1, cate1, colors, "base_tx_k_poly_1_kernel_rbd_10.png")
	print("Plot PCA kernal Done ")


	''' *** No data showing ***  
	y_nr_r = ['NR'] * 5 
	y_nr_r += ['R'] * 5 
	y_nr_hd = ['HD', 'HD', 'HD', 'HD', 'NR', 'NR', 'NR', 'NR', 'NR', 'HD']
	y_hd_r = ['HD', 'HD', 'HD', 'HD', 'R', 'R', 'R', 'R', 'R', 'HD']
	NR = y_nr_r
	HD = y_nr_hd
	R = y_hd_r
	plot.plot_PCA(data, y_nr_r, cate1, colors, "nr_r_k_poly_1.png")
	#print("Plot 1 Done ")
	plot.kernal_PCA(data, y_nr_r, cate1, colors, "nr_r_k_poly_1_kernel_rbd_10.png")
	#print("Plot 1 kernal Done ")
	'''	

#plot_k_matrix()


def gradient(): 

	"""
	#Calculate the gradient of our prediction 

	index of support vectors: 4, 26
	corresponding alpha values: lapha4 = −2.1420987e−30, lapha26 = 2.1420987e−30.

	""" 
	# a list of path+filenames to load data 
	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = bn_path + 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i) + ".csv"
				file_names.append(fn)

	big_fn = "./panel1_Data23_bf/Data23_Panel1_tx_HD3_Patient3.csv"	
	#under_sample(big_fn, 10000)
	file_names.remove(big_fn)
	file_names.append(big_fn+"_under")


	#1. Load two samples we need 

	#X_4 (include 0 index)
	x_4 = load_from_bn(file_names[4])
	s_x4 = np.sum(x_4, axis=0)  #sum rows of X_4.
	m_4 = len(x_4)

	#X_26  
	x_26 = load_from_bn(file_names[26])
	s_x26 = np.sum(x_4, axis=0)  #sum rows of X_26.
	m_26 = len(x_26)

	a_4 = -2.1420987e-30
	a_26 = 2.1420987e-30
	
	
	for fn in file_names[29:]: 
		x_i = load_from_bn(fn)
		n = len(x_i)
		grad = 0.0

		s_4 = pairwise.pairwise_kernels(x_i, x_4, 'poly', gamma=1, degree=2) #coef default = 1 
		s_s4 = np.sum(s_4, axis=1)  #sum columns of S_4
		naplaK_4 = (1/(m_4*n)) * 3 * np.outer(s_s4, s_x4) # two arrays multiple to a matrix

		grad = a_4 * naplaK_4 
		
		s_26 = pairwise.pairwise_kernels(x_i, x_26, 'poly', gamma=1, degree=2) #coef default = 1 
		s_s26 = np.sum(s_26, axis=1)  #sum columns of S_26
		naplaK_26 = (1/(m_26*n)) * 3 * np.outer(s_s26, s_x26) # two arrays multiple to a matrix

		grad += a_26 * naplaK_26

		#write gradient of this X_i to nb file 
		np.save(fn+"_gradient", np.vstack(grad))
		print(".")
		
	
#gradient()











