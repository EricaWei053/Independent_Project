#from bokeh.plotting import show
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import numpy as np 
import sys 
import os
#import flowkit as fk
import csv
#from sklearn.manifold import TSNE
import scipy.stats as ss
import random
import plot
#import svm
import keras_nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
#from keras.layers import Input, Dense
#from keras.models import Model
import openTSNE
import pickle
import pandas as pd
from pandas.plotting import scatter_matrix
#import seaborn as sns
from scipy.stats import norm
from sklearn import preprocessing


fcs_path = './cytof_data/' 
path = './panel1_Data23/'
file_names = []
group1 = ['base', 'tx'] 
group2 = ['HD', 'NR', 'R']
bn_path = "./panel1_Data23_bf/"


def convert_excel(): 
	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i)

				sample = fk.Sample(fcs_path + fn + '.fcs', subsample_count=None)
				#data_dict = sample.get_metadata()
				sample.export_csv(source='raw', subsample=False, filename=fn +'.csv', directory=path_into)
				print("write file done: " + fn )
				

def hist_plot(data, categories): 

	# Use sns. 
	data = pd.DataFrame(data, columns=categories)
	for col in categories:
		sns.kdeplot(data[col], shade=True)
		#sns.distplot(data[col])
	plt.xlim(-10,10)
	plt.show()
	

def load(num_cell, file_names):  # num_cell is the number of how many cells per individual to load

	data = [] 
	counts = [] # to record # of cells in each indvidual

	for fn in file_names:
		with open(path+fn, mode='r') as csv_file:
			csv_reader = csv.DictReader(csv_file)
			count = 0 
			for row in csv_reader:
				x = np.array(list(row.values()))
				y = x.astype(np.float)
				data.append(y)
				count += 1 
				if num_cell != 0: # has the restriction about num of lines to read 
					if count >= num_cell:
						break

			counts.append(count)
		#all_data[fn] = np.vstack(data) # maybe hstack? 
	return data, counts


def write_to_binary(data, fn): #write a list of data to one npy file
	np.save(fn, np.vstack(data))


def main(): 
	
	#convert_excel()
	#hist_plot(data, categories)

	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i) + ".csv"
				file_names.append(fn)


	'''
	###
	#Plot by PCA, kernel PCA, tSNE
	###
	data, counts = load(100, file_names)

	print("No. of data: ", len(data))
	trans_data = list(ss.zscore(data)) #transform data to z-score 
	#write_file(trans_data, "z_transformed.txt") #backup 

	targets_1 = ['base'] * (sum(counts[:15]))
	targets_1 += ['tx'] * (sum(counts[15:]))

	targets_2 = [] 
	cate2 = [] 
	idx = 0
	for gp in group1:
		for nm in group2: 
			cate2.append(gp + "_" + nm)
			for i in range(5): 
				targets_2 += [gp + "_" + nm] * counts[idx+i]
			idx += 5


	targets_3 = [] 
	for i in range(2):
		for idx in range(3):
			nm = group2[idx] 
			targets_3 += [nm] * sum(counts[(idx+i)*5:(idx+i+1)*5])
	
	
	# split into 2 groups 
	cate1 = group1 
	colors = ['r', 'b']

	#plot.plot_PCA(trans_data, targets_1, cate1, colors, "base_tx_test2.png")
	#print("Plot 1 Done ")
	plot.kernal_PCA(trans_data, targets_1, cate1, colors, "base_tx_k_cosine_15.png")
	print("Plot 1 kernal Done ")
	#plot.tSNE(trans_data, targets_1, cate1, colors, "base_tx_tSne_test.png")
	#print("Plot TSNE Done ")

	
	colors = ['b', 'g', 'r', 'c', 'm', 'y']

	#plot.plot_PCA(trans_data, targets_2, cate2, colors, "6_groups.png")
	#print("Plot 2 Done ")
	plot.kernal_PCA(trans_data, targets_2, cate2, colors, "6_groups_k_cosine_15.png")
	print("Plot 2 kernal Done ")
	#plot.tSNE(trans_data, targets_2, cate2, colors, "6_groups_tSne.png")
	#print("Plot 2 tSNE Done ")


	cate3 = group2
	colors = ['b', 'g', 'r' ] 
	
	#plot.plot_PCA(trans_data, targets_3, cate3, colors, "3_groups.png")
	#print("Plot 3 Done ")
	plot.kernal_PCA(trans_data, targets_3, cate3, colors, "3_groups_k_cosine_15.png")
	print("Plot 3 kernal Done ")
	#plot.tSNE(trans_data, targets_3, cate3, colors, "3_groups_tSne.png")
	#print("Plot 3 tSNE Done ")
	'''

	###
	# Write file into npy format. 
	###
	kernel_list = ["linear", "rbf", "sigmoid", "cosine", "poly"] 

	bn_path = "./panel1_Data23_bf/"
	sample_l = [] 
	y_1 = []
	y_2 = [] 
	
	for fn in file_names:
		data, counts = load(0, [fn])
		#sample_l.append(data)
		#Write data into binary file, each individual for each file 
		write_to_binary(data, bn_path+fn)

	print("write to binary files Done. ")


#main()

def load_from_bn(file_names):
	data = [] 
	counts = [] # to record # of rows in each file

	for fn in file_names:
		data_1 = np.load(fn + '.npy')
		data.extend(data_1)
		counts.append(len(data_1))
	return data, counts


def get_file_names(sub):
	bn_file_names = []
	# a list of path+filenames to load data 
	for j in range(2): 
		gp = group1[j]
		for k in range(3): 
			name = group2[k]
			for num in range(1, 6): 
				i = (k * 5)+ num
				fn = bn_path + 'Data23_Panel1_' + gp +'_' + name + str(num) + '_Patient' + str(i) + ".csv"+sub
				bn_file_names.append(fn)

	big_fn = "./panel1_Data23_bf/Data23_Panel1_tx_HD3_Patient3.csv"
	idx_rp = bn_file_names.index(big_fn)
	#under_sample(big_fn, 10000)
	#bn_file_names.remove(big_fn+sub)
	bn_file_names[idx_rp] = big_fn+"_under" + sub

	return bn_file_names

def get_labels():
	b_t = ['base'] * 15 + ['tx'] * 15
	r_nr = ['HD'] * 5 + ['NR'] * 5 + ['R'] * 5
	r_nr = r_nr * 2 

	return b_t, r_nr


def plot_gradient(): 

	orig_fn = get_file_names("")
	grad_fn = get_file_names("_gradient")

	###
	#Plot first sample on TSNE
	###

	total_X, X_counts = load_from_bn(orig_fn)
	
	'''
	total_X = pd.DataFrame(total_X) 
	#total_grad, grad_num = load_from_bn(grad_fn)

	#Run tSNE on whole samples 
	embedding = openTSNE.TSNE().fit(total_X) 
	

	#Back-up into file 
	pickle_out = open("tsne_all_data.pickle","wb")
	pickle.dump(embedding, pickle_out)
	pickle_out.close()
	'''

	#Load from pickle file 
	pickle_in = open("tsne_all_data.pickle","rb")
	embedding = pickle.load(pickle_in)


	vis_x = embedding[:, 0]
	vis_y = embedding[:, 1]
	marker = 0 #Mark the start point to split data. 
	for i in range(len(orig_fn)):  

		print("Loop..." + str(i))
		#Get x, y for each sample. 
		x = vis_x[marker: marker+X_counts[i]]
		y = vis_y[marker: marker+X_counts[i]]
		marker += X_counts[i]

		#Gradient for this sample  X' = X + eplsion*(df)
		data_0 = np.load(orig_fn[i] + '.npy')
		grad_0 = np.load(grad_fn[i] + '.npy')
		#grad_x = data_0 + grad_0*sys.float_info.epsilon
		grad_x = grad_0

		
		'''
		#1. Using sum columns of grad_f for coloring cells 
		targets_0 = grad_0.sum(axis=1)
		scaler = MinMaxScaler()
		scaler.fit(targets_0.reshape(-1, 1))
		norm_targ = scaler.transform(targets_0.reshape(-1, 1))
		'''
		print("Getting data.. ")
		#2. Using sum columns of vectors for coloring cells
		targets_0 = grad_x.sum(axis=1)
		scaler = MinMaxScaler()
		scaler.fit(targets_0.reshape(-1, 1))
		norm_targ = scaler.transform(targets_0.reshape(-1, 1))

		norm_targ = np.around(norm_targ, decimals=2)
		#Category need to fit in tsne. 
		cate_0 = np.unique(norm_targ)

		print(targets_0.shape)
		print(len(x), len(y))

		print("Plotting.. ")
		"""
		a. Plot each sample with same embedding.  
		"""
		fn = orig_fn[i]+"_grad_tSne_emb.png"
		#Plot tsne embedding. 
		n = len(cate_0)
		#Scale data to 0-1 range on both axis 
		norm_x, norm_y = plot.scale_data(x, y)
		plt.scatter(norm_x, norm_y, c=targets_0 ,cmap=plt.cm.get_cmap("hot"), s = 3)
		plt.colorbar()
		mp.savefig(fn)
		plt.close()


		"""
		b. Plot each sample with their own embedding.  
		"""
		#Polt single sample by openTSNE 
		#plot.open_tSNE(data_0, grad_x, targets_0, cate_0, orig_fn[i]+"_grad_tSne_vec.png")
		#print("Plot gradient TSNE Done ")


#plot_gradient()


def simple_autoencoder():

	#matrix to 2-dimension vector 

	import tensorflow as tf
	import math

	orig_fn = get_file_names("")
	input = np.load(orig_fn[0] + '.npy').transpose()

	noisy_input = input + .2 * np.random.random_sample((input.shape)) - .1
	output = input

	# Scale to [0,1]
	scaled_input_1 = np.divide((noisy_input-noisy_input.min()), (noisy_input.max()-noisy_input.min()))
	scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
	# Scale to [-1,1]
	scaled_input_2 = (scaled_input_1*2)-1
	scaled_output_2 = (scaled_output_1*2)-1
	input_data = scaled_input_2
	output_data = scaled_output_2

	# Autoencoder with 1 hidden layer
	n_samp, n_input = input_data.shape 
	#print("check n_sample")
	#print(n_samp)
	#print(n_input)
	n_hidden = 2

	x = tf.placeholder("float", [None, n_input])
	# Weights and biases to hidden layer
	Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
	bh = tf.Variable(tf.zeros([n_hidden]))
	h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
	# Weights and biases to hidden layer
	Wo = tf.transpose(Wh) # tied weights
	bo = tf.Variable(tf.zeros([n_input]))
	y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
	# Objective functions
	y_ = tf.placeholder("float", [None,n_input])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	meansq = tf.reduce_mean(tf.square(y_-y))
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	n_rounds = 5000
	batch_size = min(50, n_samp)

	for i in range(n_rounds):
	    sample = np.random.randint(n_samp, size=batch_size)
	    batch_xs = input_data[sample][:]
	    batch_ys = output_data[sample][:]
	    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
	    #if i % 100 == 0:
	    #    print(i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}), sess.run(meansq, feed_dict= {x: batch_xs, y_:batch_ys}))

	
	print("Target:")
	print(output_data)
	print("Final activations:")
	print(sess.run(y, feed_dict={x: input_data}))
	print("Final weights (input => hidden layer)")
	print(sess.run(Wh))
	print(sess.run(Wh).shape)
	print("Final biases (input => hidden layer)")
	print(sess.run(bh))
	print("Final biases (hidden layer => output)")
	print(sess.run(h, feed_dict={x: input_data}))
	
	#print(sess.run(Wh) + sess.run(bh))
	
#simple_autoencoder()


def one_hot(Y, num_class):

	targets = np.array([Y]).reshape(-1)
	one_hot_targets = np.eye(num_class)[targets]
	return one_hot_targets


def plot_hist(data):

	metrics = ['euclidean', 'manhattan' ,'cosine']
	for k in metrics: 

		cells = data
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

		np.save("dist_nn_list_" + k, dist_list)

		plt.figure()
		plt.hist(dist_list, bins='auto')
		plt.title("Histogram of distance by " + k + " kernel.")
		plt.show()
		plt.savefig('dist_nn_' + k + '.png')
		plt.clf() 
		print('plot finished. ')

test_idx = 0 
X_test = None

def keras_auto(): 

	encoding_dim = 9
	label_dim = 0 
	all_Y = None 
	
	orig_fn = get_file_names("")
	total_X, X_counts = load_from_bn(orig_fn)
	Y_bt, Y_nrr = get_labels()
	label = "bt"

	if label== "bt": 
		all_Y = Y_bt
		label_dim = 2 

	elif label== "nrr": 
		all_Y = Y_nrr
		label_dim = 3 
	else: 
		print("Lable setting wrong.")
		exit()

	

	predicts = []
	count = 0
	marker = 0 
	kernel = "poly"
	f = open("auto_out_11_29_2.txt",'w')
	
	for i in range(len(orig_fn)): 

		test_idx = i 

		#Testing data 
		X_test = total_X[marker: marker+X_counts[i]]
		marker += X_counts[i]
	
		#Testing label 
		y_test = all_Y[i]

		#Training data 
		X = total_X[:marker] + total_X[marker+X_counts[i]:]
		
		fn_train = "l_" + str(encoding_dim) + "_train_"+str(i)+".pickle"
		fn_test = "l_" + str(encoding_dim) + "_test_"+str(i)+".pickle"

		hidden_train = None 
		hidden_test = None
		if not os.path.exists(fn_train):
			print("Doing autoencoder.")
			#Autoencoder function 
			hidden_train, hidden_test = keras_nn.autoencoder(X, X_test, encoding_dim)

			#Back-up into file 
			pickle_train = open("l_" + str(encoding_dim) + "_train_"+str(i)+".pickle","wb")
			pickle.dump(hidden_train, pickle_train)
			pickle_train.close()

			pickle_test = open("l_" + str(encoding_dim) + "_test_"+str(i)+".pickle","wb")
			pickle.dump(hidden_test, pickle_test)
			pickle_test.close()
			
		else: 
			#Load from file 
			print("Loading from file ")
			train_in = open(fn_train,"rb")
			hidden_train = pickle.load(train_in)
			test_in = open(fn_test,"rb")
			hidden_test = pickle.load(test_in)
		

		'''
		#Ploting 

		#trans_data = list(ss.zscore(hidden_all)) #transform data to z-score 
		hidden_all = np.concatenate((hidden_test, hidden_test))
		plot.plot_PCA(hidden_all, Y_bt, categories= ['base', 'tx'], colors = ['r', 'b'], fn= "PCA_nn_bt.png")
		plot.plot_PCA(hidden_all, Y_nrr, categories= ['NR', 'R', 'HD'], colors = ['r', 'b', 'y'], fn= "PCA_nn_rnh.png")

		plot.tSNE(hidden_all, Y_bt, categories= ['base', 'tx'], colors = ['r', 'b'], fn= "tSNE_nn_bt.png")		
		plot.tSNE(hidden_all, Y_nrr, categories= ['NR', 'R', 'HD'], colors = ['r', 'b', 'y'], fn= "tSNE_nn_rnh.png")
		
		print("Plots done. ")
		exit()

		'''

		#get max, mean of 90th percentile of encoded training and testing 
		print("get max, mean, 90th")

		#testing max, mean, 90th 
		print(hidden_test.shape)
		print(hidden_train.shape)

		test_max = np.amax(hidden_test, axis=0)
		test_mean = np.mean(hidden_test, axis=0)
		test_90 = np.percentile(hidden_test, 90, axis=0)
		test_all = [test_max, test_mean, test_90]
		
		#Get list of training max, mean, 90th 
		walker = 0 
		X_max = [] 
		X_mean = [] 
		X_90 = []
	
		for j in range(len(orig_fn)):
			if i == j:
				continue 
			X_j = hidden_train[walker: walker+X_counts[j]]
			X_max.append(np.amax(X_j, axis=0))
			X_mean.append(np.mean(X_j, axis=0))
			X_90.append(np.percentile(X_j, 90, axis=0))
			walker += X_counts[j]

		train_all = [X_max, X_mean, X_90]

		"""
		classification for test data by using hidden layer 
		classification: svm, linear regression 
		to decide pred: max, 90th, average
		"""

		preds_svm = []
		preds_nn = [] 
		print("test index : " + str(i))
		print("true_y:" + str(y_test))

		Y = all_Y.copy()
		Y.pop(i)
		for k in range(3):
			X = np.asarray(train_all[k]) #train_max, train_mean, train_90
			X_t = np.asarray(test_all[k])

			#SVM 
			print("SVM training... ")
			g = 1.0/(3^2) 
			clf = SVC(kernel = kernel, gamma = g)  #Use setting from pair-wise kernel 
			clf.fit(X, Y)
			print("SVM predicting... ")
			pred = clf.predict(X_t.reshape(1, -1))
			preds_svm.extend(pred)
			print("SVM prediction done. ")
			print("svm prediction: ", pred)


			#Nerual Network 
			#one hot label
			s = pd.Series(Y)
			one_hot_Y = pd.get_dummies(s).values
			X = preprocessing.normalize(X)
			X_t = preprocessing.normalize(X_t.reshape(1, -1)) #reshape, single saample 
			
			predictions = keras_nn.classify(X, one_hot_Y, X_t, encoding_dim, label_dim)
			print("NN predictions: ", predictions)
			preds_nn.extend(predictions.tolist())

		predicts.append(preds_svm + preds_nn)


	result = pd.DataFrame(predicts, columns=['svm_max', 'svm_mean', 'svm_90th', 'nn_max', 'nn_mean', 'nn_90th'])
	result.insert(loc=0, column='True label', value=all_Y)
	
	f.write(" hidden dim: " + str(encoding_dim))
	f.write(" svm kernel: " + str(kernel) + '\n')
	f.write(result.to_string())
	f.close()



keras_auto()



