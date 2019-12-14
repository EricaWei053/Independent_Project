import sys 
import random
import numpy as np 
import pandas as pd
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, TensorDataset, DataLoader
from main import get_file_names, load_from_bn, get_labels
lembda=0.3

path = './panel1_Data23/'
file_names = []
group1 = ['base', 'tx'] 
group2 = ['HD', 'NR', 'R']
bn_path = "./panel1_Data23_bf/"



def choose_dev(tes_idx):
	idx_list = [k for k in range(30)]
	dev_idx = random.choice(idx_list)
	while dev_idx == tes_idx:
		dev_idx = random.choice(idx_list)

	return dev_idx


def my_loss(x_original, x_decoded, y_pred, y, lembda ):

	return ((x_original - x_decoded) ** 2).sum() \
					+ lembda * nn.CrossEntropyLoss(y_pred, y)





class ClusterDataset(Dataset):
    def __init__(self, data, slice_list):
       
        self.samples = data  # [_ * 36 columns] 

        self.cluster_indices = [] # a list of tuple (start_index, batch_size)

        start_idx = 0 
        for i in range(len(slice_list)): # should be 28 tuples for training, 1 for testing, 1 for dev. 
        	self.cluster_indices.append((start_idx, slice_list[i]))
        	start_idx += slice_list[i]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



#Reference: https://discuss.pytorch.org/t/dataloader-for-variable-batch-size/13840/5

class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=False):
        self.data_source = data_source
        
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, #"do not declare batch size in sampler " \
                                                         #"our data source already got one"
     
            self.batch_sizes = [batch_size for (start_idx, batch_size) in self.data_source.cluster_indices] # a list of batch size 
        else:
            self.batch_sizes = self.data_source.batch_sizes
        
        self.shuffle = shuffle
      

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for start_idx, batch_size in self.data_source.cluster_indices:
            
            #Chunkify data manually
            ## questioning...  
            batches = [
                self.data_source.samples[i: i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), batch_size)
            ]

            batches = []
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class my_autoencoder(nn.Module):
	def __init__(self, encoding_dim, label_dim):
		super(my_autoencoder,self).__init__()
		
		self.encoder = nn.Sequential(
			nn.Linear(1*36, 24) ,
			nn.ReLU(True),
			nn.Linear(24 , encoding_dim),
			nn.ReLU(True))

		self.decoder = nn.Sequential(             
			nn.Linear(encoding_dim, 24),
			nn.ReLU(True),
			nn.Linear(24, 1*36),
			nn.ReLU(True))

		self.classifier = nn.Sequential(
			nn.Linear(encoding_dim, encoding_dim*2),
			nn.ReLU(True),
			nn.Linear(encoding_dim*2, encoding_dim*4),
			nn.ReLU(True),
			nn.Linear(encoding_dim*4, label_dim),
			nn.Softmax()
		)


	def forward(self, x):
		#autoencoder 
		x_encoded = self.encoder(x)
		x_decoded = self.decoder(x_encoded)

		#classficiation (use max one)
	
		print("individual")
		print(x_individual.shape)
		max_tensor = torch.max(x_encoded, axis=0)[0]
		print(max_tensor.shape)
		x_out = self.classifier(max_tensor)

		return x_decoded, x_out


def train_model(model, optimizer, train_generators, dev_generator):
	"""
	Perform the actual training of the model based on the train and dev sets.
	:param model: autoencoder and classification
	:param loss_fn: a custom function that can calculate loss 
	:param optimizer: a created optimizer you will use to update your model weights
	:param train_generator: a DataLoader that provides batches of the training set
	:param dev_generator: a DataLoader that provides batches of the development set
	:return model, the trained model (autoencoder)
	"""
	last_dev_loss = np.inf
	# loop until the dev loss stops improving
	patience = 10 
	num_epochs = 10
	for epoch in range(num_epochs):
		while True:
			model.train()

			for train_g in train_generators:
				for tx, ty in train_g: 
					# zero the gradients each batch
					model.zero_grad()
					print(tx)
					print(ty)

					b_x = x.view(-1, 1*36)   # batch x, shape (batch, 28*28)
					print(b_x)

					exit()
					# calculate predictions and loss
					tx_decoded, tx_out = model(tx)
					loss = my_loss(x_original=tx, x_decoded=tx_decoded, y_pred=tx_out, y=ty, lembda=lembda)

					# calculate all the gradients
					loss.backward()
					
					# apply all the parameter updates
					optimizer.step()

				# test mode
				model.eval()
				dev_loss = 0
				with torch.no_grad():
					for dx, dy in dev_generator:
						dx_decoded, dx_out = model(dx)
						# sum up all the batch losses 
						dev_loss += my_loss(x_original=dx, x_decoded=dx_decoded, y_pred=dx_out, y=dy, lembda=lembda).item()
				
				print("Dev loss:", dev_loss)
				# check if the loss has gotten worse since last epoch; stop if so
				if dev_loss > last_dev_loss:
					patience -= 1 
					if patience <= 0: 
						break
				else:
					patience = 5 

				print("dev_loss: ", dev_loss)
				last_dev_loss = dev_loss
		
		return model


def test_model(model, test_generator):

	true_y = []
	pred_y = []

	# Keep track of the loss
	loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
	if USE_CUDA:
		loss = loss.cuda()

	model.eval()

	# Iterate over batches in the test dataset
	with torch.no_grad():
		for X_b, y_b in test_generator:
			# Predict
			y_pred = model(X_b)

			true_y.extend(y_b.cpu().detach().numpy().astype(int).tolist())
			pred_y.extend(y_pred.argmax(1).cpu().detach().numpy().astype(int).tolist())

	# Print total loss and macro F1 score
	print("Predicted label: ")
	print(pred_y)
	print("=" * 20)
	print("True label")
	print(true_y)
	print("=" * 20)


def main():


	#put data in dataloader 
	orig_fn = get_file_names("")
	total_X, X_counts = load_from_bn(orig_fn)
	Y_bt, Y_nrr = get_labels()

	#one hot label
	Y_bt = pd.get_dummies(pd.Series(Y_bt)).values
	Y_nrr = pd.get_dummies(pd.Series(Y_nrr)).values

	X =  [] 
	Y = Y_bt
	walker = 0 

	for i in range(len(X_counts)):
		X.append(total_X[walker : walker+X_counts[i]])
		walker += X_counts[i]


	#convert to tensors
	#tensor_X = torch.LongTensor(X)
	#tensor_Y_2 = torch.from_numpy(Y_bt)

	for tes_idx in range(len(X)): 

		#test data 
		#convert to pytorch 
		test_X = torch.FloatTensor(X[tes_idx])
		test_Y = torch.FloatTensor([Y[tes_idx]]*len(X[tes_idx]))
		test_slice = [len(X[tes_idx])] 
		print(test_X.shape)
		print(test_slice)
		test_dataset = TensorDataset(test_X, test_Y)
		test_dataloader = DataLoader(test_dataset)

		#train_X = torch.cat([tensor_X[0:i], tensor_X[i+1:]])
		#train_Y = torch.cat([tensor_Y[0:i], tensor_Y[i+1:]])

		#randomly choose one individual as dev dataset 
		dev_idx = choose_dev(tes_idx)
		dev_slice = [len(X[dev_idx])] 
		dev_X = torch.FloatTensor(X[dev_idx])
		dev_Y = torch.FloatTensor([Y[dev_idx]]*len(X[dev_idx]))

		print(dev_X.shape)
		print(dev_Y)

		dev_dataset = TensorDataset(dev_X, dev_Y)
		dev_dataloader = DataLoader(dev_dataset)


		#trianing dataset 
		train_X = []
		train_Y = []
		train_slice =  [] 
		for tr_idx in range(len(X)):
			if tr_idx != tes_idx and tr_idx != dev_idx:
				train_X.extend(X[tr_idx])
				train_Y.extend([Y[tr_idx]]* X_counts[tr_idx])
				train_slice.append(len(X[tr_idx]))

		print(train_slice)
		#convert to pytorch 
		print(np.array(train_X).shape)
		print(np.array(train_Y).shape)
		train_X = torch.FloatTensor(train_X)
		train_Y = torch.FloatTensor(train_Y)

		train_dataset = TensorDataset(train_X, train_Y) # create datset
		train_dataloader = DataLoader(train_dataset)

		# for base and tx 
		model = my_autoencoder(encoding_dim=9, label_dim=2)
		optimizer = torch.optim.Adam(model.parameters())
		model = train_model(model, optimizer, train_dataloader, dev_dataloader, train_slice, dev_slice)
		test_model(model, test_dataloader)


	# for R/NR/HD
	'''
	model2 = my_autoencoder(encoding_dim=9, label_dim=3)
	optimizer = torch.optim.Adam(model.parameters())
	model = train_model(model2, optimizer, train_generator, dev_generator)
	test_model(model2, test_generator)
	'''

main()





