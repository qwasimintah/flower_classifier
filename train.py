# Imports here

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



def check_for_gpu():
	# check if CUDA is available
	train_on_gpu = torch.cuda.is_available()

	if not train_on_gpu:
	    print('CUDA is not available.  Training on CPU ...')
	else:
	    print('CUDA is available!  Training on GPU ...')



def train_loop(epochs, lr, dataloaders, valid_dataloaders, train_on_gpu ):

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	#sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
	steps = 0
	valid_loss_min = np.Inf
	best_model_weights = model.state_dict()

	train_losses, test_losses = [], []
	for e in range(epochs):
	    running_loss = 0
	    train_loss = 0.0
	    valid_loss = 0.0
	    for images, labels in dataloaders:
	        if train_on_gpu:
	            images, labels = images.cuda(), labels.cuda()
	        optimizer.zero_grad()
	        log_ps = model(images)
	        loss = criterion(log_ps, labels)
	        loss.backward()
	        optimizer.step()
	        train_loss += loss.item()*images.size(0)
	        running_loss += loss.item()
	        
	    else:
	        test_loss = 0
	        accuracy = 0
	        
	        # Turn off gradients for validation, saves memory and computations
	        with torch.no_grad():
	            model.eval()
	            for images, labels in valid_dataloaders:
	                if train_on_gpu:
	                    images, labels = images.cuda(), labels.cuda()              
	                log_ps = model(images)
	                test_loss += criterion(log_ps, labels)
	                
	                ps = torch.exp(log_ps)
	                top_p, top_class = ps.topk(1, dim=1)
	                equals = top_class == labels.view(*top_class.shape)
	                accuracy += torch.mean(equals.type(torch.FloatTensor))
	                valid_loss += test_loss.item()*images.size(0)
	                
	        
	        model.train()
	        
	        train_losses.append(running_loss/len(dataloaders))
	        test_losses.append(test_loss/len(valid_dataloaders))

	        print("Epoch: {}/{}.. ".format(e+1, epochs),
	              "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders)),
	              "Test Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
	              "Test Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))
	        
	    if valid_loss <= valid_loss_min:
	        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
	        valid_loss_min,
	        valid_loss))
	        
	        checkpoint = {
	              'state_dict': model.state_dict(),
	              'class_to_idx': image_datasets.class_to_idx
	             }
	        best_model_weights = model.state_dict()
	        torch.save(checkpoint, 'drives/checkpoint_resnet.pth')
	        valid_loss_min = valid_loss


