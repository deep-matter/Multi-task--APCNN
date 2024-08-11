import torch.nn as nn 
import torch 
import numpy as np


## here thr first the build the layaers 

class LinaerProject(nn.Module):
	def __init___(self,config , **kargs):
		super(LinearProject,self).__init___()
		"""
		the module is reponsbel for the projection the feature into the comparssin into 
		the lantent representation following Agrs

		args:
		sefl.agrs : the condiguration the hyperparameters 
		return : the hidden feature the state of token Sequneces 
		"""
		self.arg = config
		self.projecttion = nn.Linear(self.arg.in_features, self.args.out_features)
		self.norm = nn.LayerNorm(self..args.out_features * 2)

	def forwsrd(self,x):
		batch ,  x.size(0), x.size(1)
		proj = self.projecttion(x)
		normalize = self.norm(proj)
		return normalize.reshape(batch, normalize.shape[1])






