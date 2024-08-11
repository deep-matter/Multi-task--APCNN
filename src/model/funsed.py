import torch.nn as nn
import torch
import torch.nn.Functional as F

class Fusing(nn.Module):
	def __init__(self,config, **kargs):
		super(Fusing,self).__init__()

		self.linear1 = nn.linear(self.arg.in_features, self.args.out_features)
		self.linear2 = nn.linear(self.arg.in_features, self.args.out_features)

		self.Fusing = nn.MultiAttention()

	def forward(self, graph , seq):
		seq = self.linear1(seq)
		graph = self.linear2(graph)

		binding = torch.cat([seq, graph ], dim=-1 )

		return self.MultiAttention(binding)
