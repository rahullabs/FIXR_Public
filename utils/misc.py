import os
import torch
import torch.nn.functional as F
import math

def process_domain_id(ids):
    domain_id_split = ids.split(',')
    domain_id = []
    for idx in range(len(domain_id_split)):
        # usr_split[idx] = int(usr_split[idx])
        # assert int(usr_split[idx]), "This is not a Number"
        
        domain_id_split[idx] = int(domain_id_split[idx])
        
        domain_folder = str(domain_id_split[idx])
        domain_id.append(domain_folder)
    return domain_id

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def mk_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
        return directory
    else:
        return directory
    
    
def get_correlation_matrix(feat, P_order=2, gamma=0.4):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(P_order+1):
			corr_mat += math.exp(-2*gamma) * (2*gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat