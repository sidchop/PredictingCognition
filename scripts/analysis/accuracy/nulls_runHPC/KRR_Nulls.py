import os
import random
import numpy as np
import pandas as pd
import sys

sys.path.append('/gpfs/milgram/project/holmes/sc2998/MM/analysis/Meta_matching_models/v1.0')
seed = 42
random.seed(seed)
np.random.seed(seed)
batch_size = 16
path_repo = '/gpfs/milgram/project/holmes/sc2998/MM/analysis/Meta_matching_models'
path_v1 = os.path.join(path_repo, 'v1.0')
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

#args 1 = int (dataset: 0=hcpep, 1=tcp, 2=cnp)
#args 2 = perms
ind = int(sys.argv[1])
perms = int(sys.argv[2])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, zscore
from CBIG_model_pytorch import stacking, stacking_nulls, covariance_rowwise
from sklearn.metrics import explained_variance_score, r2_score
from torch.utils.data import DataLoader
from CBIG_model_pytorch import multi_task_dataset
from sklearn.utils import shuffle
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_v1 = os.path.join(path_repo, 'v1.0')
path_model_weight = os.path.join(path_v1, 'CBIG_ukbb_dnn_Zscore.pkl_torch') #change DNN model here
net = torch.load(path_model_weight,   map_location=torch.device('cpu') )
net.to(device)
net.train(False)


# In[1]:

#HCP_EP
fc_data_hcpep = np.loadtxt('//gpfs/milgram/project/holmes/sc2998/MM/analysis/data/HCP_EP/brain/FC_n163_gsr.txt')


#TCP (aka PC)
fc_data_tcp = np.loadtxt('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/PC/brain/FC_mean_matched_n101_GSR.txt')


#CNP (aka UCLA)
fc_data_cnp = np.loadtxt('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/UCLA/brain/FC_n256_matched_GSR.txt')
# load in FC data for each of the three clincial datasets


# In[2]:
# load in global cognitve function princomp scores

cogpc_hcpep = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/HCP_EP/behaviour/pheno_cog_PC_n145.txt', sep=" ")

cogpc_tcp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/PC/behaviour/pheno_cog_PC_n101.txt', sep=" ")

cogpc_cnp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/UCLA/behaviour/pheno_cog_PC_n256.txt', sep=" ")


# In[3]:
# load in covars, clean and zscore

covars_hcpep = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/HCP_EP/behaviour/covars_age_sex_fd.txt', sep=" ")
covars_hcpep.sex[covars_hcpep.sex == 'M'] = 1
covars_hcpep.sex[covars_hcpep.sex == 'F'] = 0
covars_hcpep.interview_age = covars_hcpep.interview_age / 12
covars_hcpep = covars_hcpep.drop(['src_subject_id'], axis=1) 

covars_tcp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/PC/behaviour/covars_age_sex_fd.txt', sep=" ")
covars_tcp.Gender[covars_tcp.Gender == 'M'] = 1
covars_tcp.Gender[covars_tcp.Gender == 'F'] = 0
covars_tcp = covars_tcp.drop(['SUBJECT_ID'], axis=1) 

covars_cnp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/UCLA/behaviour/covars_age_sex_fd.txt', sep=" ")
covars_cnp.gender[covars_cnp.gender == 'M'] = 1
covars_cnp.gender[covars_cnp.gender== 'F'] = 0
covars_cnp = covars_cnp.drop(['V1'], axis=1) 

# load in selected hyperparams from obs model
hp_hcpep = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/output/MM_stacking/hcpep/hcpep_best_param_cogPC.txt', sep = " ", header = None)
hp_tcp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/output/MM_stacking/pc/tcp_best_param_cogPC.txt', sep = " ", header = None)
hp_cnp = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/output/MM_stacking/ucla/cnp_best_param_cogPC.txt', sep = " ", header = None)



# data prep
# select subjects which have both fc and cog data 
hcp_ep_subjlist = pd.read_csv('/gpfs/milgram/project/holmes/sc2998/MM/analysis/data/HCP_EP/behaviour/fmri_subj_id_order_n163.txt', sep=" ")
hcp_ep_inclu_subj_index = hcp_ep_subjlist["x"].isin(cogpc_hcpep["src_subject_id"])
fc_data_hcpep = fc_data_hcpep[hcp_ep_inclu_subj_index,:]

# remove id and group cols. 
cogpc_hcpep = cogpc_hcpep.drop(['src_subject_id'], axis=1) 
cogpc_hcpep = cogpc_hcpep[['Cog_PC1']]
cogpc_tcp = cogpc_tcp.drop(['SUBJECT_ID', 'Group'], axis=1) 
cogpc_cnp = cogpc_cnp.drop(['V1'], axis=1) 

#convert to np array
cogpc_hcpep = cogpc_hcpep.to_numpy()
cogpc_tcp = cogpc_tcp.to_numpy()
cogpc_cnp = cogpc_cnp.to_numpy()

#store cog and fc abd covars data into lists 
fc_list = [fc_data_hcpep , fc_data_tcp , fc_data_cnp]
cog_list = [cogpc_hcpep, cogpc_tcp , cogpc_cnp]
covars_list = [covars_hcpep, covars_tcp, covars_cnp]
hp_list = [hp_hcpep,hp_tcp, hp_cnp ]

names = ['hcpep', 'tcp', 'cnp']

# In[4]:
# Model options
ind=ind #0=hcpep; 1=tcp; 2=cnp
perms = perms #set up number of splits 
regress_covars = False
regress_from_brian = False #if True, will regress covars from brain 
GSR = True
# In[5]:

kf_pearson_results = []
kf_COD_results = []

x_input = fc_list[ind]
y_input = cog_list[ind]
hp  = hp_list[ind].drop([0], axis=1) 

#define storage arrays
corr = np.zeros((y_input.shape[1],perms))
r2 = np.zeros((y_input.shape[1],perms))

#zscore fc data for GSR-DNN, but leave unstanderdized if regressing covars from brain data (zscore after regression)
if regress_from_brian == False and GSR  == True: 
    fc_list = [zscore(fc_data_hcpep) , zscore(fc_data_tcp) , zscore(fc_data_cnp)]
    
#if covars are being regressed then cbind x/y and covars
if regress_covars == True:
    if regress_from_brian == False:
        y_input = np.c_[y_input, covars_list[ind]]
    if regress_from_brian == True:
        x_input = np.c_[x_input, covars_list[ind]]

for k in range(perms):
    y_input_permuted = shuffle(y_input,  random_state=k)
    
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_input_permuted, train_size=0.7, random_state=k)
    
    if regress_covars == True:
        if regress_from_brian == False:
            cog_pc = y_train[:,0]
            cog_pc = cog_pc.reshape(-1,1)
            #covars = y_train[:,3].reshape(-1,1) #hm only
            covars = y_train[:,1:4]  # all covars
            model = LinearRegression().fit(covars,cog_pc)
            prediction = model.predict(covars)
            y_train = (cog_pc - prediction)
            betas = model.coef_
            y_test = y_test[:,0] - (model.intercept_ + y_test[:,1]*betas[0,0] +  y_test[:,2]*betas[0,1] +  y_test[:,3]*betas[0,2])
            #y_test = y_test[:,0] - (model.intercept_  +  y_test[:,3]*betas[0,0])
            y_test=y_test.reshape(-1,1)
        if regress_from_brian == True:
             brain_fc = x_train[:,:-3]
             #covars = ##x_train[:,3].reshape(-1,1)
             covars = x_train[:,-3:]
             model = LinearRegression().fit(covars,brain_fc)
             prediction = model.predict(covars)
             x_train = (brain_fc - prediction)
             betas = model.coef_
             x_test = x_test[:,:-3] - (model.intercept_ + x_test[:,:-3]*betas[:,0] +  x_test[:,:-3]*betas[:,1] +  x_test[:,:-3]*betas[:,2])
            # ## y_test = y_test[:,0] - (model.intercept_  +  y_test[:,3]*betas[0,0])
             x_test = x_test.astype('float64')
             x_train = x_train.astype('float64')
             if GSR == True:
                 temp_zx = zscore(np.concatenate((x_train,x_test)).astype('float64'))
                 temp_zx  = np.split(temp_zx, [x_train.shape[0]])
                 x_train = temp_zx[0]
                 x_test = temp_zx[1]                                                                      
    
    for s in range(y_train.shape[1]):
            #rand_alpha=np.random.choice(hp.iloc[[s]].to_numpy()[0,:])
            #y_test_pred, _  = stacking_nulls(x_train, x_test, y_train[:,s], alpha=rand_alpha)
            y_test_pred, _, _  = stacking(x_train, x_test, y_train[:,s])
	    #y_train_haufe[:,s], _, _ = stacking(x_train, x_train, y_train[:,s], splits=5)
            corr[s,k] = pearsonr(y_test_pred, y_test)[0]
            r2[s,k] = r2_score(y_test_pred, y_test)
            #cov = covariance_rowwise(x_train, y_train_haufe) + np.squeeze(cov)       
           
    print(ind, k)

varnames = ["PC1"]
kf_pearson_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(corr)], axis=1)
kf_COD_results = pd.concat([pd.DataFrame(varnames), pd.DataFrame(r2)], axis=1)
#kf_Haufe_results = pd.DataFrame(kf_Haufe_results, columns=varnames)

output_path = '/gpfs/milgram/project/holmes/sc2998/MM/analysis/output/KRR/nulls/'

kf_pearson_results.to_csv(str(output_path  + names[ind] + '_' + 'pearsonr_cogPC_GSR_nulls.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

kf_COD_results.to_csv(str(output_path + names[ind] + '_' + 'COD_cogPC_GSR_nulls.txt'), 
                           sep=' ',
                           index=False,
                           header=False)

#kf_Haufe_results.to_csv(str(output_path  + names[ind] + '_' + 'haufe_cogPC_GSR_nulls.txt'), 
#                           sep=' ',
#                           index=False,
#                           header=False)
