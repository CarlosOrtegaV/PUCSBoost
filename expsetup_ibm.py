from pos_noisyneg.utils import make_noisy_negatives
from pos_noisyneg.utils import ave_savings_score

from ecsmodels.methodologies.cs_boost import CSBoost

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import expit
import xgboost as xgb

def binarize_prob(prob, threshold=0.50):
  
  labels = np.zeros(prob.shape, dtype=int)
  labels[prob>threshold] = 1
  return labels

def top_kperc_precision(y, prob, k_perc=0.01):
   
    k =  np.round(len(prob)*k_perc).astype(int)
    ix_sorted_desc = np.argsort(prob)[::-1][:k]
    top_kperc_precision_  = np.mean(y[ix_sorted_desc])

    return top_kperc_precision_

def top_kperc_savings(y, prob, amount, cf=10.0, k_perc=0.01):
   
    k =  np.round(len(prob)*k_perc).astype(int)
    ix_sorted_desc = np.argsort(prob)[::-1][:k]
    savs_numerator = np.sum(y[ix_sorted_desc] * amount[ix_sorted_desc] - cf)
    savs_denominator = np.sum(y * amount)
    top_kperc_savings_ = savs_numerator / savs_denominator

    return top_kperc_savings_

#### Cost Parameters ####
cf = 10
# List of Configurations
list_random_state = list(np.arange(10))  # Number of repetitions for data partition

list_flip_ratio = {
                    0.25: {
                        'flip_ratio': 0.25,
                        'scaling_factor': 0.9,
                        'lower_limit_prob': 0.2
                        },
                    0.50: {
                        'flip_ratio': 0.50,
                        'scaling_factor': 0.6,
                        'lower_limit_prob': 0.2
                        },
                    0.75: {
                        'flip_ratio': 0.75,
                        'scaling_factor': 0.3,
                        'lower_limit_prob': 0.2
                        }
                    }


list_label_noise = ['scar','sarpg','sarcost']

# List of metrics

# XGBoost
list_ap_xgb_ce = []
list_as_xgb_ce = []
list_roc_xgb_ce = []
list_f1_opt_xgb_ce = []
list_recall_xgb_ce = []
list_precision_xgb_ce = []
list_top1_precision_xgb_ce = []
list_top1_savings_xgb_ce = []

# CSBoost
list_ap_xgb_csboost = []
list_as_xgb_csboost = []
list_roc_xgb_csboost = []
list_f1_opt_xgb_csboost = []
list_recall_xgb_csboost = []
list_precision_xgb_csboost = []
list_top1_precision_xgb_csboost = []
list_top1_savings_xgb_csboost = []

# nnPU SCAR
list_ap_xgb_nnpu_scar = []
list_as_xgb_nnpu_scar = []
list_roc_xgb_nnpu_scar = []
list_f1_opt_xgb_nnpu_scar = []
list_recall_xgb_nnpu_scar = []
list_precision_xgb_nnpu_scar = []
list_top1_precision_xgb_nnpu_scar = []
list_top1_savings_xgb_nnpu_scar = []

# nnPU SARPG
list_ap_xgb_nnpu_sarpg = []
list_as_xgb_nnpu_sarpg = []
list_roc_xgb_nnpu_sarpg = []
list_f1_opt_xgb_nnpu_sarpg = []
list_recall_xgb_nnpu_sarpg = []
list_precision_xgb_nnpu_sarpg = []
list_top1_precision_xgb_nnpu_sarpg = []
list_top1_savings_xgb_nnpu_sarpg = []

# nnPU SARCost
list_ap_xgb_nnpu_sarcost = []
list_as_xgb_nnpu_sarcost = []
list_roc_xgb_nnpu_sarcost = []
list_f1_opt_xgb_nnpu_sarcost = []
list_recall_xgb_nnpu_sarcost = []
list_precision_xgb_nnpu_sarcost = []
list_top1_precision_xgb_nnpu_sarcost = []
list_top1_savings_xgb_nnpu_sarcost = []

# CSPU SCAR
list_ap_xgb_cspu_scar = []
list_as_xgb_cspu_scar = []
list_roc_xgb_cspu_scar = []
list_f1_opt_xgb_cspu_scar = []
list_recall_xgb_cspu_scar = []
list_precision_xgb_cspu_scar = []
list_top1_precision_xgb_cspu_scar = []
list_top1_savings_xgb_cspu_scar = []

# CSPU SARPG
list_ap_xgb_cspu_sarpg = []
list_as_xgb_cspu_sarpg = []
list_roc_xgb_cspu_sarpg = []
list_f1_opt_xgb_cspu_sarpg = []
list_recall_xgb_cspu_sarpg = []
list_precision_xgb_cspu_sarpg = []
list_top1_precision_xgb_cspu_sarpg = []
list_top1_savings_xgb_cspu_sarpg = []

# CSPU SARCost
list_ap_xgb_cspu_sarcost = []
list_as_xgb_cspu_sarcost = []
list_roc_xgb_cspu_sarcost = []
list_f1_opt_xgb_cspu_sarcost = []
list_recall_xgb_cspu_sarcost = []
list_precision_xgb_cspu_sarcost = []
list_top1_precision_xgb_cspu_sarcost = []
list_top1_savings_xgb_cspu_sarcost = []

# PU-CSBoost SCAR EK
list_ap_xgb_pucsboost_scar_ek = []
list_as_xgb_pucsboost_scar_ek = []
list_roc_xgb_pucsboost_scar_ek = []
list_f1_opt_xgb_pucsboost_scar_ek = []
list_recall_xgb_pucsboost_scar_ek = []
list_precision_xgb_pucsboost_scar_ek = []
list_top1_precision_xgb_pucsboost_scar_ek = []
list_top1_savings_xgb_pucsboost_scar_ek = []

# PU-CSBoost SARPG
list_ap_xgb_pucsboost_sarpg = []
list_as_xgb_pucsboost_sarpg = []
list_roc_xgb_pucsboost_sarpg = []
list_f1_opt_xgb_pucsboost_sarpg = []
list_recall_xgb_pucsboost_sarpg = []
list_precision_xgb_pucsboost_sarpg = []
list_top1_precision_xgb_pucsboost_sarpg = []
list_top1_savings_xgb_pucsboost_sarpg = []

# PU-CSBoost SARCost
list_ap_xgb_pucsboost_sarcost = []
list_as_xgb_pucsboost_sarcost = []
list_roc_xgb_pucsboost_sarcost = []
list_f1_opt_xgb_pucsboost_sarcost = []
list_recall_xgb_pucsboost_sarcost = []
list_precision_xgb_pucsboost_sarcost = []
list_top1_precision_xgb_pucsboost_sarcost = []
list_top1_savings_xgb_pucsboost_sarcost = []

set_thresholds = np.linspace(0.01, 0.99, num=100)

#### Generate Data ####
df = pd.read_csv('./data/ibm_scaled.csv')
df = df[df['Amount']>0]
df=df.drop(df.std()[df.std() == 0].index.values, axis=1)

y = np.asarray(df['Class'])
amount = np.asarray(df['Amount'])

ix_pos = np.where(y)[0]

#### Data Transformation X ####
X = df.iloc[:,:-2].copy(deep=True)
X['L_Amount'] = np.log(amount)

for q in tqdm(list_label_noise, desc='Label Noise'):

    for j in tqdm(list_flip_ratio, desc='Flip Ratio'):
        
        for r in tqdm(list_random_state, desc='Data Partition'):
          
# =============================================================================
#           DATA PREPROCESSING       
# =============================================================================

            noisy_y, _, __ = make_noisy_negatives(y,
                                                  X=X,
                                                  flip_ratio=list_flip_ratio[j]['flip_ratio'],
                                                  scaling_factor=list_flip_ratio[j]['scaling_factor'],
                                                  lower_limit_prob=list_flip_ratio[j]['lower_limit_prob'],
                                                  cost_variable=amount,
                                                  label_noise = q,
                                                  random_state = r)
            
            X = np.asarray(X)
            
            ## Class Y Generation
            class_y = noisy_y.copy()
            class_y[np.logical_and(np.array(noisy_y == 0), np.array(y == 1))] = 2
            
            conditions = [class_y == 0, class_y == 2, class_y == 1]
            values = ['negs', 'noisy_negs', 'pos']
            strata_label = np.select(conditions, values)
            
            print('Percentage of Positives : ', np.mean(y))
            
            ### Training & Test Dataset ###
                        
            rs = StratifiedShuffleSplit(n_splits = 1, train_size = 0.70, random_state = 123)
            ix_tr, ix_ts = [(a, b) for a, b in rs.split(X, strata_label)][0]
              
            #### Feature Preprocessing ####
            
            standard_scaler = StandardScaler().fit(X[ix_tr])
            X_tr = standard_scaler.transform(X[ix_tr])
            X_ts = standard_scaler.transform(X[ix_ts])
                        
# =============================================================================
#           MODELING
# =============================================================================
            
            # Data Split
            rs_xgb = StratifiedShuffleSplit(n_splits = 1, test_size = 0.50, random_state = r)
            ix_xgb_tr, ix_xgb_val = [(a, b) for a, b in rs_xgb.split(X_tr, strata_label[ix_tr])][0]
            
            # Training and Valdiation Cost Matrix for IDCS techniques
            cost_matrix_xgb_tr = np.zeros((len(X[ix_tr][ix_xgb_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
            cost_matrix_xgb_tr[:, 0, 0] = 0.0
            cost_matrix_xgb_tr[:, 0, 1] = amount[ix_tr][ix_xgb_tr]
            cost_matrix_xgb_tr[:, 1, 0] = cf
            cost_matrix_xgb_tr[:, 1, 1] = cf
        
            cost_matrix_xgb_vl = np.zeros((len(X[ix_tr][ix_xgb_val]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
            cost_matrix_xgb_vl[:, 0, 0] = 0.0
            cost_matrix_xgb_vl[:, 0, 1] = amount[ix_tr][ix_xgb_val]
            cost_matrix_xgb_vl[:, 1, 0] = cf
            cost_matrix_xgb_vl[:, 1, 1] = cf
            
            # Configuration of XGBoost Parameters
            params = {'random_state': r, 
                      'tree_method': 'exact',
                      'objective':'binary:logistic',
                      'verbosity': 0, 
                      'reg_alpha': 0,
                      'reg_lambda':0}
            
# --------------------------- XGBoost Cross-Entropy -------------------------- #

            dtrain = xgb.DMatrix(X_tr[ix_xgb_tr], label=noisy_y[ix_tr][ix_xgb_tr])
            dval = xgb.DMatrix(X_tr[ix_xgb_val], label=noisy_y[ix_tr][ix_xgb_val])   
            
            m_xgb_ce = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, early_stopping_rounds=50,
                            evals=[(dval, 'eval')], verbose_eval=False)
            
            prob_xgb_tr = m_xgb_ce.inplace_predict(X_tr[ix_xgb_tr])
            
            prob_xgb_vl = m_xgb_ce.inplace_predict(X_tr[ix_xgb_val])

            prob_xgb_ce = m_xgb_ce.inplace_predict(X_ts)

            # Non-threshold measures
            list_roc_xgb_ce.append(roc_auc_score(y[ix_ts], prob_xgb_ce))
            list_ap_xgb_ce.append(average_precision_score(y[ix_ts], prob_xgb_ce))
            print('AP XGBoost: ', list_ap_xgb_ce[-1])
            list_as_xgb_ce.append(ave_savings_score(y[ix_ts], prob_xgb_ce, amount[ix_ts], cf=cf))
            print('AS XGBoost: ', list_as_xgb_ce[-1])
            list_top1_precision_xgb_ce.append(top_kperc_precision(y[ix_ts], prob_xgb_ce))
            list_top1_savings_xgb_ce.append(top_kperc_savings(y[ix_ts], prob_xgb_ce, amount[ix_ts], cf=cf))

            # Threshold-dependent measures       
          
            prob_vl_xgb_ce = m_xgb_ce.inplace_predict(X_tr[ix_xgb_val])
            set_labels_xgb_ce = [binarize_prob(prob_vl_xgb_ce, threshold=i) for i in set_thresholds]
            set_f1_scores_xgb_ce = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_ce]
            ix_max_f1score = np.where(set_f1_scores_xgb_ce==np.max(set_f1_scores_xgb_ce))[0][0]
            labels_xgb_ce = binarize_prob(prob_xgb_ce, threshold=set_thresholds[ix_max_f1score])
            
            # Optimal F1-score
            list_f1_opt_xgb_ce.append(f1_score(y[ix_ts], labels_xgb_ce))
            list_recall_xgb_ce.append(recall_score(y[ix_ts], labels_xgb_ce))
            list_precision_xgb_ce.append(precision_score(y[ix_ts], labels_xgb_ce))
              
# ------------------------------ XGBoost CSBoost ----------------------------- #
        
            m_xgb_csboost = CSBoost(obj='aec').fit(X_tr[ix_xgb_tr], 
                                                noisy_y[ix_tr][ix_xgb_tr], 
                                                X_tr[ix_xgb_val], 
                                                noisy_y[ix_tr][ix_xgb_val], 
                                                cost_matrix_train=cost_matrix_xgb_tr, 
                                                cost_matrix_val=cost_matrix_xgb_vl)
            
            prob_xgb_csboost = expit(m_xgb_csboost.inplace_predict(X_ts))
      
            # Non-threshold measures
            list_roc_xgb_csboost.append(roc_auc_score(y[ix_ts], prob_xgb_csboost))
            list_ap_xgb_csboost.append(average_precision_score(y[ix_ts], prob_xgb_csboost))
            print('AP CSBoost: ', list_ap_xgb_csboost[-1])
            list_as_xgb_csboost.append(ave_savings_score(y[ix_ts], prob_xgb_csboost, amount[ix_ts], cf=cf))
            print('AS CSBoost: ', list_as_xgb_csboost[-1])
            list_top1_precision_xgb_csboost.append(top_kperc_precision(y[ix_ts], prob_xgb_csboost))
            list_top1_savings_xgb_csboost.append(top_kperc_savings(y[ix_ts], prob_xgb_csboost, amount[ix_ts], cf=cf))
            
            # Threshold-dependent measures    
            prob_vl_xgb_csboost = m_xgb_csboost.inplace_predict(X_tr[ix_xgb_val])
            set_labels_xgb_csboost = [binarize_prob(prob_vl_xgb_csboost, threshold=i) for i in set_thresholds]
            set_f1_scores_xgb_csboost = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_csboost]
            ix_max_f1score = np.where(set_f1_scores_xgb_csboost==np.max(set_f1_scores_xgb_csboost))[0][0]
            labels_xgb_csboost = binarize_prob(prob_xgb_csboost, threshold=set_thresholds[ix_max_f1score])
            
            # Optimal F1-score
            list_f1_opt_xgb_csboost.append(f1_score(y[ix_ts], labels_xgb_csboost))
            list_recall_xgb_csboost.append(recall_score(y[ix_ts], labels_xgb_csboost))
            list_precision_xgb_csboost.append(precision_score(y[ix_ts], labels_xgb_csboost))

# ---------------------------------------------------------------------------- #
#                            Labeling Mechanism SCAR                           #
# ---------------------------------------------------------------------------- #

            prior_y = np.mean(y)
            prior_s = np.mean(noisy_y)
            c = prior_s/prior_y

            # Training Proba. Unlabeled to be Positive
            prob_unlpos_tr_c = (1.0 - c)/c
            prob_unlpos_tr_ = np.mean(noisy_y)/(1-np.mean(noisy_y))
            
            prob_unlpos_tr = prob_unlpos_tr_c*prob_unlpos_tr_
            prob_unlpos_tr = np.min((prob_unlpos_tr, 1.0))
            
            # Validation Proba. Unlabeled to be Positive
            prob_unlpos_vl_c = (1.0 - c)/c
            prob_unlpos_vl_ = np.mean(noisy_y)/(1.0-np.mean(noisy_y))

            prob_unlpos_vl= prob_unlpos_vl_c*prob_unlpos_vl_
            prob_unlpos_vl = np.min((prob_unlpos_vl, 1.0))
            
            cost_matrix_xgb_pucsboost_scar_tr = np.zeros((len(X_tr[ix_xgb_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
            cost_matrix_xgb_pucsboost_scar_tr[:, 0, 0] = amount[ix_tr][ix_xgb_tr]*prob_unlpos_tr
            cost_matrix_xgb_pucsboost_scar_tr[:, 0, 1] = amount[ix_tr][ix_xgb_tr]
            cost_matrix_xgb_pucsboost_scar_tr[:, 1, 0] = cf
            cost_matrix_xgb_pucsboost_scar_tr[:, 1, 1] = cf
            
            cost_matrix_xgb_pucsboost_scar_vl = np.zeros((len(X_tr[ix_xgb_val]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
            cost_matrix_xgb_pucsboost_scar_vl[:, 0, 0] = amount[ix_tr][ix_xgb_val]*prob_unlpos_vl
            cost_matrix_xgb_pucsboost_scar_vl[:, 0, 1] = amount[ix_tr][ix_xgb_val]
            cost_matrix_xgb_pucsboost_scar_vl[:, 1, 0] = cf
            cost_matrix_xgb_pucsboost_scar_vl[:, 1, 1] = cf

# ----------------------- non-negative PU XGBoost SCAR ----------------------- #

            m_xgb_nnpu_scar = CSBoost(obj='nnpu_scar',
                                      prior_y=np.mean(y),
                                      prob_lab=np.mean(noisy_y),  
                                      validation=True,
                                      random_state=r).fit(X_tr[ix_xgb_tr], 
                                                          noisy_y[ix_tr][ix_xgb_tr], 
                                                          X_tr[ix_xgb_val], 
                                                          noisy_y[ix_tr][ix_xgb_val])
            
            prob_xgb_nnpu_scar = expit(m_xgb_nnpu_scar.inplace_predict(X_ts))
            
            # Non-threshold measures
            list_roc_xgb_nnpu_scar.append(roc_auc_score(y[ix_ts], prob_xgb_nnpu_scar))
            list_ap_xgb_nnpu_scar.append(average_precision_score(y[ix_ts], prob_xgb_nnpu_scar))
            print('AP nnPU SCAR: ', list_ap_xgb_nnpu_scar[-1])
            list_as_xgb_nnpu_scar.append(ave_savings_score(y[ix_ts], prob_xgb_nnpu_scar, amount[ix_ts], cf=cf))
            print('AS nnPU SCAR: ', list_as_xgb_nnpu_scar[-1])
            list_top1_precision_xgb_nnpu_scar.append(top_kperc_precision(y[ix_ts], prob_xgb_nnpu_scar))
            list_top1_savings_xgb_nnpu_scar.append(top_kperc_savings(y[ix_ts], prob_xgb_nnpu_scar, amount[ix_ts], cf=cf))
            
            # Threshold-dependent measures    
            prob_vl_xgb_nnpu_scar = m_xgb_nnpu_scar.inplace_predict(X_tr[ix_xgb_val])
            set_labels_xgb_nnpu_scar = [binarize_prob(prob_vl_xgb_nnpu_scar, threshold=i) for i in set_thresholds]
            set_f1_scores_xgb_nnpu_scar = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_nnpu_scar]
            ix_max_f1score = np.where(set_f1_scores_xgb_nnpu_scar==np.max(set_f1_scores_xgb_nnpu_scar))[0][0]
            labels_xgb_nnpu_scar = binarize_prob(prob_xgb_nnpu_scar, threshold=set_thresholds[ix_max_f1score])
            
            # Optimal F1-score
            list_f1_opt_xgb_nnpu_scar.append(f1_score(y[ix_ts], labels_xgb_nnpu_scar))
            list_recall_xgb_nnpu_scar.append(recall_score(y[ix_ts], labels_xgb_nnpu_scar))
            list_precision_xgb_nnpu_scar.append(precision_score(y[ix_ts], labels_xgb_nnpu_scar))
            
# --------------------- XGBoost PUCSBoost SCAR Elkan-Noto -------------------- #
        
            m_xgb_pucsboost_scar_ek = CSBoost(obj='aec',
                                              random_state=r,
                                              validation=True).fit(X_tr[ix_xgb_tr], 
                                                                   noisy_y[ix_tr][ix_xgb_tr], 
                                                                   X_tr[ix_xgb_val], 
                                                                   noisy_y[ix_tr][ix_xgb_val], 
                                                                   cost_matrix_train=cost_matrix_xgb_pucsboost_scar_tr, 
                                                                   cost_matrix_val=cost_matrix_xgb_pucsboost_scar_vl)
            
            prob_xgb_pucsboost_scar_ek = expit(m_xgb_pucsboost_scar_ek.inplace_predict(X_ts))
          
            # Non-threshold measures
            list_roc_xgb_pucsboost_scar_ek.append(roc_auc_score(y[ix_ts], prob_xgb_pucsboost_scar_ek))
            list_ap_xgb_pucsboost_scar_ek.append(average_precision_score(y[ix_ts], prob_xgb_pucsboost_scar_ek))
            print('AP PUCSBoost SCAR EK: ', list_ap_xgb_pucsboost_scar_ek[-1])
            list_as_xgb_pucsboost_scar_ek.append(ave_savings_score(y[ix_ts], prob_xgb_pucsboost_scar_ek, amount[ix_ts], cf=cf))
            print('AS PUCSBoost SCAR EK: ', list_as_xgb_pucsboost_scar_ek[-1])
            list_top1_precision_xgb_pucsboost_scar_ek.append(top_kperc_precision(y[ix_ts], prob_xgb_pucsboost_scar_ek))
            list_top1_savings_xgb_pucsboost_scar_ek.append(top_kperc_savings(y[ix_ts], prob_xgb_pucsboost_scar_ek, amount[ix_ts], cf=cf))
    
            # Threshold-dependent measures        
            prob_vl_xgb_pucsboost_scar_ek = m_xgb_pucsboost_scar_ek.inplace_predict(X_tr[ix_xgb_val])
            set_labels_xgb_pucsboost_scar_ek = [binarize_prob(prob_vl_xgb_pucsboost_scar_ek, threshold=i) for i in set_thresholds]
            set_f1_scores_xgb_pucsboost_scar_ek = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_pucsboost_scar_ek]
            ix_max_f1score = np.where(set_f1_scores_xgb_pucsboost_scar_ek==np.max(set_f1_scores_xgb_pucsboost_scar_ek))[0][0]
            labels_xgb_pucsboost_scar_ek = binarize_prob(prob_xgb_pucsboost_scar_ek, threshold=set_thresholds[ix_max_f1score])
            
            # Optimal F1-score
            list_f1_opt_xgb_pucsboost_scar_ek.append(f1_score(y[ix_ts], labels_xgb_pucsboost_scar_ek))
            list_recall_xgb_pucsboost_scar_ek.append(recall_score(y[ix_ts], labels_xgb_pucsboost_scar_ek))
            list_precision_xgb_pucsboost_scar_ek.append(precision_score(y[ix_ts], labels_xgb_pucsboost_scar_ek))

# ----------------------------- CSPU XGBoost SCAR ---------------------------- #
        
            m_xgb_cspu_scar = CSBoost(obj='puwce_scar',
                                      prior_y=np.mean(y),
                                      prob_lab=np.mean(noisy_y), 
                                      random_state=r,
                                      validation=True).fit(X_tr[ix_xgb_tr], 
                                                           noisy_y[ix_tr][ix_xgb_tr], 
                                                           X_tr[ix_xgb_val], 
                                                           noisy_y[ix_tr][ix_xgb_val],
                                                           cost_matrix_train=cost_matrix_xgb_tr, 
                                                           cost_matrix_val=cost_matrix_xgb_vl)
            
            prob_xgb_cspu_scar = expit(m_xgb_cspu_scar.inplace_predict(X_ts))
          
            # Non-threshold measures
            list_roc_xgb_cspu_scar.append(roc_auc_score(y[ix_ts], prob_xgb_cspu_scar))
            list_ap_xgb_cspu_scar.append(average_precision_score(y[ix_ts], prob_xgb_cspu_scar))
            print('AP CSPU SCAR: ', list_ap_xgb_cspu_scar[-1])
            list_as_xgb_cspu_scar.append(ave_savings_score(y[ix_ts], prob_xgb_cspu_scar, amount[ix_ts], cf=cf))
            print('AS CSPU SCAR: ', list_as_xgb_cspu_scar[-1])
            list_top1_precision_xgb_cspu_scar.append(top_kperc_precision(y[ix_ts], prob_xgb_cspu_scar))
            list_top1_savings_xgb_cspu_scar.append(top_kperc_savings(y[ix_ts], prob_xgb_cspu_scar, amount[ix_ts], cf=cf))
    
            # Threshold-dependent measures        
            prob_vl_xgb_cspu_scar = m_xgb_cspu_scar.inplace_predict(X_tr[ix_xgb_val])
            set_labels_xgb_cspu_scar = [binarize_prob(prob_vl_xgb_cspu_scar, threshold=i) for i in set_thresholds]
            set_f1_scores_xgb_cspu_scar = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_cspu_scar]
            ix_max_f1score = np.where(set_f1_scores_xgb_cspu_scar==np.max(set_f1_scores_xgb_cspu_scar))[0][0]
            labels_xgb_cspu_scar = binarize_prob(prob_xgb_cspu_scar, threshold=set_thresholds[ix_max_f1score])
            
            # Optimal F1-score
            list_f1_opt_xgb_cspu_scar.append(f1_score(y[ix_ts], labels_xgb_cspu_scar))
            list_recall_xgb_cspu_scar.append(recall_score(y[ix_ts], labels_xgb_cspu_scar))
            list_precision_xgb_cspu_scar.append(precision_score(y[ix_ts], labels_xgb_cspu_scar))
                        
# ---------------------------------------------------------------------------- #
#                           Labeling Mechanism SARPG                           #
# ---------------------------------------------------------------------------- #
            if q == 'sarpg':

                k = list_flip_ratio[j]['scaling_factor']
                
                # Propensity Scores
                propensity_lr = LogisticRegression(penalty=None, 
                                                   n_jobs=-1, 
                                                   random_state=r)
                
                propensity_lr.fit(X, y)
                
                proba = k * propensity_lr.predict_proba(X)[:,1]
                
                e = proba[ix_tr]
                e[e<0.01]=0.01
                
                e_tr = e[ix_xgb_tr]
                e_vl = e[ix_xgb_val]
                
                # Training Proba. Unlabeled to be Positive
                
                prob_unlpos_tr_e = (1.0 - e_tr)/e_tr
                
                prob_unlpos_tr_ = prob_xgb_tr/(1.0-prob_xgb_tr)
                
                prob_unlpos_tr = prob_unlpos_tr_e*prob_unlpos_tr_
                prob_unlpos_tr[prob_unlpos_tr>1.0] = 1.0
                
                # Validation Proba. Unlabeled to be Positive
                
                prob_unlpos_vl_e = (1.0 - e_vl)/e_vl

                prob_unlpos_vl_ = prob_xgb_vl/(1.0-prob_xgb_vl)

                prob_unlpos_vl= prob_unlpos_vl_e*prob_unlpos_vl_
                prob_unlpos_vl[prob_unlpos_vl>1.0] = 1.0
                
                cost_matrix_xgb_pucsboost_sarpg_tr = np.zeros((len(X_tr[ix_xgb_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_xgb_pucsboost_sarpg_tr[:, 0, 0] = amount[ix_tr][ix_xgb_tr]*prob_unlpos_tr
                cost_matrix_xgb_pucsboost_sarpg_tr[:, 0, 1] = amount[ix_tr][ix_xgb_tr]
                cost_matrix_xgb_pucsboost_sarpg_tr[:, 1, 0] = cf
                cost_matrix_xgb_pucsboost_sarpg_tr[:, 1, 1] = cf
                
                cost_matrix_xgb_pucsboost_sarpg_vl = np.zeros((len(X_tr[ix_xgb_val]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_xgb_pucsboost_sarpg_vl[:, 0, 0] = amount[ix_tr][ix_xgb_val]*prob_unlpos_vl
                cost_matrix_xgb_pucsboost_sarpg_vl[:, 0, 1] = amount[ix_tr][ix_xgb_val]
                cost_matrix_xgb_pucsboost_sarpg_vl[:, 1, 0] = cf
                cost_matrix_xgb_pucsboost_sarpg_vl[:, 1, 1] = cf  

# ----------------------- non-negative PU XGBoost SARPG ---------------------- #

                m_xgb_nnpu_sarpg = CSBoost(obj='nnpu_sar', 
                                           validation=True,
                                           random_state=r).fit(X_tr[ix_xgb_tr], 
                                                               noisy_y[ix_tr][ix_xgb_tr], 
                                                               X_tr[ix_xgb_val], 
                                                               noisy_y[ix_tr][ix_xgb_val],
                                                               propensity_score_train=e_tr,
                                                               propensity_score_val=e_vl)
                
                prob_xgb_nnpu_sarpg = expit(m_xgb_nnpu_sarpg.inplace_predict(X_ts))
               
                
                # Non-threshold measures
                list_roc_xgb_nnpu_sarpg.append(roc_auc_score(y[ix_ts], prob_xgb_nnpu_sarpg))
                list_ap_xgb_nnpu_sarpg.append(average_precision_score(y[ix_ts], prob_xgb_nnpu_sarpg))
                print('AP nnPU SARPG: ', list_ap_xgb_nnpu_sarpg[-1])
                list_as_xgb_nnpu_sarpg.append(ave_savings_score(y[ix_ts], prob_xgb_nnpu_sarpg, amount[ix_ts], cf=cf))
                print('AS nnPU SARPG: ', list_as_xgb_nnpu_sarpg[-1])
                list_top1_precision_xgb_nnpu_sarpg.append(top_kperc_precision(y[ix_ts], prob_xgb_nnpu_sarpg))
                list_top1_savings_xgb_nnpu_sarpg.append(top_kperc_savings(y[ix_ts], prob_xgb_nnpu_sarpg, amount[ix_ts], cf=cf))
                
                # Threshold-dependent measures    
                prob_vl_xgb_nnpu_sarpg = m_xgb_nnpu_sarpg.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_nnpu_sarpg = [binarize_prob(prob_vl_xgb_nnpu_sarpg, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_nnpu_sarpg = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_nnpu_sarpg]
                ix_max_f1score = np.where(set_f1_scores_xgb_nnpu_sarpg==np.max(set_f1_scores_xgb_nnpu_sarpg))[0][0]
                labels_xgb_nnpu_sarpg = binarize_prob(prob_xgb_nnpu_sarpg, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_nnpu_sarpg.append(f1_score(y[ix_ts], labels_xgb_nnpu_sarpg))
                list_recall_xgb_nnpu_sarpg.append(recall_score(y[ix_ts], labels_xgb_nnpu_sarpg))
                list_precision_xgb_nnpu_sarpg.append(precision_score(y[ix_ts], labels_xgb_nnpu_sarpg))

# -------------------------- XGBoost PUCSBoost SARPG ------------------------- #
    
                m_xgb_pucsboost_sarpg = CSBoost(obj='aec', validation=True).fit(X_tr[ix_xgb_tr], 
                                                                        noisy_y[ix_tr][ix_xgb_tr], 
                                                                        X_tr[ix_xgb_val], 
                                                                        noisy_y[ix_tr][ix_xgb_val], 
                                                                        cost_matrix_train=cost_matrix_xgb_pucsboost_sarpg_tr, 
                                                                        cost_matrix_val=cost_matrix_xgb_pucsboost_sarpg_vl)
                
                prob_xgb_pucsboost_sarpg = expit(m_xgb_pucsboost_sarpg.inplace_predict(X_ts))
                
                # Non-threshold measures
                list_roc_xgb_pucsboost_sarpg.append(roc_auc_score(y[ix_ts], prob_xgb_pucsboost_sarpg))
                list_ap_xgb_pucsboost_sarpg.append(average_precision_score(y[ix_ts], prob_xgb_pucsboost_sarpg))
                print('AP PUCSBoost SARPG : ', list_ap_xgb_pucsboost_sarpg[-1])
                list_as_xgb_pucsboost_sarpg.append(ave_savings_score(y[ix_ts], prob_xgb_pucsboost_sarpg, amount[ix_ts], cf=cf))
                print('AS PUCSBoost SARPG : ', list_as_xgb_pucsboost_sarpg[-1])
                list_top1_precision_xgb_pucsboost_sarpg.append(top_kperc_precision(y[ix_ts], prob_xgb_pucsboost_sarpg))
                list_top1_savings_xgb_pucsboost_sarpg.append(top_kperc_savings(y[ix_ts], prob_xgb_pucsboost_sarpg, amount[ix_ts], cf=cf))
                
                # Threshold-dependent measures        
                prob_vl_xgb_pucsboost_sarpg = m_xgb_pucsboost_sarpg.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_pucsboost_sarpg = [binarize_prob(prob_vl_xgb_pucsboost_sarpg, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_pucsboost_sarpg = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_pucsboost_sarpg]
                ix_max_f1score = np.where(set_f1_scores_xgb_pucsboost_sarpg==np.max(set_f1_scores_xgb_pucsboost_sarpg))[0][0]
                labels_xgb_pucsboost_sarpg = binarize_prob(prob_xgb_pucsboost_sarpg, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_pucsboost_sarpg.append(f1_score(y[ix_ts], labels_xgb_pucsboost_sarpg))
                list_recall_xgb_pucsboost_sarpg.append(recall_score(y[ix_ts], labels_xgb_pucsboost_sarpg))
                list_precision_xgb_pucsboost_sarpg.append(precision_score(y[ix_ts], labels_xgb_pucsboost_sarpg))

# ---------------------------- CSPU XGBoost SARPG ---------------------------- #
        
                m_xgb_cspu_sarpg = CSBoost(obj='puwce_sar',
                                        random_state=r,
                                        validation=True).fit(X_tr[ix_xgb_tr], 
                                                            noisy_y[ix_tr][ix_xgb_tr], 
                                                            X_tr[ix_xgb_val], 
                                                            noisy_y[ix_tr][ix_xgb_val],
                                                            cost_matrix_train=cost_matrix_xgb_tr, 
                                                            cost_matrix_val=cost_matrix_xgb_vl,
                                                            propensity_score_train=e_tr,
                                                            propensity_score_val=e_vl)
                
                prob_xgb_cspu_sarpg = expit(m_xgb_cspu_sarpg.inplace_predict(X_ts))
            
                # Non-threshold measures
                list_roc_xgb_cspu_sarpg.append(roc_auc_score(y[ix_ts], prob_xgb_cspu_sarpg))
                list_ap_xgb_cspu_sarpg.append(average_precision_score(y[ix_ts], prob_xgb_cspu_sarpg))
                print('AP CSPU SARPG: ', list_ap_xgb_cspu_sarpg[-1])
                list_as_xgb_cspu_sarpg.append(ave_savings_score(y[ix_ts], prob_xgb_cspu_sarpg, amount[ix_ts], cf=cf))
                print('AS CSPU SARPG: ', list_as_xgb_cspu_sarpg[-1])
                list_top1_precision_xgb_cspu_sarpg.append(top_kperc_precision(y[ix_ts], prob_xgb_cspu_sarpg))
                list_top1_savings_xgb_cspu_sarpg.append(top_kperc_savings(y[ix_ts], prob_xgb_cspu_sarpg, amount[ix_ts], cf=cf))
        
                # Threshold-dependent measures        
                prob_vl_xgb_cspu_sarpg = m_xgb_cspu_sarpg.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_cspu_sarpg = [binarize_prob(prob_vl_xgb_cspu_sarpg, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_cspu_sarpg = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_cspu_sarpg]
                ix_max_f1score = np.where(set_f1_scores_xgb_cspu_sarpg==np.max(set_f1_scores_xgb_cspu_sarpg))[0][0]
                labels_xgb_cspu_sarpg = binarize_prob(prob_xgb_cspu_sarpg, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_cspu_sarpg.append(f1_score(y[ix_ts], labels_xgb_cspu_sarpg))
                list_recall_xgb_cspu_sarpg.append(recall_score(y[ix_ts], labels_xgb_cspu_sarpg))
                list_precision_xgb_cspu_sarpg.append(precision_score(y[ix_ts], labels_xgb_cspu_sarpg))
                
            else:           
                list_ap_xgb_nnpu_sarpg.append(np.NaN)
                list_as_xgb_nnpu_sarpg.append(np.NaN)
                list_roc_xgb_nnpu_sarpg.append(np.NaN)
                list_f1_opt_xgb_nnpu_sarpg.append(np.NaN)
                list_recall_xgb_nnpu_sarpg.append(np.NaN)
                list_precision_xgb_nnpu_sarpg.append(np.NaN)
                list_top1_precision_xgb_nnpu_sarpg.append(np.NaN)
                list_top1_savings_xgb_nnpu_sarpg.append(np.NaN)

                list_ap_xgb_pucsboost_sarpg.append(np.NaN)
                list_as_xgb_pucsboost_sarpg.append(np.NaN)
                list_roc_xgb_pucsboost_sarpg.append(np.NaN)
                list_f1_opt_xgb_pucsboost_sarpg.append(np.NaN)
                list_recall_xgb_pucsboost_sarpg.append(np.NaN)
                list_precision_xgb_pucsboost_sarpg.append(np.NaN)
                list_top1_precision_xgb_pucsboost_sarpg.append(np.NaN)
                list_top1_savings_xgb_pucsboost_sarpg.append(np.NaN)

                list_ap_xgb_cspu_sarpg.append(np.NaN)
                list_as_xgb_cspu_sarpg.append(np.NaN)
                list_roc_xgb_cspu_sarpg.append(np.NaN)
                list_f1_opt_xgb_cspu_sarpg.append(np.NaN)
                list_recall_xgb_cspu_sarpg.append(np.NaN)
                list_precision_xgb_cspu_sarpg.append(np.NaN)
                list_top1_precision_xgb_cspu_sarpg.append(np.NaN)
                list_top1_savings_xgb_cspu_sarpg.append(np.NaN)

# ---------------------------------------------------------------------------- #
#                           Labeling Mechanism SARCost                         #
# ---------------------------------------------------------------------------- #
            if q == 'sarcost':

                k = list_flip_ratio[j]['scaling_factor']
                
                # Propensity Scores
                propensity_lr = LogisticRegression(penalty=None, 
                                                n_jobs=-1, 
                                                random_state=r)
                
                propensity_lr.fit(X, y)

                # Propensity Scores
                lower_limit_prob = list_flip_ratio[j]['lower_limit_prob']
                
                qt = QuantileTransformer(random_state=r)
                qt.fit(amount[y==1].reshape(-1,1))
                quantiles = qt.transform(amount.reshape(-1,1))
                
                proba = lower_limit_prob**(1-quantiles)
                proba = proba.reshape(-1,)
                
                prob_sarpg_aux = k * propensity_lr.predict_proba(X)[:,1]
                prob_sarpg_aux[prob_sarpg_aux<0.01]=0.01
                proba = (proba**0.25) * (prob_sarpg_aux**0.75)
                
                e = proba[ix_tr]
                e[e<0.01]=0.01

                e_tr = e[ix_xgb_tr]
                e_vl = e[ix_xgb_val]
                
                # Training Proba. Unlabeled to be Positive
                prob_unlpos_tr = ((1 - e_tr)/e_tr)*prob_xgb_tr/(1-prob_xgb_tr)
                prob_unlpos_tr[prob_unlpos_tr>1] = 1
                            
                # Validation Proba. Unlabeled to be Positive
                prob_unlpos_vl= ((1 - e_vl)/e_vl)*prob_xgb_vl/(1-prob_xgb_vl)
                prob_unlpos_vl[prob_unlpos_vl>1] = 1
                
                cost_matrix_xgb_pucsboost_sarcost_tr = np.zeros((len(X_tr[ix_xgb_tr]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_xgb_pucsboost_sarcost_tr[:, 0, 0] = amount[ix_tr][ix_xgb_tr]*prob_unlpos_tr
                cost_matrix_xgb_pucsboost_sarcost_tr[:, 0, 1] = amount[ix_tr][ix_xgb_tr]
                cost_matrix_xgb_pucsboost_sarcost_tr[:, 1, 0] = cf
                cost_matrix_xgb_pucsboost_sarcost_tr[:, 1, 1] = cf
                
                cost_matrix_xgb_pucsboost_sarcost_vl = np.zeros((len(X_tr[ix_xgb_val]), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
                cost_matrix_xgb_pucsboost_sarcost_vl[:, 0, 0] = amount[ix_tr][ix_xgb_val]*prob_unlpos_vl
                cost_matrix_xgb_pucsboost_sarcost_vl[:, 0, 1] = amount[ix_tr][ix_xgb_val]
                cost_matrix_xgb_pucsboost_sarcost_vl[:, 1, 0] = cf
                cost_matrix_xgb_pucsboost_sarcost_vl[:, 1, 1] = cf

# ----------------------- non-negative PU XGBoost SARCost -------------------- #

                m_xgb_nnpu_sarcost = CSBoost(obj='nnpu_sar', 
                                           validation=True,
                                           random_state=r).fit(X_tr[ix_xgb_tr], 
                                                               noisy_y[ix_tr][ix_xgb_tr], 
                                                               X_tr[ix_xgb_val], 
                                                               noisy_y[ix_tr][ix_xgb_val],
                                                               propensity_score_train=e_tr,
                                                               propensity_score_val=e_vl)
                
                prob_xgb_nnpu_sarcost = expit(m_xgb_nnpu_sarcost.inplace_predict(X_ts))
               
                
                # Non-threshold measures
                list_roc_xgb_nnpu_sarcost.append(roc_auc_score(y[ix_ts], prob_xgb_nnpu_sarcost))
                list_ap_xgb_nnpu_sarcost.append(average_precision_score(y[ix_ts], prob_xgb_nnpu_sarcost))
                print('AP nnPU SARCost: ', list_ap_xgb_nnpu_sarcost[-1])
                list_as_xgb_nnpu_sarcost.append(ave_savings_score(y[ix_ts], prob_xgb_nnpu_sarcost, amount[ix_ts], cf=cf))
                print('AS nnPU SARCost: ', list_as_xgb_nnpu_sarcost[-1])
                list_top1_precision_xgb_nnpu_sarcost.append(top_kperc_precision(y[ix_ts], prob_xgb_nnpu_sarcost))
                list_top1_savings_xgb_nnpu_sarcost.append(top_kperc_savings(y[ix_ts], prob_xgb_nnpu_sarcost, amount[ix_ts], cf=cf))
                
                # Threshold-dependent measures    
                prob_vl_xgb_nnpu_sarcost = m_xgb_nnpu_sarcost.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_nnpu_sarcost = [binarize_prob(prob_vl_xgb_nnpu_sarcost, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_nnpu_sarcost = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_nnpu_sarcost]
                ix_max_f1score = np.where(set_f1_scores_xgb_nnpu_sarcost==np.max(set_f1_scores_xgb_nnpu_sarcost))[0][0]
                labels_xgb_nnpu_sarcost = binarize_prob(prob_xgb_nnpu_sarcost, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_nnpu_sarcost.append(f1_score(y[ix_ts], labels_xgb_nnpu_sarcost))
                list_recall_xgb_nnpu_sarcost.append(recall_score(y[ix_ts], labels_xgb_nnpu_sarcost))
                list_precision_xgb_nnpu_sarcost.append(precision_score(y[ix_ts], labels_xgb_nnpu_sarcost))

# ------------------------- XGBoost PUCSBoost SARCost ------------------------ #
                
                m_xgb_pucsboost_sarcost = CSBoost(obj='aec', validation=True).fit(X_tr[ix_xgb_tr], 
                                                                        noisy_y[ix_tr][ix_xgb_tr], 
                                                                        X_tr[ix_xgb_val], 
                                                                        noisy_y[ix_tr][ix_xgb_val], 
                                                                        cost_matrix_train=cost_matrix_xgb_pucsboost_sarcost_tr, 
                                                                        cost_matrix_val=cost_matrix_xgb_pucsboost_sarcost_vl)
                
                prob_xgb_pucsboost_sarcost = expit(m_xgb_pucsboost_sarcost.inplace_predict(X_ts))
                
                # Non-threshold measures
                list_roc_xgb_pucsboost_sarcost.append(roc_auc_score(y[ix_ts], prob_xgb_pucsboost_sarcost))
                list_ap_xgb_pucsboost_sarcost.append(average_precision_score(y[ix_ts], prob_xgb_pucsboost_sarcost))
                print('AP PUCSBoost SARCost : ', list_ap_xgb_pucsboost_sarcost[-1])
                list_as_xgb_pucsboost_sarcost.append(ave_savings_score(y[ix_ts], prob_xgb_pucsboost_sarcost, amount[ix_ts], cf=cf))
                print('AS PUCSBoost SARCost : ', list_as_xgb_pucsboost_sarcost[-1])
                list_top1_precision_xgb_pucsboost_sarcost.append(top_kperc_precision(y[ix_ts], prob_xgb_pucsboost_sarcost))
                list_top1_savings_xgb_pucsboost_sarcost.append(top_kperc_savings(y[ix_ts], prob_xgb_pucsboost_sarcost, amount[ix_ts], cf=cf))
                
                # Threshold-dependent measures        
                prob_vl_xgb_pucsboost_sarcost = m_xgb_pucsboost_sarcost.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_pucsboost_sarcost = [binarize_prob(prob_vl_xgb_pucsboost_sarcost, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_pucsboost_sarcost = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_pucsboost_sarcost]
                ix_max_f1score = np.where(set_f1_scores_xgb_pucsboost_sarcost==np.max(set_f1_scores_xgb_pucsboost_sarcost))[0][0]
                labels_xgb_pucsboost_sarcost = binarize_prob(prob_xgb_pucsboost_sarcost, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_pucsboost_sarcost.append(f1_score(y[ix_ts], labels_xgb_pucsboost_sarcost))
                list_recall_xgb_pucsboost_sarcost.append(recall_score(y[ix_ts], labels_xgb_pucsboost_sarcost))
                list_precision_xgb_pucsboost_sarcost.append(precision_score(y[ix_ts], labels_xgb_pucsboost_sarcost))

# ---------------------------- CSPU XGBoost SARCOST -------------------------- #
        
                m_xgb_cspu_sarcost = CSBoost(obj='puwce_sar',
                                        random_state=r,
                                        validation=True).fit(X_tr[ix_xgb_tr], 
                                                            noisy_y[ix_tr][ix_xgb_tr], 
                                                            X_tr[ix_xgb_val], 
                                                            noisy_y[ix_tr][ix_xgb_val],
                                                            cost_matrix_train=cost_matrix_xgb_tr, 
                                                            cost_matrix_val=cost_matrix_xgb_vl,
                                                            propensity_score_train=e_tr,
                                                            propensity_score_val=e_vl)
                
                prob_xgb_cspu_sarcost = expit(m_xgb_cspu_sarcost.inplace_predict(X_ts))
            
                # Non-threshold measures
                list_roc_xgb_cspu_sarcost.append(roc_auc_score(y[ix_ts], prob_xgb_cspu_sarcost))
                list_ap_xgb_cspu_sarcost.append(average_precision_score(y[ix_ts], prob_xgb_cspu_sarcost))
                print('AP CSPU SARCost: ', list_ap_xgb_cspu_sarcost[-1])
                list_as_xgb_cspu_sarcost.append(ave_savings_score(y[ix_ts], prob_xgb_cspu_sarcost, amount[ix_ts], cf=cf))
                print('AS CSPU SARCost: ', list_as_xgb_cspu_sarcost[-1])
                list_top1_precision_xgb_cspu_sarcost.append(top_kperc_precision(y[ix_ts], prob_xgb_cspu_sarcost))
                list_top1_savings_xgb_cspu_sarcost.append(top_kperc_savings(y[ix_ts], prob_xgb_cspu_sarcost, amount[ix_ts], cf=cf))
        
                # Threshold-dependent measures        
                prob_vl_xgb_cspu_sarcost = m_xgb_cspu_sarcost.inplace_predict(X_tr[ix_xgb_val])
                set_labels_xgb_cspu_sarcost = [binarize_prob(prob_vl_xgb_cspu_sarcost, threshold=i) for i in set_thresholds]
                set_f1_scores_xgb_cspu_sarcost = [f1_score(y[ix_tr][ix_xgb_val], i) for i in set_labels_xgb_cspu_sarcost]
                ix_max_f1score = np.where(set_f1_scores_xgb_cspu_sarcost==np.max(set_f1_scores_xgb_cspu_sarcost))[0][0]
                labels_xgb_cspu_sarcost = binarize_prob(prob_xgb_cspu_sarcost, threshold=set_thresholds[ix_max_f1score])
                
                # Optimal F1-score
                list_f1_opt_xgb_cspu_sarcost.append(f1_score(y[ix_ts], labels_xgb_cspu_sarcost))
                list_recall_xgb_cspu_sarcost.append(recall_score(y[ix_ts], labels_xgb_cspu_sarcost))
                list_precision_xgb_cspu_sarcost.append(precision_score(y[ix_ts], labels_xgb_cspu_sarcost))

            else:           
                list_ap_xgb_nnpu_sarcost.append(np.NaN)
                list_as_xgb_nnpu_sarcost.append(np.NaN)
                list_roc_xgb_nnpu_sarcost.append(np.NaN)
                list_f1_opt_xgb_nnpu_sarcost.append(np.NaN)
                list_recall_xgb_nnpu_sarcost.append(np.NaN)
                list_precision_xgb_nnpu_sarcost.append(np.NaN)
                list_top1_precision_xgb_nnpu_sarcost.append(np.NaN)
                list_top1_savings_xgb_nnpu_sarcost.append(np.NaN)

                list_ap_xgb_pucsboost_sarcost.append(np.NaN)
                list_as_xgb_pucsboost_sarcost.append(np.NaN)
                list_roc_xgb_pucsboost_sarcost.append(np.NaN)
                list_f1_opt_xgb_pucsboost_sarcost.append(np.NaN)
                list_recall_xgb_pucsboost_sarcost.append(np.NaN)
                list_precision_xgb_pucsboost_sarcost.append(np.NaN)
                list_top1_precision_xgb_pucsboost_sarcost.append(np.NaN)
                list_top1_savings_xgb_pucsboost_sarcost.append(np.NaN)

                list_ap_xgb_cspu_sarcost.append(np.NaN)
                list_as_xgb_cspu_sarcost.append(np.NaN)
                list_roc_xgb_cspu_sarcost.append(np.NaN)
                list_f1_opt_xgb_cspu_sarcost.append(np.NaN)
                list_recall_xgb_cspu_sarcost.append(np.NaN)
                list_precision_xgb_cspu_sarcost.append(np.NaN)
                list_top1_precision_xgb_cspu_sarcost.append(np.NaN)
                list_top1_savings_xgb_cspu_sarcost.append(np.NaN)
    
# =============================================================================
#           EXPORTING RESULTS TO CSV
# =============================================================================

df_ref = pd.DataFrame()

list_flip_ratio_ = [list_flip_ratio[i]['flip_ratio'] for i in list_flip_ratio.keys()]
list_scaling_factor = [list_flip_ratio[i]['scaling_factor'] for i in list_flip_ratio.keys()]
list_lower_limit_prob = [list_flip_ratio[i]['lower_limit_prob'] for i in list_flip_ratio.keys()]

labelnoise_col = list(np.repeat(list_label_noise, len(list_random_state)*len(list_flip_ratio) ))
flipratio_col = list(np.repeat(list_flip_ratio_, len(list_random_state)))*len(list_label_noise)
scalingfactor_col = list(np.repeat(list_scaling_factor, len(list_random_state)))*len(list_label_noise)
lowerlimitprob_col = list(np.repeat(list_lower_limit_prob, len(list_random_state)))*len(list_label_noise)

randomstate_col = list_random_state*len(list_label_noise)*len(list_flip_ratio)
   
df_ref['label_noise'] = labelnoise_col
df_ref['flip_ratio'] = flipratio_col
df_ref['scaling_factor'] = scalingfactor_col
df_ref['lower_limit_prob'] = lowerlimitprob_col
df_ref['random_state'] = randomstate_col

df_ref['ap_xgb_ce'] = list_ap_xgb_ce
df_ref['as_xgb_ce'] = list_as_xgb_ce
df_ref['roc_xgb_ce'] = list_roc_xgb_ce
df_ref['f1_opt_xgb_ce'] = list_f1_opt_xgb_ce
df_ref['rec_xgb_ce'] = list_recall_xgb_ce
df_ref['pre_xgb_ce'] = list_precision_xgb_ce
df_ref['top1perc_pre_xgb_ce'] = list_top1_precision_xgb_ce
df_ref['top1perc_sav_xgb_ce'] = list_top1_savings_xgb_ce

df_ref['ap_xgb_csboost'] = list_ap_xgb_csboost
df_ref['as_xgb_csboost'] = list_as_xgb_csboost
df_ref['roc_xgb_csboost'] = list_roc_xgb_csboost
df_ref['f1_opt_xgb_csboost'] = list_f1_opt_xgb_csboost
df_ref['rec_xgb_csboost'] = list_recall_xgb_csboost
df_ref['pre_xgb_csboost'] = list_precision_xgb_csboost
df_ref['top1perc_pre_xgb_csboost'] = list_top1_precision_xgb_csboost
df_ref['top1perc_sav_xgb_csboost'] = list_top1_savings_xgb_csboost

df_ref['ap_xgb_pucsboost_scar_ek'] = list_ap_xgb_pucsboost_scar_ek
df_ref['as_xgb_pucsboost_scar_ek'] = list_as_xgb_pucsboost_scar_ek
df_ref['roc_xgb_pucsboost_scar_ek'] = list_roc_xgb_pucsboost_scar_ek
df_ref['f1_opt_xgb_pucsboost_scar_ek'] = list_f1_opt_xgb_pucsboost_scar_ek
df_ref['rec_xgb_pucsboost_scar_ek'] = list_recall_xgb_pucsboost_scar_ek
df_ref['pre_xgb_pucsboost_scar_ek'] = list_precision_xgb_pucsboost_scar_ek
df_ref['top1perc_pre_xgb_pucsboost_scar_ek'] = list_top1_precision_xgb_pucsboost_scar_ek
df_ref['top1perc_sav_xgb_pucsboost_scar_ek'] = list_top1_savings_xgb_pucsboost_scar_ek

df_ref['ap_xgb_pucsboost_sarpg'] = list_ap_xgb_pucsboost_sarpg
df_ref['as_xgb_pucsboost_sarpg'] = list_as_xgb_pucsboost_sarpg
df_ref['roc_xgb_pucsboost_sarpg'] = list_roc_xgb_pucsboost_sarpg
df_ref['f1_opt_xgb_pucsboost_sarpg'] = list_f1_opt_xgb_pucsboost_sarpg
df_ref['rec_xgb_pucsboost_sarpg'] = list_recall_xgb_pucsboost_sarpg
df_ref['pre_xgb_pucsboost_sarpg'] = list_precision_xgb_pucsboost_sarpg
df_ref['top1perc_pre_xgb_pucsboost_sarpg'] = list_top1_precision_xgb_pucsboost_sarpg
df_ref['top1perc_sav_xgb_pucsboost_sarpg'] = list_top1_savings_xgb_pucsboost_sarpg

df_ref['ap_xgb_pucsboost_sarcost'] = list_ap_xgb_pucsboost_sarcost
df_ref['as_xgb_pucsboost_sarcost'] = list_as_xgb_pucsboost_sarcost
df_ref['roc_xgb_pucsboost_sarcost'] = list_roc_xgb_pucsboost_sarcost
df_ref['f1_opt_xgb_pucsboost_sarcost'] = list_f1_opt_xgb_pucsboost_sarcost
df_ref['rec_xgb_pucsboost_sarcost'] = list_recall_xgb_pucsboost_sarcost
df_ref['pre_xgb_pucsboost_sarcost'] = list_precision_xgb_pucsboost_sarcost
df_ref['top1perc_pre_xgb_pucsboost_sarcost'] = list_top1_precision_xgb_pucsboost_sarcost
df_ref['top1perc_sav_xgb_pucsboost_sarcost'] = list_top1_savings_xgb_pucsboost_sarcost

df_ref['ap_xgb_nnpu_scar'] = list_ap_xgb_nnpu_scar
df_ref['as_xgb_nnpu_scar'] = list_as_xgb_nnpu_scar
df_ref['roc_xgb_nnpu_scar'] = list_roc_xgb_nnpu_scar
df_ref['f1_opt_xgb_nnpu_scar'] = list_f1_opt_xgb_nnpu_scar
df_ref['rec_xgb_nnpu_scar'] = list_recall_xgb_nnpu_scar
df_ref['pre_xgb_nnpu_scar'] = list_precision_xgb_nnpu_scar
df_ref['top1perc_pre_xgb_nnpu_scar'] = list_top1_precision_xgb_nnpu_scar
df_ref['top1perc_sav_xgb_nnpu_scar'] = list_top1_savings_xgb_nnpu_scar

df_ref['ap_xgb_nnpu_sarpg'] = list_ap_xgb_nnpu_sarpg
df_ref['as_xgb_nnpu_sarpg'] = list_as_xgb_nnpu_sarpg
df_ref['roc_xgb_nnpu_sarpg'] = list_roc_xgb_nnpu_sarpg
df_ref['f1_opt_xgb_nnpu_sarpg'] = list_f1_opt_xgb_nnpu_sarpg
df_ref['rec_xgb_nnpu_sarpg'] = list_recall_xgb_nnpu_sarpg
df_ref['pre_xgb_nnpu_sarpg'] = list_precision_xgb_nnpu_sarpg
df_ref['top1perc_pre_xgb_nnpu_sarpg'] = list_top1_precision_xgb_nnpu_sarpg
df_ref['top1perc_sav_xgb_nnpu_sarpg'] = list_top1_savings_xgb_nnpu_sarpg

df_ref['ap_xgb_nnpu_sarcost'] = list_ap_xgb_nnpu_sarcost
df_ref['as_xgb_nnpu_sarcost'] = list_as_xgb_nnpu_sarcost
df_ref['roc_xgb_nnpu_sarcost'] = list_roc_xgb_nnpu_sarcost
df_ref['f1_opt_xgb_nnpu_sarcost'] = list_f1_opt_xgb_nnpu_sarcost
df_ref['rec_xgb_nnpu_sarcost'] = list_recall_xgb_nnpu_sarcost
df_ref['pre_xgb_nnpu_sarcost'] = list_precision_xgb_nnpu_sarcost
df_ref['top1perc_pre_xgb_nnpu_sarcost'] = list_top1_precision_xgb_nnpu_sarcost
df_ref['top1perc_sav_xgb_nnpu_sarcost'] = list_top1_savings_xgb_nnpu_sarcost

df_ref['ap_xgb_cspu_scar'] = list_ap_xgb_cspu_scar
df_ref['as_xgb_cspu_scar'] = list_as_xgb_cspu_scar
df_ref['roc_xgb_cspu_scar'] = list_roc_xgb_cspu_scar
df_ref['f1_opt_xgb_cspu_scar'] = list_f1_opt_xgb_cspu_scar
df_ref['rec_xgb_cspu_scar'] = list_recall_xgb_cspu_scar
df_ref['pre_xgb_cspu_scar'] = list_precision_xgb_cspu_scar
df_ref['top1perc_pre_xgb_cspu_scar'] = list_top1_precision_xgb_cspu_scar
df_ref['top1perc_sav_xgb_cspu_scar'] = list_top1_savings_xgb_cspu_scar

df_ref['ap_xgb_cspu_sarpg'] = list_ap_xgb_cspu_sarpg
df_ref['as_xgb_cspu_sarpg'] = list_as_xgb_cspu_sarpg
df_ref['roc_xgb_cspu_sarpg'] = list_roc_xgb_cspu_sarpg
df_ref['f1_opt_xgb_cspu_sarpg'] = list_f1_opt_xgb_cspu_sarpg
df_ref['rec_xgb_cspu_sarpg'] = list_recall_xgb_cspu_sarpg
df_ref['pre_xgb_cspu_sarpg'] = list_precision_xgb_cspu_sarpg
df_ref['top1perc_pre_xgb_cspu_sarpg'] = list_top1_precision_xgb_cspu_sarpg
df_ref['top1perc_sav_xgb_cspu_sarpg'] = list_top1_savings_xgb_cspu_sarpg

df_ref['ap_xgb_cspu_sarcost'] = list_ap_xgb_cspu_sarcost
df_ref['as_xgb_cspu_sarcost'] = list_as_xgb_cspu_sarcost
df_ref['roc_xgb_cspu_sarcost'] = list_roc_xgb_cspu_sarcost
df_ref['f1_opt_xgb_cspu_sarcost'] = list_f1_opt_xgb_cspu_sarcost
df_ref['rec_xgb_cspu_sarcost'] = list_recall_xgb_cspu_sarcost
df_ref['pre_xgb_cspu_sarcost'] = list_precision_xgb_cspu_sarcost
df_ref['top1perc_pre_xgb_cspu_sarcost'] = list_top1_precision_xgb_cspu_sarcost
df_ref['top1perc_sav_xgb_cspu_sarcost'] = list_top1_savings_xgb_cspu_sarcost

df_ref.to_csv('exp_setup_ibm.csv', index=False, na_rep='NA')

