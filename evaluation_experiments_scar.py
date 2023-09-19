import scipy.stats as stats
import pandas as pd
import numpy as np

### Functions ####
def imandavenport_test(num_models, num_datasets, model_ranks):

  chisqF = (12*num_datasets/(num_models*(num_models+1)))*(sum(model_ranks**2)-(num_models*(num_models+1)**2)/4)
  Ff = (num_datasets-1)*chisqF/(num_datasets*(num_models-1)-chisqF)

  df1 = num_models - 1
  df2 = (num_models-1)*(num_datasets-1)
  pvalue = 1 - stats.f.cdf(Ff, df1, df2)
  print('p-value ROC ranks: ', pvalue)
  return pvalue
    
def bonfholm_test(num_models, num_datasets, model_names, model_ranks, alpha, control = None):
  denominator_ = (num_models*(num_models+1)/(6*num_datasets))**0.5
  list_pv_ = np.array(model_ranks)
  if control == None:
    ix_min_ = np.where(list_pv_==np.min(model_ranks))[0]
  else:
    ix_min_ = np.where(model_names == control)[0]

  list_pv = np.delete(list_pv_, ix_min_)
  model_names_ = np.delete(model_names, ix_min_)

  z_scores = np.asarray([ (list_pv[i] - model_ranks[ix_min_] )/denominator_ for i, _ in enumerate(list_pv) ])
  p_values = stats.norm.sf(abs(z_scores)) #one-sided
  p_values = p_values.reshape(1,-1).flatten()
  ix_sort = np.argsort(p_values)
  decision = np.ones(num_models - 1, dtype=bool)

  for m, i in enumerate(p_values[ix_sort]):  
    if i <= alpha/(num_models-1-m):
      decision[ix_sort[m]] = False
      
  
  return model_names[ix_min_], model_names_[decision], p_values

TOP_DATASETS = 5
FLIP_RATIO = '25'
## Load Data

df = pd.read_csv('expsetup_mainresults_scar.csv')

cols = ['random_state','flip_ratio']

cond = 'flip_ratio == 0.'+FLIP_RATIO

df_ = df.copy()
df_ = df_.query(cond)
df_filtered = df_.drop(columns = cols)

datasets_ = ['belgian_bank','kbc','ulb','ieee','banksim','ibm']
ap_models_ = ['ap_xgb_ce','ap_xgb_csboost','ap_xgb_pucsboost_scar_ek',
              'ap_xgb_nnpu_scar','ap_xgb_cspu_scar']
              
as_models_ = ['as_xgb_ce','as_xgb_csboost','as_xgb_pucsboost_scar_ek',
              'as_xgb_nnpu_scar','as_xgb_cspu_scar']

top1pre_models_ = ['top1perc_pre_xgb_ce',
                   'top1perc_pre_xgb_csboost',
                   'top1perc_pre_xgb_pucsboost_scar_ek',
                   'top1perc_pre_xgb_nnpu_scar',
                   'top1perc_pre_xgb_cspu_scar']

top1sav_models_ = ['top1perc_sav_xgb_ce',
                   'top1perc_sav_xgb_csboost',
                   'top1perc_sav_xgb_pucsboost_scar_ek',
                   'top1perc_sav_xgb_nnpu_scar',
                   'top1perc_sav_xgb_cspu_scar']



ap_models_perdataset_ = ['ap_xgb_pucsboost_scar_ek','ap_xgb_cspu_scar','ap_xgb_csboost',
              'ap_xgb_nnpu_scar','ap_xgb_ce']
df_filtered.copy().groupby('dataset').mean().loc[datasets_, ap_models_perdataset_].transpose()

as_models_perdataset_ = ['as_xgb_pucsboost_scar_ek','as_xgb_cspu_scar','as_xgb_csboost',
              'as_xgb_nnpu_scar','as_xgb_ce']
df_filtered.copy().groupby('dataset').mean().loc[datasets_, as_models_perdataset_].transpose()


# Average Metrics

# ---------------------------------- PR AUC ---------------------------------- #

# Average 
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[ap_models_].to_string() )
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[as_models_].to_string() )
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[top1pre_models_].to_string() )
print(df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[top1sav_models_].to_string() )

# Standard Deviation
print(df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[ap_models_].to_string() )
print(df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[as_models_].to_string() )
print(df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[top1pre_models_].to_string() )
print(df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[top1sav_models_].to_string() )

# Average Rank
print(df_[ap_models_].copy().rank(axis = 1, ascending = False).mean(axis=0).to_string() )
print(df_[as_models_].copy().rank(axis = 1, ascending = False).mean(axis=0).to_string() )

# EXPORT EXCEL FILES

df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[ap_models_].to_csv('mean'+FLIP_RATIO+'_ap_results.csv', header=None)
df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[as_models_].to_csv('mean'+FLIP_RATIO+'_as_results.csv', header=None)
df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[top1pre_models_].to_csv('mean'+FLIP_RATIO+'_top1pre_results.csv', header=None)
df_filtered.copy().groupby('dataset').mean().mean(axis = 0).loc[top1sav_models_].to_csv('mean'+FLIP_RATIO+'_top1sav_results.csv', header=None)

df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[ap_models_].to_csv('sd'+FLIP_RATIO+'_ap_results.csv', header=None)
df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[as_models_].to_csv('sd'+FLIP_RATIO+'_as_results.csv', header=None)
df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[top1pre_models_].to_csv('sd'+FLIP_RATIO+'_top1pre_results.csv', header=None)
df_filtered.copy().groupby('dataset').std().mean(axis = 0).loc[top1sav_models_].to_csv('sd'+FLIP_RATIO+'_top1sav_results.csv', header=None)


#### TECHNIQUES ####
roc_models = ['roc_srf_puhd','roc_srf_hd', 'roc_pubag','roc_elkno','roc_welog',
             'roc_spyem','roc_rnkpr','roc_rf',
             'roc_rf_puhd','roc_dt_puhd','roc_dt_hd','roc_rf_hd',
             'roc_upu','roc_nnpu','roc_imbnnpu']

ap_models = ['ap_srf_puhd','ap_srf_hd', 'ap_pubag','ap_elkno','ap_welog',
             'ap_spyem','ap_rnkpr','ap_rf',
             'ap_rf_puhd','ap_dt_puhd','ap_dt_hd','ap_rf_hd',
             'ap_upu','ap_nnpu','ap_imbnnpu']

as_models = ['as_srf_puhd','as_srf_hd', 'as_pubag','as_elkno','as_welog',
             'as_spyem','as_rnkpr','as_rf',
             'as_rf_puhd','as_dt_puhd','as_dt_hd','as_rf_hd',
             'as_upu','as_nnpu','as_imbnnpu']

## Evaluation ##

# Optimal F1-score
df_f1scoremax_ = df_[as_models].copy().rank(axis = 1, ascending = False)
df_f1scoremax_['dataset'] = df_['dataset']

df_as_ranks = df_f1scoremax_.groupby('dataset').mean()
df_as_transpose = df_f1scoremax_.groupby('dataset').mean().T
df_f1scoremax_rank_mean_ = df_as_transpose.mean(axis = 1)

# PR-AUC
df_ap_ = df_[ap_models].copy().rank(axis = 1, ascending = False)
df_ap_['dataset'] = df_['dataset']

df_ap_ranks = df_ap_.groupby('dataset').mean()
df_ap_transpose = df_ap_.groupby('dataset').mean().T
df_ap_rank_mean_ = df_ap_transpose.mean(axis = 1)

# ROC-AUC 
df_roc_ = df_[roc_models].copy().rank(axis = 1, ascending = False)
df_roc_['dataset'] = df_['dataset']

df_roc_ranks = df_roc_.groupby('dataset').mean()
df_roc_transpose = df_roc_.groupby('dataset').mean().T
df_roc_rank_mean_ = df_roc_transpose.mean(axis = 1)

#### HYPOTHESIS TESTING ####
N = len(np.unique(df_f1scoremax_['dataset'])) * 20 

## Optimal F1-Score
k = len(df_f1scoremax_rank_mean_)

model_names_ = np.asarray(list(df_f1scoremax_rank_mean_.index))

imandavenport_test(k, N, df_f1scoremax_rank_mean_)

# Holm's Test F1 Score Max
lowest_rank_f1scoremax, no_rejected_f1scoremax, adjpvalues_f1scoremax = bonfholm_test(k, N, model_names_, df_f1scoremax_rank_mean_, 0.05, control='as_srf_puhd')

## PR-AUC
k = len(df_ap_rank_mean_)

model_names_ = np.asarray(list(df_ap_rank_mean_.index))

imandavenport_test(k, N, df_ap_rank_mean_)

# Holm's Test PR
lowest_rank_pr, no_rejected_pr, adjpvalues_pr = bonfholm_test(k, N, model_names_, df_ap_rank_mean_, 0.05, control='ap_rf_hd')

## ROC-AUC
k = len(df_roc_rank_mean_)

model_names_ = np.asarray(list(df_roc_rank_mean_.index))

imandavenport_test(k, N, df_roc_rank_mean_)

# Holm's Test PR
lowest_rankroc_, no_rejectedroc_, adjpvaluesroc_ = bonfholm_test(k, N, model_names_, df_roc_rank_mean_, 0.05, control='roc_pubag')

###############################################################################

df = pd.read_csv('experimental_results_2v.csv')

cols = ['data_partition']

cond = 'flip_ratio != "0.50"'

df_ = df.copy()
df_ = df_.query(cond)
df_filtered = df_.drop(columns = cols)
df_filtered_diff = df_filtered.loc[df['flip_ratio'] == 0.25].groupby('dataset').mean() - df_filtered.loc[df['flip_ratio'] == 0.75].groupby('dataset').mean()

# ROC AUC 
roc_models = ['roc_srf_puhd','roc_srf_hd', 'roc_pubag','roc_elkno','roc_welog',
             'roc_spyem','roc_rnkpr','roc_rf',
             'roc_rf_puhd','roc_dt_puhd','roc_dt_hd','roc_rf_hd',
             'roc_upu','roc_nnpu','roc_imbnnpu']

ix_roc_drop = df_filtered_diff[roc_models].mean(axis = 1).argsort()[::-1]
df_filtered_diff[roc_models].mean(axis = 1)[ix_roc_drop][:TOP_DATASETS]
['pizzacutter1', 'chile', 'speech', 'fraud_ieee', 'fraud_creditcard']

# AP AUC
ap_models = ['ap_srf_puhd','ap_srf_hd', 'ap_pubag','ap_elkno','ap_welog',
             'ap_spyem','ap_rnkpr','ap_rf',
             'ap_rf_puhd','ap_dt_puhd','ap_dt_hd','ap_rf_hd',
             'ap_upu','ap_nnpu','ap_imbnnpu']

ix_ap_drop = df_filtered_diff[ap_models].mean(axis = 1).argsort()[::-1]
df_filtered_diff[ap_models].mean(axis = 1)[ix_ap_drop][:TOP_DATASETS]
['chile', 'mnist', 'satellite', 'korean5', 'cover']

# F1 score
as_models = ['as_srf_puhd','as_srf_hd', 'as_pubag','as_elkno','as_welog',
             'as_spyem','as_rnkpr','as_rf',
             'as_rf_puhd','as_dt_puhd','as_dt_hd','as_rf_hd',
             'as_upu','as_nnpu','as_imbnnpu']

ix_as_drop = df_filtered_diff[as_models].mean(axis = 1).argsort()[::-1]
df_filtered_diff[as_models].mean(axis = 1)[ix_as_drop][:TOP_DATASETS]
['kddcup-land_vs_portsweep', 'cover', 'korean5', 'cargood', 'chile']

#################################################################################