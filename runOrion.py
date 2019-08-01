#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:49:29 2019


Hyper-parameter search with Orion for Train_RnGChrono and Pred_RnGChrono
 

@author: nicholas
"""

import os
import json
import sys
# Get all arguemnts
from Arguments import args 
import Train_RnGChrono_ORION
import Pred_RnGChrono

# Get Orion
from orion.client import report_results


# Making the --id a proper folder (need for Orion, adapted elsewhere)
args.id += '/'      


# Save all arguments values
if not os.path.isdir(args.id): os.mkdir(args.id)
with open(args.id+'arguments.json', 'w') as fp:
    json.dump(vars(args), fp, sort_keys=True, indent=4)
    fp.write('\n\n'+str(sys.argv))



# Set args for pre-training on ML
args.dataTrain = 'MLRnGChronoTRAIN.json' 
args.dataValid  = 'MLRnGChronoVALID.json' 
args.noiseTrain = True
args.noiseEval = True
args.completionTrain = 10 
args.completionPred = 0
args.completionPredEpoch = 0 
args.activations = 'relu'
args.last_layer_activation = 'softmax'
args.loss_fct = 'BCE'
args.seed = True

# Execute the pre-trainig on ML 
Train_RnGChrono_ORION.main(args) 




# Set args for training on ReDial after pre-training
args.preModel = args.id + 'ML_model.pth'
args.dataTrain = 'ReDialRnGChronoTRAIN.json' 
args.dataValid  = 'ReDialRnGChronoVALID.json' 
args.noiseTrain = False
args.noiseEval = False
args.completionTrain = 100 
args.completionPred = 0 
args.completionPredEpoch = 0 
args.seed = True
args.no_data_merge = True

# Execute training on ReDial
Train_RnGChrono_ORION.main(args) 




# Set args for prediction of one model, 
args.seed = True
args.M1_path = args.id + 'Re_model.pth'   
args.completionPredChrono = 100

# Execute prediction on the ReDial model
NDCGs_1model = Pred_RnGChrono.main(args) 
assert NDCGs_1model != -1, "Orion's objective not evaluated"




# For Orion, print results (MongoDB,...)

report_results([dict(
    name='NDCG with genres',
    type='objective',
    value=-NDCGs_1model[1]),
    dict(
    name='NDCG without genres',
    type='constraint',
    value=-NDCGs_1model[0]),
#    dict(
#    name='valid_pred_error',
#    type='constraint',
#    value=pred_err),
#    dict(
#    name='valid_reconst_error',
#    type='constraint',
#    value=valid_err),
#    dict(
#    name='g',
#    type='constraint',
#    value=model.g.data.item())
    ])
