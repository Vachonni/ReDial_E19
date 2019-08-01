#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:49:29 2019


Hyper-parameter search with Orion for Train_RnGChrono and Pred_RnGChrono
 

@author: nicholas
"""


########  IMPORTS  ########


import os
import json
import sys
# Get all arguemnts
from Arguments import args 
import Train_RnGChrono_ORION
import Pred_RnGChrono

# Get Orion
from orion.client import report_results




########  ARGUMENTS  ########


# Making the --id a proper folder (need for Orion, adapted elsewhere)
args.id += '/'      

# Managing the lack og 'choice' in ORION
if args.ORION_NOpreTrain == 1: args.NOpreTrain = True
if args.ORION_g_type == 0: args.g_type = 'one'
if args.ORION_g_type == 1: args.g_type = 'unit'
if args.ORION_g_type == 2: args.g_type = 'genres'
if args.ORION_zero == 1: args.zero11 = True
if args.ORION_zero == 2: args.zero12 = True



print(vars(args))




# Save all arguments values
if not os.path.isdir(args.id): os.mkdir(args.id)
with open(args.id+'arguments.json', 'w') as fp:
    json.dump(vars(args), fp, sort_keys=True, indent=4)
    fp.write('\n\n'+str(sys.argv))



########  TRAINING  ########


# NO Pretraing on ML    
if args.NOpreTrain: 
    args.dataTrain = 'ReDialRnGChronoTRAIN.json' 
    args.dataValid  = 'ReDialRnGChronoVALID.json' 
    args.noiseTrain = False
    args.noiseEval = False
    args.completionTrain = 100 
    args.completionPred = 0
    args.completionPredEpoch = 0 
    args.activations = 'relu'
    args.last_layer_activation = 'softmax'
    args.loss_fct = 'BCE'
    args.seed = True 
    
    # Execute training on ReDial
    Train_RnGChrono_ORION.main(args) 
    
    
# Pretraing on ML    
else:    
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

    # delete the ML_model (for space considerations)
    os.remove(args.id + 'ML_model.pth')




########  PREDICTION  ########
    

# Set args for prediction of one model, 
args.seed = True
args.M1_path = args.id + 'Re_model.pth'   
args.completionPredChrono = 100

# Execute prediction on the ReDial model
NDCGs_1model = Pred_RnGChrono.main(args) 
assert NDCGs_1model != -1, "Orion's objective not evaluated"






########  ORION  ########


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

