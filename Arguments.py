#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:46:45 2019


List of argumnents usable with parser 


@author: nicholas
"""

import argparse


parser = argparse.ArgumentParser(description='Train an AutoEncoder Recommender')



parser.add_argument('--id', type=str, metavar='', required=True, help='ID of experience. Will be used when saving file.')


# Data
parser.add_argument('--dataPATH', type=str, metavar='', default='./Data/', \
                    help='Path to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='ReDialRnGChronoTRAIN.json', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='ReDialRnGChronoVALID.json', \
                    help='File name of Dataset to for validation')
parser.add_argument('--incl_genres', type=bool, metavar='', default=True, \
                    help='If False, no use genres (Dataset part empty for genres)')
parser.add_argument('--merge_data', type=bool, metavar='', default=True, \
                    help='If True, mentionned and to be mentionned data are added. Used in Dataset \
                    ALWAYS use False when for PredChrono.')


# Training
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=1000, help='Number of epoch')
parser.add_argument('--criterion', type=str, metavar='', default='BCEWLL', \
                    choices=['BCEWLL', 'BCE'], help='Loss function')
parser.add_argument('--noiseTrain', type=bool, metavar='', default=False, \
                    help='When True, mimics ReDial inputs by allowing only from 1 to 7 (random)\
                    ratings as inputs.')
parser.add_argument('--noiseEval', type=bool, metavar='', default=False, \
                    help='When True, mimics ReDial inputs by allowing only from 1 to 7 (random)\
                    ratings as inputs.')
parser.add_argument('--weights', type=float, metavar='', default=1, \
                    help='Weights multiplying the errors on ratings of 0 (underrepresented) \
                    during training.  1 -> no weights')

parser.add_argument('--patience', type=int, metavar='', default=1, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--completionTrain', type=float, metavar='', default=100, \
                    help='% of data used during 1 training epoch ~ "early stopping"')
parser.add_argument('--completionPred', type=float, metavar='', default=100, \
                    help='% of data used for prediction')
parser.add_argument('--completionPredChrono', type=float, metavar='', default=100, \
                    help='% of data used for chrono prediction')
parser.add_argument('--EARLY', type=bool, metavar='', default=False, \
                    help="Train at 10%, Pred at 1% and PredChrono at 1%")



# Model
parser.add_argument('--g_type', type=str, metavar='', default='one', \
                    choices=['none', 'one', 'genres', 'unit'], \
                    help="Parameter(s) learned for genres inputs. None means no learning, One is \
                    for one global parameter, Unit is one parameter per unit,...")
parser.add_argument('--layer1', type=int, metavar='', default=344, \
                    help='Integers corresponding to the first hidden layer size')
parser.add_argument('--layer2', type=int, metavar='', default=27, \
                    help='Integers corresponding to the second hidden layer size. 0 if none.')
parser.add_argument('--activations', type=str, metavar='', default='sigmoid', \
                    choices=['relu', 'sigmoid'],\
                    help='Activations in hidden layers of the model')
parser.add_argument('--last_layer_activation', type=str, metavar='', default='none', \
                    choices=['none', 'sigmoid', 'softmax'],\
                    help='Last layer activation of the model')
parser.add_argument('--preModel', type=str, metavar='', default='none', \
                    help='Path to a pre-trained model to start with. Should \
                    include a GenresWrapper of same type')


# Genres 
parser.add_argument('--genresDict', type=str, metavar='', default='dict_genresInter_idx_UiD.json', \
                    help='File name of Dict of genres')
parser.add_argument('--top_cut', type=int, metavar='', default=100, \
                    help='number of movies in genres vector (for torch Dataset)')


# Metrics
parser.add_argument('--topx', type=int, metavar='', default=100, \
                    help='for NDCG mesure, size of top ranks considered')


# Others
parser.add_argument('--seed', type=bool, metavar='', default=False, \
                    help="If True, random always give the same")

parser.add_argument('--orion', type=bool, metavar='', default=False, \
                    help="Run Orion - Hyper Parameter search")

parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")

parser.add_argument('--DEBUG', type=bool, metavar='', default=False, \
                    help="Reduced dataset and epoch to 1 for rapid debug purposes")



args = parser.parse_args()






# ASSERTIONS

if args.criterion == 'BCE':
    assert args.last_layer_activation != 'none','Need last layer activation with BCE'
if args.criterion == 'BCEWLL':
    assert args.last_layer_activation == 'none',"Last layer activation must be 'none' with BCEWLL"

assert 0 <= args.completionTrain <=100,'completionTrain should be in [0,100]'
assert 0 <= args.completionPred <=100,'completionPred should be in [0,100]'
assert 0 <= args.completionPredChrono <=100,'completionPredChrono should be in [0,100]'





# CONVERSION
# (group of hyper-parameters group under a name for efficiency when running)

if args.EARLY:
    args.completionTrain = 10 
    args.completionPred = 1
    args.completionPredChrono = 1
    
    