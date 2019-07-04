#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:02:56 2019


Training AE Recommender 


@author: nicholas
"""


########  IMPORTS  ########


import sys
import json
import torch
from torch import optim
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

# Personnal imports
import AutoEncoders 
import Utils
import Settings 
from Arguments import args 
# To use Orion
if args.orion:
    from orion.client import report_results




########  INIT  ########


# Print agrs that will be used
print(sys.argv)

# Add 1 to nb_movies_in_total because index of movies starts at 1
nb_movies = Settings.nb_movies_in_total + 1

# Cuda availability check
if args.DEVICE == "cuda" and not torch.cuda.is_available():
    raise ValueError("DEVICE specify a GPU computation but CUDA is not available")
  
# Seed 
if args.seed:
    manualSeed = 1
    # Python
  #  random.seed(manualSeed)
    # Numpy
    np.random.seed(manualSeed)
    # Torch
    torch.manual_seed(manualSeed)
    # Torch with GPU
    if args.DEVICE == "cuda":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True





########  MODEL  ########
    
    
# Organize args layers 
if args.layer2 == 0:
    layers = [nb_movies, args.layer1]
else: 
    layers = [nb_movies, args.layer1, args.layer2]


# Create a new model    
if args.preModel == 'none':     
    print('******* Creating NEW Model *******')      
    # Create basic model
    model_base = AutoEncoders.AsymmetricAutoEncoder(layers, nl_type=args.activations, \
                                                    is_constrained=False, dp_drop_prob=0.0, \
                                                    last_layer_activations=False,\
                                                    lla = args.last_layer_activation).to(args.DEVICE)
    model = AutoEncoders.GenresWrapperChrono(model_base, args.g_type).to(args.DEVICE)
# Load an existing model
else:
    print('******* Load EXISTING Model *******')      
    model_base = AutoEncoders.AsymmetricAutoEncoder(layers, nl_type=args.activations, \
                                                    is_constrained=False, dp_drop_prob=0.0, \
                                                    last_layer_activations=False, \
                                                    lla = args.last_layer_activation).to(args.DEVICE)
    model = AutoEncoders.GenresWrapperChrono(model_base, args.g_type).to(args.DEVICE)
    checkpoint = torch.load(args.preModel, map_location=args.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])


optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.0)


if args.criterion == 'BCEWLL':
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
elif args.criterion == 'BCE':
    criterion = torch.nn.BCELoss(reduction='none')





######## DATA ########

######## LOAD DATA 
# R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
print('******* Loading SAMPLES from *******', args.dataPATH + args.dataTrain)
train_data = json.load(open(args.dataPATH + args.dataTrain))
valid_data = json.load(open(args.dataPATH + args.dataValid))
if args.DEBUG: 
    train_data = train_data[:128]
    valid_data = valid_data[:128]

# G (genres) - Format [ [UserID, [movies uID of genres mentionned]] ]    
print('******* Loading GENRES from *******', args.genresDict)
dict_genresInter_idx_UiD = json.load(open(args.dataPATH + args.genresDict))

# Getting the popularity vector 
print('** Including popularity')
popularity = np.load(args.dataPATH + 'popularity_vector.npy')
popularity = torch.from_numpy(popularity).float()

######## CREATING DATASET ListRatingDataset 
print('******* Creating torch datasets *******')
train_dataset = Utils.RnGChronoDataset(train_data, dict_genresInter_idx_UiD, \
                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                       args.no_data_merge, args.noiseTrain, args.top_cut)
valid_dataset = Utils.RnGChronoDataset(valid_data, dict_genresInter_idx_UiD, \
                                       nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                       args.no_data_merge, args.noiseEval, args.top_cut)

######## CREATE DATALOADER
print('******* Creating dataloaders *******\n\n')    
kwargs = {}
if(args.DEVICE == "cuda"):
    kwargs = {'num_workers': 0, 'pin_memory': False}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)    






######## RUN TRAINING AND VALIDATION  ########

train_losses = []
valid_losses = []

global_pred_err_epoch = []
ReDial_pred_err_epoch = []
avrg_ranks_epoch = []



pred_mean_values = []

#old_g = torch.zeros(1).to(args.DEVICE)




if args.DEBUG: args.epoch = 1
for epoch in range(args.epoch):
    
    print('\n\n\n\n     ==> Epoch:', epoch, '\n')
    
    train_loss = Utils.TrainReconstruction(train_loader, model, criterion, optimizer, \
                                           args.weights, args.completionTrain)
    eval_loss = Utils.EvalReconstruction(valid_loader, model, criterion, \
                                         args.completionTrain)
    
    
    """ """
    pred_mean_values += train_loss[0]
    train_loss = train_loss[1]
    
    plt.plot(pred_mean_values)
    plt.title('Avrg Pred value by batch')
    plt.xlabel('batch')
    plt.ylabel('avrg pred value')
    plt.show()
    """ """
    
    
    
    
    train_losses.append(train_loss)
    valid_losses.append(eval_loss)
    losses = [train_losses, valid_losses]  
    
    print('\nEND EPOCH {:3d} \nTrain Reconstruction loss on targets: {:.4f}\
          \nValid Reconstruction Loss on tragets: {:.4f}' \
          .format(epoch, train_loss, eval_loss))
    print("Parameter g - Avrg: {:.4f} Min: {:.4f} Max: {:.4f}" \
          .format(model.g.data.mean().item(), model.g.data.min().item(), \
                  model.g.data.max().item()))
#    print('g equal?', (old_g == model.g.data).all())
#    old_g = model.g.data.clone()
    
    
    
    # CHRONO EVALUATION

    if args.completionPredChrono != 0:
        print("\n\n\n\nEvaluating prediction error CHORNOLOGICAL...")
        # Use only samples where genres mentionned (gm)
        RnG_valid_gm_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
        valid_gm_dataset = Utils.RnGChronoDataset(RnG_valid_gm_data, dict_genresInter_idx_UiD, nb_movies, \
                                                  popularity, args.DEVICE, args.exclude_genres, True, args.noiseEval, args.top_cut)
                                                                                # True because: Nerver data merge in Chrono
        valid_gm_loader = torch.utils.data.DataLoader(valid_gm_dataset, batch_size=args.batch, shuffle=True, **kwargs)    

 
        l1, l0, e1, e0, a1, a0, mr1, mr0, r1, r0, d1, d0 = \
             Utils.EvalPredictionRnGChrono(valid_gm_loader, model, \
                                           criterion, args.completionPredChrono, args.topx)
        
        
        print("\n  ====> RESULTS <==== \n")
        print("Global avrg pred error with {:.4f} and without {:.4f}".format(l1, l0))
    
        
        print("\n  ==> BY Nb of mentions, on to be mentionned Liked <== \n")
        
        
        avrg_e1, avrg_e0 = Utils.ChronoPlot(e1, e0, 'Avrg pred error')
        print("ReDial liked avrg pred error with {:.4f} and without {:.4f}".format(avrg_e1, avrg_e0))
        
        avrg_a1, avrg_a0 = Utils.ChronoPlot(a1, a0, 'Avrg_rank')
        print("ReDial liked avrg ranks with {:.2f} and without {:.2f}".format(avrg_a1, avrg_a0))
        
        avrg_mr1, avrg_mr0 = Utils.ChronoPlot(mr1, mr0, 'MMRR')
        print("ReDial MMRR with {:.2f} and without {:.2f}".format(avrg_mr1, avrg_mr0))
        
        avrg_r1, avrg_r0 = Utils.ChronoPlot(r1, r0, 'MRR')
        print("ReDial MRR with {:.2f} and without {:.2f}".format(avrg_r1, avrg_r0))
        
        avrg_d1, avrg_d0 = Utils.ChronoPlot(d1, d0, 'NDCG')
        print("ReDial NDCG with {:.2f} and without {:.2f}".format(avrg_d1, avrg_d0))
        
        
        # BY EPOCH GRAPHS
        
        print("\n\n\n\n  => BY Epoch <= \n")
        global_pred_err_epoch.append((l1.item(),l0.item()))
        ReDial_pred_err_epoch.append((avrg_e1,avrg_e0))
        avrg_ranks_epoch.append((avrg_a1,avrg_a0))
        
        Utils.EpochPlot(global_pred_err_epoch, 'Global avrg pred error')
        Utils.EpochPlot(ReDial_pred_err_epoch, 'ReDial liked avrg pred error')
        Utils.EpochPlot(avrg_ranks_epoch, 'ReDial liked avrg ranks')
    
    
    

    # Patience - Stop if the Model didn't improve in the last 'patience' epochs
    patience = args.patience
    if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
        print('--------------------------------------------------------------------------------')
        print('-                               STOPPED TRAINING                               -')
        print('-  Recent valid losses:', valid_losses[-patience:])
        print('--------------------------------------------------------------------------------')
        break

    
    # Save fisrt model and model that improves valid reconstruction loss
    precedent_losses = valid_losses[:-1]
    if precedent_losses == []: precedent_losses = [0]     # Cover 1st epoch for min([])'s error
    if epoch == 0 or eval_loss < min(precedent_losses):
        print('\n   Saving...')
        state = {
                'epoch': epoch,
                'eval_loss': eval_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses
                }
        torch.save(state, './Results/AE_'+args.id+'.pth')
        print('......saved.')
        
        







#%%
######## PREDICITON ERROR EVALUATION ON FINAL MODEL ########

# FOR GLOBAL - OLD EVALUATION

print('\nEvaluation prediction error GLOBAL...')
# Loader of only 1 sample (user) in order to predict for each rating
loader_bs1 = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
pred_err, pred_rank_liked, pred_rank_disliked = Utils.EvalPredictionGenresRaw(loader_bs1, model, criterion, args.completionPred)


print("\n  ====> RESULTS <==== \n")

if len(train_losses) - train_losses.index(min(train_losses)) > patience:
    train_err = round(train_losses[-patience].item(), 4)
else: 
    train_err = round(train_losses[-1].item(), 4)
print("Best reconstruction loss TRAIN: {}".format(train_err))

if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
    valid_err = round(valid_losses[-(patience+1)].item(), 4)
else: 
    valid_err = round(valid_losses[-1].item(), 4)
print("Best reconstruction loss VALID: {}".format(valid_err))

print("\nAvrg prediction error: {}".format(round(pred_err.mean(), 4)))
#print("\nAvrg liked ranking: {}, which is in first {}%".format(int(pred_rank_liked.mean()), \
#      round(pred_rank_liked.mean()/len(Settings.l_ReDUiD)*100, 1)))
#print("Avrg disliked ranking: {}, which is in first {}%".format(int(pred_rank_disliked.mean()), \
#      round(pred_rank_disliked.mean()/len(Settings.l_ReDUiD)*100, 1)))
print("\nAvrg liked ranking: {}, which is in first {}%".format(int(pred_rank_liked.mean()), \
      round(pred_rank_liked.mean()/nb_movies*100, 1)))
print("Avrg disliked ranking: {}, which is in first {}%".format(int(pred_rank_disliked.mean()), \
      round(pred_rank_disliked.mean()/nb_movies*100, 1)))



#%%


# Eval average nDCG and MRR for liked
ndcg = []
mrr = []
for r in pred_rank_liked:
    # In Numpy
    r = np.array([r])
    ndcg.append(Utils.nDCG(r, 100))
    mrr.append(Utils.RR(r))

print('For linked movies, average nDCG is', mean(ndcg), 'and MRR is', mean(mrr))

# Eval average nDCG and MRR for disliked
ndcg = []
mrr = []
for r in pred_rank_disliked:
    # In Numpy
    r = np.array([r])
    ndcg.append(Utils.nDCG(r, 100))
    mrr.append(Utils.RR(r))

print('\nFor disliked linked movies, average nDCG is', mean(ndcg), 'and MRR is', mean(mrr))

print("\nParameter g = ", model.g.data)
print("\n g average:", model.g.data.mean())

print("\n\n\n\n\n\n\n")   



# For Orion, print results (MongoDB,...)
if args.orion:
    report_results([dict(
        name='valid_pred_rank_liked',
        type='objective',
        value=pred_rank_liked),
        dict(
        name='valid_pred_rank_DISliked',
        type='constraint',
        value=pred_rank_disliked),
        dict(
        name='valid_pred_error',
        type='constraint',
        value=pred_err),
        dict(
        name='valid_reconst_error',
        type='constraint',
        value=valid_err),
        dict(
        name='g',
        type='constraint',
        value=model.g.data.item())])

     
#%%

    
plt.plot(pred_mean_values)
plt.title('Avrg Pred value by batch')
plt.xlabel('batch')
plt.ylabel('avrg pred value')
plt.show()


plt.plot(valid_losses)
plt.title('Valid losses by epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#%%

for batch_idx, (masks, inputs, targets) in enumerate(train_loader):
    inputs[0] = inputs[0]
    inputs[1][0] = inputs[1][0]
    inputs[1][1] = inputs[1][1]
    pred = model(inputs)
    if model.model_pre.lla == 'none':
        pred = torch.nn.Sigmoid()(pred)
 #   pred = pred[:,Settings.l_ReDUiD]
    pred = pred[0] #.mean(0)
    
    pred = pred.detach().cpu().numpy()
    
    print('Genres:', inputs[1][1][0].sum(), (inputs[1][1][0]**2).sum())   
    print('** Inputs **',inputs[0][0][masks[0][0] == 1])
    
    print('**All genres:** {}, indx{}'.format((inputs[1][1]**2).sum(1), inputs[1][0]))
    
    plt.hist(pred, 100)
    plt.title('Histogram - Prediction values for one sample')
    plt.xlabel('Pred values')
    plt.ylabel('Qt.')
    plt.show()
    
    if batch_idx >= 0:break

















































    
    







































