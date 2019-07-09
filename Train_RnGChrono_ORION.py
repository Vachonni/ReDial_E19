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
# Use only samples where there is a genres mention
valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
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
# Use only samples where there is a genres mention
valid_g_dataset = Utils.RnGChronoDataset(valid_g_data, dict_genresInter_idx_UiD, \
                                         nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                         args.no_data_merge, args.noiseEval, args.top_cut)
# FOR CHRONO (hence no_data_merge is True) + use only samples where there is a genres mention
valid_chrono_dataset = Utils.RnGChronoDataset(valid_g_data, dict_genresInter_idx_UiD, \
                                         nb_movies, popularity, args.DEVICE, args.exclude_genres, \
                                         True, args.noiseEval, args.top_cut)           


######## CREATE DATALOADER
print('******* Creating dataloaders *******\n\n')    
kwargs = {}
if(args.DEVICE == "cuda"):
    kwargs = {'num_workers': 0, 'pin_memory': False}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=True, **kwargs)    
# For PredRaw - Loader of only 1 sample (user) 
loader_bs1 = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
# For PredChrono
valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    




######## RUN TRAINING AND VALIDATION  ########

train_losses = []
valid_losses = []

l_loss_epoch = []
l_avrg_rank_epoch = []
l_rr_epoch = []
l_ndcg_epoch = []

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
    
    

        
        
#        # BY EPOCH GRAPHS
#        
#        print("\n\n\n\n  => BY Epoch <= \n")
#        global_pred_err_epoch.append((l1.item(),l0.item()))
#        ReDial_pred_err_epoch.append((avrg_e1,avrg_e0))
#        avrg_ranks_epoch.append((avrg_a1,avrg_a0))
#        
#        Utils.EpochPlot(global_pred_err_epoch, 'Global avrg pred error')
#        Utils.EpochPlot(ReDial_pred_err_epoch, 'ReDial liked avrg pred error')
#        Utils.EpochPlot(avrg_ranks_epoch, 'ReDial liked avrg ranks')
    
    
    # PRINT BY EPOCH ON DATA NO CHRONO
    
    print('\nPredRaw...')
    
    lgl, lnl, lgn, lnn, agl, anl, agn, ann, rgl, rnl, rgn, rnn, ngl, nnl, ngn, nnn = \
         Utils.EvalPredictionGenresRaw(loader_bs1, model, criterion, args.completionPred)
     
    l_loss_epoch.append((mean(lgl), mean(lnl), mean(lgn), mean(lnn)))
    Utils.EpochPlot(l_loss_epoch, 'Avrg error by epoch, PredRaw')
    l_avrg_rank_epoch.append((mean(agl), mean(anl), mean(agn), mean(ann)))
    Utils.EpochPlot(l_avrg_rank_epoch, 'Avrg ranks by epoch, PredRaw')
    l_rr_epoch.append((mean(rgl), mean(rnl), mean(rgn), mean(rnn)))
    Utils.EpochPlot(l_rr_epoch, 'Avrg RR by epoch, PredRaw')
    l_ndcg_epoch.append((mean(ngl), mean(nnl), mean(ngn), mean(nnn)))
    Utils.EpochPlot(l_ndcg_epoch, 'Avrg NDCG by epoch, PredRaw')    
    
    


    # Patience - Stop if the Model didn't improve in the last 'patience' epochs
    patience = args.patience
    if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
        print('--------------------------------------------------------------------------------')
        print('-                               STOPPED TRAINING                               -')
        print('-  Recent valid losses:', valid_losses[-patience:])
        print('--------------------------------------------------------------------------------')
        break

    
    # SAVING - Fisrt model and model that improves valid reconstruction loss
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

#print('\nPredRaw...')
#
#pred_err, pred_rank_liked, pred_rank_disliked = Utils.EvalPredictionGenresRaw(loader_bs1, model, criterion, args.completionPred)


print("\n\n\n\n\n  ====> RESULTS <==== \n\n")

if len(train_losses) - train_losses.index(min(train_losses)) > patience:
    train_err = round(train_losses[-patience].item(), 4)
else: 
    train_err = round(train_losses[-1].item(), 4)
print("Best Reconstruction Loss TRAIN: {}".format(train_err))

if len(valid_losses) - valid_losses.index(min(valid_losses)) > patience:
    valid_err = round(valid_losses[-(patience+1)].item(), 4)
else: 
    valid_err = round(valid_losses[-1].item(), 4)
print("Best Reconstruction Loss VALID: {}".format(valid_err))

print("\nAvrg Prediction Error on Liked: {:.4f}".format(mean(lgl+lnl)))
print("Avrg Prediction Error on Not Liked: {:.4f}".format(mean(lgn+lnn)))

print('\n\n\nRANKS (avrg)\n')
print("Avrg liked ranking: {:.0f}, which is in first {:.1f}%".format(mean(agl+anl), \
      mean(agl+anl)/nb_movies*100))
print("Avrg disliked ranking: {:.0f}, which is in first {:.1f}%".format(mean(agn+ann), \
      mean(agn+ann)/nb_movies*100))
print('-----')
print("Genres + liked: {:.0f}".format(mean(agl)))
print("No Genres + liked: {:.0f}".format(mean(anl)))
print("Genres + Not liked: {:.0f}".format(mean(agn)))
print("No Genres + Not liked: {:.0f}".format(mean(ann)))


print('\n\n\nNDCG (avrg)\n')
print("Liked: {:.4f}".format(mean(ngl+nnl)))
print("Not liked: {:.4f}".format(mean(ngn+nnn)))
print('-----')
print("Genres + liked: {:.4f}".format(mean(ngl)))
print("No Genres + liked: {:.4f}".format(mean(nnl)))
print("Genres + Not liked: {:.4f}".format(mean(ngn)))
print("No Genres + Not liked: {:.4f}".format(mean(nnn)))

#%%

#
#
## For Orion, print results (MongoDB,...)
#if args.orion:
#    report_results([dict(
#        name='valid_pred_rank_liked',
#        type='objective',
#        value=pred_rank_liked),
#        dict(
#        name='valid_pred_rank_DISliked',
#        type='constraint',
#        value=pred_rank_disliked),
#        dict(
#        name='valid_pred_error',
#        type='constraint',
#        value=pred_err),
#        dict(
#        name='valid_reconst_error',
#        type='constraint',
#        value=valid_err),
#        dict(
#        name='g',
#        type='constraint',
#        value=model.g.data.item())])
#
#     
#%%

print('\n\n\n\n\n')
    
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

















































    
    







































