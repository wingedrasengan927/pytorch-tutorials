'''
Contains various utilities from data preprocessing, and model training
'''

import torch
from tqdm import tqdm

def train_model(n_epochs, model, train_dataloader, test_dataloader, loss, optimizer, device):
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(n_epochs)):
        
        # train
        model.train()
        
        cummulative_loss = 0
        n_batches = 0
        for sequences, labels in train_dataloader:
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
            labels = labels.unsqueeze(1)
            
            outputs = model(sequences)
            train_loss = loss(outputs, labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            cummulative_loss += train_loss
            n_batches += 1
        
        loss_per_epoch = cummulative_loss / n_batches
        train_loss_list.append(loss_per_epoch)
        
        # val
        model.eval()
        
        cummulative_loss_val = 0
        n_batches_val = 0
        for sequences, labels in test_dataloader:    
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
            labels = labels.unsqueeze(1)
            
            with torch.no_grad():
                outputs = model(sequences)
                val_loss = loss(outputs, labels) 
                
            cummulative_loss_val += val_loss
            n_batches_val += 1
            
        loss_per_epoch_val = cummulative_loss_val / n_batches_val
        val_loss_list.append(loss_per_epoch_val)

    return train_loss_list, val_loss_list

