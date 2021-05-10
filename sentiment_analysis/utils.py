'''
Contains various utilities from data preprocessing, and model training
'''

import re
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import nltk
from sklearn.model_selection import train_test_split
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words("english")
lm = WordNetLemmatizer()

# Helper Functions

def  clean_text(text):
    '''
    Remove Punctuation and Numbers
    '''
    text = text.lower()
    text = text.replace("<br>", "")
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)  
    # remove numbers
    text = re.sub(r"\d+", "", text)
    
    return text

def remove_stop_words(word_list):
    '''
    Remove stop words like ours, if etc
    '''
    new_word_list = [word for word in word_list if word not in stop_words]
    
    return new_word_list

def remove_br(word_list):
    '''
    Remove br at the end of the word in a word list
    '''
    new_word_list = [re.sub("br$", "", word) for word in word_list]
    return new_word_list

def tokenize_text(text):
    '''
    Tokenize Text
    '''
    return word_tokenize(text)

def lemmatize_text(word_list):
    '''
    Lemmatize text
    '''
    new_word_list = [lm.lemmatize(word) for word in word_list]
    return new_word_list

def pad_sequence(sequence, max_sequence_length):
    '''
    Pad sequences to max_sequence_length with padding_idx = 0
    '''
    padding_idx = 0 
    sequence_length = len(sequence)
    if sequence_length > max_sequence_length:
        sequence = sequence[:max_sequence_length]
    elif sequence_length < max_sequence_length:
        for i in range(max_sequence_length - sequence_length):
            sequence.append(padding_idx)
    return sequence

def preprocess_text(text):
    '''
    clean sentences, remove punctuation, perform tokenization, stop word removal, lemmatization.
    '''
    text = clean_text(text)
    word_list = tokenize_text(text)
    word_list = remove_br(word_list)
    word_list = remove_stop_words(word_list)
    word_list = lemmatize_text(word_list)

    return word_list

def train_test_split_tensors(X, y, train_size=0.8, batch_size=32):
    '''
    X - training data
        numpy array 
        shape: (n_sequences, max_sequence_length)
    y - target
        numpy array
        shape: (n_sequences)
    train_size - % of the training data
        float
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    print("Data Split in the following way:")
    print(f"X train: {X_train.shape}\n X test: {X_test.shape} \n Y train: {Y_train.shape}\n Y test: {Y_test.shape}")
    print("Creating dataloaders...")

    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    Y_train_tensor = torch.from_numpy(Y_train)
    Y_test_tensor = torch.from_numpy(Y_test)

    train_ds = TensorDataset(X_train_tensor, Y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, Y_test_tensor)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    print("Done")

    return train_dataloader, test_dataloader

def get_accuracy(model, train_dataloader, val_dataloader, device):
    result = dict()
    for mode, loader in [("train", train_dataloader), ("val", val_dataloader)]:
        corrects = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in loader:
                sequences = sequences.to(device=device)
                labels = labels.to(device=device)
                outputs = model(sequences)
                _, preds = torch.max(outputs, dim=1)
                total += labels.shape[0]
                corrects += int(sum(preds == labels))

        result[mode] = round(corrects/total, 4) * 100
        
    return result

def train_model(n_epochs, model, train_dataloader, test_dataloader, loss, optimizer, device):
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in tqdm(range(n_epochs)):
        
        # train
        cummulative_loss = 0
        n_batches = 0
        for sequences, labels in train_dataloader:
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
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
        cummulative_loss_val = 0
        n_batches_val = 0
        for sequences, labels in test_dataloader:    
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
            with torch.no_grad():
                outputs = model(sequences)
                val_loss = loss(outputs, labels) 
                
            cummulative_loss_val += val_loss
            n_batches_val += 1
            
        loss_per_epoch_val = cummulative_loss_val / n_batches_val
        val_loss_list.append(loss_per_epoch_val)

        acc = get_accuracy(model, train_dataloader, test_dataloader, device)
        train_acc_list.append(acc["train"])
        val_acc_list.append(acc["val"])

    return train_acc_list, val_acc_list, train_loss_list, val_loss_list