#!/usr/bin/env python
# coding: utf-8

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import json
import time

import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from scipy.signal import find_peaks


# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Set random seed for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os

# Loader the picker
model = sbm.EQTransformer.from_pretrained("original")

# Set up device
device = torch.device(f"cuda:{config['device']['device_id']}" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
print(f"Using device: {device}")
model.to(device)

# Load the data
print("Loading data...")
data = sbd.OKLA_1Mil_120s_Ver_3(sampling_rate=100, force=True, component_order="ENZ")


# Create a random sample
sample_fraction = config['training']['sample_fraction']
print(f"Creating random sample of {sample_fraction*100}% of the data...")

# Create a random mask for sampling
np.random.seed(42)  # For reproducibility

mask = np.random.random(len(data)) < sample_fraction
data.filter(mask)

print(f"Sampled dataset size: {len(data)}")

#print("Sample metadata:")
#data.metadata.head()

# Split data
train, dev, test = data.train_dev_test()

#train = data.train()
#dev = data.dev()
#test = data.test()

print("Train:", train)
print("Dev:", dev)
print("Test:", test)

# Set up data augmentation

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}

# Create the data generators for training and validation
train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)

# Define phase lists for labeling
p_phases = [key for key, val in phase_dict.items() if val == "P"]
s_phases = [key for key, val in phase_dict.items() if val == "S"]

# Setup detection labeller
detection_labeller = sbg.DetectionLabeller(
    p_phases, s_phases=s_phases,key=("X", "detections")
)

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)

# Select a sample from the test generator
sample = train_generator[np.random.randint(len(train_generator))]

# Extract waveform (X), labels (y), and detections
waveform = sample["X"]  # Shape: (3, N)
#labels = sample["y"]  # Shape: (3, N)
#detections = sample["detections"]  # Shape: (1, N)

time_axis = np.arange(waveform.shape[1])  # Create a time axis

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.3})

channel_names = ["Channel E", "Channel N", "Channel Z"]

waveform_colors = ['#a3b18a', '#588157', '#344e41']  # Custom colors for channels
label_colors = ['#15616d', '#ff7d00']
detection_color = '#c1121f'  # Red for detections

# Plot waveforms
for i in range(3):
    axs[i].plot(time_axis, waveform[i], color=waveform_colors[i], linewidth=1.5)
    axs[i].set_title(f"{channel_names[i]} - Seismic Waveform", fontsize=12, fontweight='bold')
    axs[i].set_ylabel("Amplitude", fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.6)

# Plot dataset ground truth labels
#axs[3].plot(time_axis, labels[0], color=label_colors[0], linewidth=1.5, label="P-phase")
#axs[3].plot(time_axis, labels[1], color=label_colors[1], linewidth=1.5, label="S-phase")
#axs[3].plot(time_axis, detections[0], color=detection_color, linestyle="--", linewidth=1.8, label="True Event")
#axs[3].set_title("Dataset Ground Truth", fontsize=12, fontweight='bold')
#axs[3].set_ylabel("Probability", fontsize=10)
axs[2].grid(True, linestyle='--', alpha=0.6)
#axs[2].legend(fontsize=10, loc="upper left")

# Improve x-axis visibility
axs[2].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')
axs[2].tick_params(axis='x', labelsize=10)

plt.savefig("EQT_Model_Data_Example_Raw.png", dpi=300, bbox_inches='tight')
#plt.show()

augmentations = [
    sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=6000, windowlen=12000, selection="random", strategy="variable"),
    #sbg.RandomWindow(windowlen=3000, strategy="pad"),
    #sbg.FixedWindow(p0=-3000, windowlen=6000, strategy="pad"),
    sbg.RandomWindow(windowlen=6000, strategy="pad"),
    sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    #sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(sigma=30, dim=0),
    detection_labeller,
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)
test_generator.add_augmentations(augmentations)


# In[63]:


sample = train_generator[102]

print("Example:", sample)
print("Example:", sample["X"])
print("Example:", sample["X"].shape)
print("Example:", sample["y"])
print("Example:", sample["y"].shape)
print("Example:", sample["detections"])

# Select a sample from the test generator
sample = train_generator[np.random.randint(len(train_generator))]

# Extract waveform (X), labels (y), and detections
waveform = sample["X"]  # Shape: (3, N)
labels = sample["y"]  # Shape: (3, N)
detections = sample["detections"]  # Shape: (1, N)

time_axis = np.arange(waveform.shape[1])  # Create a time axis

# Create figure and subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.3})

channel_names = ["Channel E", "Channel N", "Channel Z"]

waveform_colors = ['#a3b18a', '#588157', '#344e41']  # Custom colors for channels
label_colors = ['#15616d', '#ff7d00']
detection_color = '#c1121f'  # Red for detections

# Plot waveforms
for i in range(3):
    axs[i].plot(time_axis, waveform[i], color=waveform_colors[i], linewidth=1.5)
    axs[i].set_title(f"{channel_names[i]} - Seismic Waveform", fontsize=12, fontweight='bold')
    axs[i].set_ylabel("Amplitude", fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.6)

# Plot dataset ground truth labels
axs[3].plot(time_axis, labels[0], color=label_colors[0], linewidth=1.5, label="P-phase")
axs[3].plot(time_axis, labels[1], color=label_colors[1], linewidth=1.5, label="S-phase")
axs[3].plot(time_axis, detections[0], color=detection_color, linestyle="--", linewidth=1.8, label="True Event")
axs[3].set_title("Dataset Ground Truth", fontsize=12, fontweight='bold')
axs[3].set_ylabel("Probability", fontsize=10)
axs[3].grid(True, linestyle='--', alpha=0.6)
axs[3].legend(fontsize=10, loc="upper left")

# Improve x-axis visibility
axs[3].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')
axs[3].tick_params(axis='x', labelsize=10)

plt.savefig("EQT_Model_Training_Example.png", dpi=300, bbox_inches='tight')
#plt.show()


# In[64]:


augmentation_Part_2 = [
    sbg.ChangeDtype(np.float32, "X"),
    sbg.ChangeDtype(np.float32, "y"),
    sbg.ChangeDtype(np.float32, "detections"),
]


# In[45]:

train_generator.add_augmentations(augmentation_Part_2)
dev_generator.add_augmentations(augmentation_Part_2)
test_generator.add_augmentations(augmentation_Part_2)


# In[65]:


# Select a sample from the test generator
sample = train_generator[np.random.randint(len(train_generator))]

# Extract waveform (X), labels (y), and detections
waveform = sample["X"]  # Shape: (3, N)
labels = sample["y"]  # Shape: (3, N)
detections = sample["detections"]  # Shape: (1, N)

time_axis = np.arange(waveform.shape[1])  # Create a time axis

# Create figure and subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.3})

channel_names = ["Channel E", "Channel N", "Channel Z"]

waveform_colors = ['#a3b18a', '#588157', '#344e41']  # Custom colors for channels
label_colors = ['#15616d', '#ff7d00']
detection_color = '#c1121f'  # Red for detections

# Plot waveforms
for i in range(3):
    axs[i].plot(time_axis, waveform[i], color=waveform_colors[i], linewidth=1.5)
    axs[i].set_title(f"{channel_names[i]} - Seismic Waveform", fontsize=12, fontweight='bold')
    axs[i].set_ylabel("Amplitude", fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.6)

# Plot dataset ground truth labels
axs[3].plot(time_axis, labels[0], color=label_colors[0], linewidth=1.5, label="P-phase")
axs[3].plot(time_axis, labels[1], color=label_colors[1], linewidth=1.5, label="S-phase")
axs[3].plot(time_axis, detections[0], color=detection_color, linestyle="--", linewidth=1.8, label="True Event")
axs[3].set_title("Dataset Ground Truth", fontsize=12, fontweight='bold')
axs[3].set_ylabel("Probability", fontsize=10)
axs[3].grid(True, linestyle='--', alpha=0.6)
axs[3].legend(fontsize=10, loc="upper left")

# Improve x-axis visibility
axs[3].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')
axs[3].tick_params(axis='x', labelsize=10)

plt.savefig("EQT_Model_Training_ExampleAfterAugmentation_2.png", dpi=300, bbox_inches='tight')
#plt.show()


# In[66]:


# Parameters for peak detection
sampling_rate = config['peak_detection']['sampling_rate']
height = config['peak_detection']['height']
distance = config['peak_detection']['distance']

batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']

# Load the data for machine learning
train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
val_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)


# Loss function and their weight assignment
loss_fn = torch.nn.BCELoss()
loss_weights = tuple(config['training']['loss_weights'])

# Learning rate and number of epochs
learning_rate = config['training']['learning_rate']
epochs = config['training']['epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path='checkpoint.pt', 
                 best_model_path='best_model.pth', final_model_path='final_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.final_model_path = final_model_path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.save_best_model(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_final_model(model)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.save_best_model(model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
    
    def save_best_model(self, model):
        if self.verbose:
            print(f'Saving best model to {self.best_model_path}')
        torch.save(model.state_dict(), self.best_model_path)
    
    def save_final_model(self, model):
        if self.verbose:
            print(f'Early stopping triggered. Saving final model to {self.final_model_path}')
        torch.save(model.state_dict(), self.final_model_path)



# In[67]:


def train_model(train_loader, val_loader, model, optimizer, loss_fn, loss_weights, num_epochs=25, patience=7):
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3, verbose=True)
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        model.train()
        train_loss = 0
        size = len(train_loader.dataset)
        
        for batch_id, batch in enumerate(train_loader):
            # Compute prediction and loss
            det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
            p_true = batch["y"][:,0].to(model.device)
            s_true = batch["y"][:,1].to(model.device)
            det_true = batch["detections"][:,0].to(model.device)
            
            loss = loss_weights[0]*loss_fn(det_pred, det_true) + \
                   loss_weights[1]*loss_fn(p_pred, p_true) + \
                   loss_weights[2]*loss_fn(s_pred, s_true)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if batch_id % 10 == 0:
            #if batch_id % 5 == 0:
                loss_val, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss_val:>7f} [{current:>5d}/{size:>5d}]")
            
            train_loss += loss.item()
        
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                det_pred, p_pred, s_pred = model(batch["X"].to(model.device))
                p_true = batch["y"][:,0].to(model.device)
                s_true = batch["y"][:,1].to(model.device)
                det_true = batch["detections"][:,0].to(model.device)
                
                val_loss += (loss_weights[0]*loss_fn(det_pred, det_true).item() + 
                           loss_weights[1]*loss_fn(p_pred, p_true).item() + 
                           loss_weights[2]*loss_fn(s_pred, s_true).item())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1} results: Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
 
    # Visualize the training history
    plot_training_history(history)
    return model, history

# Function to visualize training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the gap between training and validation loss
    plt.fill_between(range(len(history['train_loss'])), 
                    history['train_loss'], history['val_loss'],
                    alpha=0.3, color='red', 
                    where=(np.array(history['val_loss']) > np.array(history['train_loss'])),
                    label='Potential Overfitting Gap')
    
    #plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


# In[ ]:


# Usage example
model, history = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    loss_weights=loss_weights,
    num_epochs=config['training']['epochs'],
    patience=config['training']['patience']
)


# In[ ]:





