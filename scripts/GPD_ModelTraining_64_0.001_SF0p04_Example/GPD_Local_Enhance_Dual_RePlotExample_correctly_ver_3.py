#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

import json
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

from seisbench.util import worker_seeding
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from scipy.signal import find_peaks

import logging
import time
from datetime import datetime
import psutil
import warnings
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import defaultdict

# Set random seed for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os


# In[2]:


# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging system"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{timestamp}.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Create specialized loggers
    train_logger = logging.getLogger('training')
    performance_logger = logging.getLogger('performance')
    debug_logger = logging.getLogger('debug')
    
    # Create separate log files for different aspects
    train_handler = logging.FileHandler(f'logs/training_metrics_{timestamp}.log')
    performance_handler = logging.FileHandler(f'logs/performance_{timestamp}.log')
    debug_handler = logging.FileHandler(f'logs/debug_{timestamp}.log')
    
    train_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    performance_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    train_logger.addHandler(train_handler)
    performance_logger.addHandler(performance_handler)
    debug_logger.addHandler(debug_handler)
    
    return train_logger, performance_logger, debug_logger


# In[3]:


class SystemMonitor:
    """Monitor system resources during training"""
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_system_stats(self):
        """Get current system resource usage"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'process_memory_mb': self.process.memory_info().rss / (1024**2)
        }
        
        if self.gpu_available:
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
        
        return stats


# In[4]:


class ModelAnalyzer:
    """Analyze model architecture and parameters"""
    def __init__(self, model):
        self.model = model
        
    def analyze_model(self):
        """Comprehensive model analysis"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        }
        
        # Layer analysis
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params
                })
        
        analysis['layers'] = layer_info
        return analysis


# In[5]:


class MetricsTracker:
    """Track and analyze training metrics"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.class_names = ['P-Phase', 'S-Phase', 'Noise']
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def compute_classification_metrics(self, y_true, y_pred):
        """Compute detailed classification metrics"""
        # Convert to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Get predicted classes
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_true_classes, y_pred_classes),
            'f1_macro': f1_score(y_true_classes, y_pred_classes, average='macro'),
            'f1_weighted': f1_score(y_true_classes, y_pred_classes, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes),
            'classification_report': classification_report(
                y_true_classes, y_pred_classes, 
                target_names=self.class_names, output_dict=True
            )
        }
        
        return metrics


# In[6]:


class EarlyStopping:
    """Enhanced early stopping with detailed logging"""
    def __init__(self, patience=7, verbose=False, delta=0, 
                 checkpoint_path='checkpoint.pt',
                 best_model_path='best_model.pth',
                 final_model_path='final_model.pth',
                 logger=None):
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
        self.logger = logger or logging.getLogger(__name__)
    
    def __call__(self, val_loss, model, epoch=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.save_best_model(model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            message = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if epoch is not None:
                message += f' (Epoch {epoch})'
            self.logger.info(message)
            if self.verbose:
                print(message)
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_final_model(model, epoch)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.save_best_model(model, epoch)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, epoch=None):
        message = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...'
        if epoch is not None:
            message += f' (Epoch {epoch})'
        self.logger.info(message)
        if self.verbose:
            print(message)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
    
    def save_best_model(self, model, epoch=None):
        message = f'Saving best model to {self.best_model_path}'
        if epoch is not None:
            message += f' (Epoch {epoch})'
        self.logger.info(message)
        if self.verbose:
            print(message)
        torch.save(model.state_dict(), self.best_model_path)
    
    def save_final_model(self, model, epoch=None):
        message = f'Early stopping triggered. Saving final model to {self.final_model_path}'
        if epoch is not None:
            message += f' (Epoch {epoch})'
        self.logger.info(message)
        if self.verbose:
            print(message)
        torch.save(model.state_dict(), self.final_model_path)


# In[7]:


def load_config(config_path='config.json'):
    """Load configuration from JSON file with error handling"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        logging.info(f"Config contents: {json.dumps(config, indent=2)}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        # Return default config
        default_config = {
            "device": {"use_cuda": True, "device_id": 0},
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
                "num_workers": 4,
                "sigma": 20,
                "probabilities": [0.4, 0.4, 0.2],
                "highpass": None,
                "lowpass": None,
                "early_stopping": {"patience": 10, "delta": 0.0001},
                "lr_scheduler": {"factor": 0.5, "patience": 5},
                "optimization": {"pin_memory": True, "persistent_workers": True}
            },
            "peak_detection": {"sampling_rate": 100}
        }
        logging.info("Using default configuration")
        return default_config


# In[18]:


#def setup_device(config):
#    """Enhanced device setup with detailed logging"""
#    logger = logging.getLogger(__name__)
#    
#    if config['device']['use_cuda'] and torch.cuda.is_available():
#        device = torch.device(f"cuda:{config['device']['device_id']}")
#        logger.info(f"CUDA is available. Using GPU: {device}")
#        logger.info(f"GPU Name: {torch.cuda.get_device_name(device)}")
#        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
#    else:
#        device = torch.device("cpu")
#        if config['device']['use_cuda']:
#            logger.warning("CUDA requested but not available. Using CPU instead.")
#        else:
#            logger.info("Using CPU as specified in config")
#    
#    logger.info(f"Using device: {device}")
#    return device

def setup_device(config):
    """Enhanced device setup with DataParallel support and detailed logging."""
    logger = logging.getLogger(__name__)

    if config['device']['use_cuda'] and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"CUDA is available. Number of GPUs detected: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

        if num_gpus > 1:
            device_ids = list(range(num_gpus))
            device = torch.device("cuda:0")
            logger.info(f"Using DataParallel on devices: {device_ids}")
            return device, device_ids
        else:
            device = torch.device(f"cuda:{config['device']['device_id']}")
            logger.info(f"Using single GPU: {device}")
            return device, None
    else:
        device = torch.device("cpu")
        if config['device']['use_cuda']:
            logger.warning("CUDA requested but not available. Using CPU instead.")
        else:
            logger.info("Using CPU as specified in config")
        return device, None


# In[19]:


def create_data_generators(data, config):
    """Create data generators with detailed logging"""
    logger = logging.getLogger(__name__)
    
    # Only select events with magnitude below 1
    #mask = data.metadata["source_magnitude"] < 1
    #data.filter(mask)

    train, dev, test = data.train_dev_test()
    
    logger.info(f"Dataset split sizes:")
    logger.info(f"  Training: {len(train)} samples")
    logger.info(f"  Development: {len(dev)} samples")
    logger.info(f"  Test: {len(test)} samples")
    logger.info(f"  Total: {len(data)} samples")
    
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

    phase_dict_p = {k: v for k, v in phase_dict.items() if v == "P"}
    phase_dict_s = {k: v for k, v in phase_dict.items() if v == "S"}

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    # Define filter augmentations
    filter_augs = []
    if config['training']['highpass'] is not None:
        filter_augs = [sbg.Filter(config['training']['lowpass'], config['training']['highpass'], "highpass")]
        logger.info(f"Applied highpass filter: {config['training']['highpass']} Hz")

    augmentations = [
        sbg.OneOf(
            [
                sbg.WindowAroundSample(list(phase_dict_p.keys()), samples_before=200, windowlen=400, selection="random", strategy="variable"),
                sbg.WindowAroundSample(list(phase_dict_s.keys()), samples_before=200, windowlen=400, selection="random", strategy="variable"),
                sbg.WindowAroundSample(list(phase_dict_p.keys()), samples_before=700, windowlen=400, selection="first", strategy="variable")
            ],
            probabilities=config['training']['probabilities']
        ),
        sbg.RandomWindow(windowlen=400, strategy="pad"),
        sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(sigma=config['training']['sigma'], dim=0)
    ]

    logger.info(f"Applied augmentations:")
    logger.info(f"  Window probabilities: {config['training']['probabilities']}")
    logger.info(f"  Sigma for probabilistic labeller: {config['training']['sigma']}")

    for generator in [train_generator, dev_generator, test_generator]:
        if filter_augs:
            generator.add_augmentations(filter_augs)
        generator.add_augmentations(augmentations)
        generator.add_augmentations([
            sbg.ProbabilisticPointLabeller(position=0.5),
            sbg.ChangeDtype(np.float32)
        ])

    return train_generator, dev_generator, test_generator


# In[20]:


def create_data_loaders(train_generator, dev_generator, test_generator, config):
    """Create data loaders with detailed logging"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"DataLoader configuration:")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Number of workers: {config['training']['num_workers']}")
    logger.info(f"  Pin memory: {config['training']['optimization']['pin_memory']}")
    logger.info(f"  Persistent workers: {config['training']['optimization']['persistent_workers']}")
    
    train_loader = DataLoader(
        train_generator,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['optimization']['pin_memory'],
        worker_init_fn=worker_seeding,
        persistent_workers=config['training']['optimization']['persistent_workers'],
        prefetch_factor=config['training']['optimization'].get('prefetch_factor', 2 )  # Added prefetch factor for efficiency
    )

    dev_loader = DataLoader(
        dev_generator,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['optimization']['pin_memory'],
        worker_init_fn=worker_seeding,
        persistent_workers=config['training']['optimization']['persistent_workers'],
        prefetch_factor=config['training']['optimization'].get('prefetch_factor', 2 )  # Added prefetch factor for efficiency
    )

    test_loader = DataLoader(
        test_generator,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['optimization']['pin_memory'],
        worker_init_fn=worker_seeding,
        persistent_workers=config['training']['optimization']['persistent_workers'],
        prefetch_factor=config['training']['optimization'].get('prefetch_factor', 2 )  # Added prefetch factor for efficiency
    )

    return train_loader, dev_loader, test_loader



# In[10]:


# In[21]:


def loss_fn(y_pred, y_true, eps=1e-5):
    """Vector cross entropy loss with numerical stability"""
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)
    h = h.mean()
    return -h

def evaluate_initial_performance(model, data_loader, device, logger, metrics_tracker):
    """Evaluate initial model performance before training"""
    logger.info("Evaluating initial model performance...")
    
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            
            #pred = model(batch["X"].to(device))
            #loss = loss_fn(pred, batch["y"].to(device))
            
            # Hongyu Xiao: Modify here to include softmax
            pred_logits = model(batch["X"].to(device))

            # 1. ADD THIS LINE: Convert logits to probabilities using softmax
            pred_probabilities = torch.softmax(pred_logits, dim=1)

            # 2. CHANGE THIS LINE: Use the new probabilities to calculate loss
            loss = loss_fn(pred_probabilities, batch["y"].to(device))

            total_loss += loss.item()
            num_batches += 1
            
            #all_predictions.append(pred.cpu())
            all_predictions.append(pred_logits.cpu())
            all_targets.append(batch["y"].cpu())
    
    model.train()
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Compute classification metrics
    classification_metrics = metrics_tracker.compute_classification_metrics(
        all_targets, all_predictions
    )
    
    logger.info(f"Initial Performance:")
    logger.info(f"  Loss: {avg_loss:.6f}")
    logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro): {classification_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (weighted): {classification_metrics['f1_weighted']:.4f}")
    
    # Log per-class metrics
    for class_name in ['P-Phase', 'S-Phase', 'Noise']:
        if class_name in classification_metrics['classification_report']:
            class_metrics = classification_metrics['classification_report'][class_name]
            logger.info(f"  {class_name} - Precision: {class_metrics['precision']:.4f}, "
                       f"Recall: {class_metrics['recall']:.4f}, F1: {class_metrics['f1-score']:.4f}")
    
    return avg_loss, classification_metrics

def train_loop(dataloader, model, optimizer, device, epoch, system_monitor, 
               train_logger, performance_logger, metrics_tracker):
    """Enhanced training loop with comprehensive logging"""
    size = len(dataloader.dataset)
    total_loss = 0
    num_batches = 0
    batch_times = []
    
    all_predictions = []
    all_targets = []
    
    model.train()
    epoch_start_time = time.time()
    
    for batch_id, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        # Previous code:
        ## Forward pass
        #pred = model(batch["X"].to(device))
        #loss = loss_fn(pred, batch["y"].to(device))

        # Hongyu Xiao: Modify here to include softmax
        pred_logits = model(batch["X"].to(device))

        # 1. ADD THIS LINE: Convert logits to probabilities using softmax
        pred_probabilities = torch.softmax(pred_logits, dim=1)

        # 2. CHANGE THIS LINE: Use the new probabilities to calculate loss
        loss = loss_fn(pred_probabilities, batch["y"].to(device))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect metrics
        total_loss += loss.item()
        num_batches += 1
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Store predictions for metrics calculation
        #all_predictions.append(pred.detach().cpu())
        all_predictions.append(pred_logits.detach().cpu())
        all_targets.append(batch["y"].cpu())

        # Log every 5->20 batches
        #if batch_id % 5 == 0:
        if batch_id % 20 == 0:
            loss_val, current = loss.item(), batch_id * batch["X"].shape[0]
            train_logger.info(f"Epoch {epoch}, Batch {batch_id}: loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
            
            # System monitoring every 10->50 batches
            #if batch_id % 10 == 0:
            if batch_id % 50 == 0:
                system_stats = system_monitor.get_system_stats()
                performance_logger.info(f"Epoch {epoch}, Batch {batch_id} - System stats: {system_stats}")
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches
    avg_batch_time = np.mean(batch_times)
    
    # Calculate training metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    classification_metrics = metrics_tracker.compute_classification_metrics(all_targets, all_predictions)
    
    # Log epoch summary
    train_logger.info(f"Epoch {epoch} Training Summary:")
    train_logger.info(f"  Average Loss: {avg_loss:.6f}")
    train_logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    train_logger.info(f"  F1 (macro): {classification_metrics['f1_macro']:.4f}")
    train_logger.info(f"  Epoch Time: {epoch_time:.2f}s")
    train_logger.info(f"  Average Batch Time: {avg_batch_time:.4f}s")
    train_logger.info(f"  Samples/second: {size/epoch_time:.1f}")
    
    # Update metrics tracker
    metrics_tracker.update(
        epoch=epoch,
        train_loss=avg_loss,
        train_accuracy=classification_metrics['accuracy'],
        train_f1_macro=classification_metrics['f1_macro'],
        epoch_time=epoch_time,
        avg_batch_time=avg_batch_time
    )
    
    return avg_loss, classification_metrics

def test_loop(dataloader, model, device, epoch, phase, system_monitor, 
              performance_logger, metrics_tracker):
    """Enhanced testing loop with comprehensive logging"""
    num_batches = len(dataloader)
    test_loss = 0
    all_predictions = []
    all_targets = []

    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            # Previous code:
            #pred = model(batch["X"].to(device))
            #loss = loss_fn(pred, batch["y"].to(device))

            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            # Hongyu Xiao: Modify here to include softmax
            pred_logits = model(X)
            #pred_logits = model(batch["X"].to(device)) # Data: 20250812, Hongyu Xiao

            # 1. ADD THIS LINE: Convert logits to probabilities using softmax
            pred_probabilities = torch.softmax(pred_logits, dim=1)

            # 2. CHANGE THIS LINE: Use the new probabilities to calculate loss
            loss = loss_fn(pred_probabilities, batch["y"].to(device))

            test_loss += loss.item()
        
            #all_predictions.append(pred.cpu())
            all_predictions.append(pred_logits.cpu())
            all_targets.append(batch["y"].cpu())
    
    model.train()
    
    eval_time = time.time() - start_time
    avg_loss = test_loss / num_batches
    
    # Calculate comprehensive metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    classification_metrics = metrics_tracker.compute_classification_metrics(all_targets, all_predictions)
    
    # Log results
    performance_logger.info(f"Epoch {epoch} {phase} Results:")
    performance_logger.info(f"  Average Loss: {avg_loss:>8f}")
    performance_logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
    performance_logger.info(f"  F1 (macro): {classification_metrics['f1_macro']:.4f}")
    performance_logger.info(f"  F1 (weighted): {classification_metrics['f1_weighted']:.4f}")
    performance_logger.info(f"  Evaluation Time: {eval_time:.2f}s")
    
    # Log per-class metrics
    for class_name in ['P-Phase', 'S-Phase', 'Noise']:
        if class_name in classification_metrics['classification_report']:
            class_metrics = classification_metrics['classification_report'][class_name]
            performance_logger.info(f"  {class_name} - Precision: {class_metrics['precision']:.4f}, "
                                  f"Recall: {class_metrics['recall']:.4f}, F1: {class_metrics['f1-score']:.4f}")
    
    # Log confusion matrix
    performance_logger.info(f"  Confusion Matrix:\n{classification_metrics['confusion_matrix']}")
    
    print(f"{phase} avg loss: {avg_loss:>8f}, Accuracy: {classification_metrics['accuracy']:.4f}")
    
    # Update metrics tracker
    metrics_key_prefix = 'val' if phase == 'Validation' else 'test'
    metrics_tracker.update(**{
        f'{metrics_key_prefix}_loss': avg_loss,
        f'{metrics_key_prefix}_accuracy': classification_metrics['accuracy'],
        f'{metrics_key_prefix}_f1_macro': classification_metrics['f1_macro'],
        f'{metrics_key_prefix}_f1_weighted': classification_metrics['f1_weighted']
    })
    
    return avg_loss, classification_metrics


# In[22]:


def plot_training_history(metrics_tracker, save_path="enhanced_training_history.png"):
    """Enhanced training history plot with multiple metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(metrics_tracker.metrics['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, metrics_tracker.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, metrics_tracker.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].fill_between(epochs, metrics_tracker.metrics['train_loss'], 
                           metrics_tracker.metrics['val_loss'], alpha=0.2, color='gray')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, metrics_tracker.metrics['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, metrics_tracker.metrics['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # F1 Score plot
    axes[1, 0].plot(epochs, metrics_tracker.metrics['train_f1_macro'], 'b-', label='Training F1 (macro)', linewidth=2)
    axes[1, 0].plot(epochs, metrics_tracker.metrics['val_f1_macro'], 'r-', label='Validation F1 (macro)', linewidth=2)
    axes[1, 0].set_title('Training and Validation F1 Score (Macro)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Training time plot
    if 'epoch_time' in metrics_tracker.metrics:
        axes[1, 1].plot(epochs, metrics_tracker.metrics['epoch_time'], 'g-', label='Epoch Time', linewidth=2)
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# In[23]:


def plot_sample_waveform(generator, save_path="GPD_Waveform_With_Label_Bar.png"):
    """Plot a sample waveform with its label"""
    sample = generator[np.random.randint(len(generator))]
    waveform = sample["X"]  # Shape: (3, N)
    label_vector = sample["y"]  # e.g., [0., 0., 1.]

    # Metadata
    class_names = ['P-Phase', 'S-Phase', 'Noise']
    label_class = class_names[np.argmax(label_vector)]
    mid_index = waveform.shape[1] // 2
    time_axis = np.arange(waveform.shape[1])

    # Create figure with 4 rows: 3 for waveform, 1 for label bar
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False, 
                            gridspec_kw={'height_ratios': [1, 1, 1, 1], 'hspace': 0.3})

    channel_names = ["Channel E", "Channel N", "Channel Z"]
    waveform_colors = ['#a3b18a', '#588157', '#344e41']

    # Plot waveforms
    for i in range(3):
        axs[i].plot(time_axis, waveform[i], color=waveform_colors[i], linewidth=1.5)
        axs[i].axvline(x=mid_index, color='red', linestyle='--', linewidth=1.2)
        axs[i].set_title(f"{channel_names[i]} - Seismic Waveform", fontsize=12, fontweight='bold')
        axs[i].set_ylabel("Amplitude", fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.6)

    axs[2].set_xlabel("Time (samples)", fontsize=11, fontweight='bold')

    # Plot label vector as bar chart
    axs[3].bar(class_names, label_vector, color=['#6c757d', '#adb5bd', '#ced4da'])
    axs[3].set_ylim(0, 1.2)
    axs[3].set_ylabel("Label", fontsize=10)
    axs[3].set_xlabel("Middle Point Classification", fontsize=12, fontweight='bold')
    axs[3].grid(axis='y', linestyle='--', alpha=0.4)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# In[24]:


def plot_test_sample_with_prediction(generator, model, device, save_path="GPD_Waveform_Single_Test_With_Label_Bar.png"):
    """Plot a test sample waveform with both true and predicted labels"""
    # Get a random sample from test generator
    sample = generator[np.random.randint(len(generator))]
    waveform = sample["X"]  # Shape: (3, N)
    true_label = sample["y"]  # True label

    # Get model prediction
    model.eval()
    with torch.no_grad():
        # Ensure waveform is in the correct format and on the right device
        waveform_tensor = torch.from_numpy(waveform).float()  # Ensure float type
        if waveform_tensor.dim() == 2:  # If shape is (3, N)
            waveform_tensor = waveform_tensor.unsqueeze(0)  # Add batch dimension -> (1, 3, N)
        waveform_tensor = waveform_tensor.to(device)
        
        # Get model prediction
        pred_logits = model(waveform_tensor)
        raw_logits = pred_logits.cpu().numpy()[0]
        pred_label = raw_logits

    # Metadata
    class_names = ['P-Phase', 'S-Phase', 'Noise']
    true_class = class_names[np.argmax(true_label)]
    pred_class = class_names[np.argmax(pred_label)]
    mid_index = waveform.shape[1] // 2
    time_axis = np.arange(waveform.shape[1])

    # Create figure with 5 rows: 3 for waveform, 1 for true label, 1 for predicted label
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=False, 
                           gridspec_kw={'height_ratios': [1, 1, 1, 1, 1], 'hspace': 0.3})

    channel_names = ["Channel E", "Channel N", "Channel Z"]
    waveform_colors = ['#a3b18a', '#588157', '#344e41']

    # Plot waveforms
    for i in range(3):
        axs[i].plot(time_axis, waveform[i], color=waveform_colors[i], linewidth=1.5)
        axs[i].axvline(x=mid_index, color='red', linestyle='--', linewidth=1.2)
        axs[i].set_title(f"{channel_names[i]} - Seismic Waveform", fontsize=12, fontweight='bold')
        axs[i].set_ylabel("Amplitude", fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.6)

    # Plot true label
    axs[3].bar(class_names, true_label, color=['#6c757d', '#adb5bd', '#ced4da'])
    axs[3].set_ylim(0, 1.2)
    axs[3].set_ylabel("True Label", fontsize=10)
    axs[3].set_title(f"True Classification: {true_class}", fontsize=12, fontweight='bold')
    axs[3].grid(axis='y', linestyle='--', alpha=0.4)

    # Plot predicted label
    axs[4].bar(class_names, pred_label, color=['#6c757d', '#adb5bd', '#ced4da'])
    axs[4].set_ylim(0, 1.2)
    axs[4].set_ylabel("Predicted Label", fontsize=10)
    axs[4].set_title(f"Predicted Classification: {pred_class}", fontsize=12, fontweight='bold')
    axs[4].grid(axis='y', linestyle='--', alpha=0.4)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print prediction details
    print("\nTest Sample Prediction Details:")
    print(f"True Classification: {true_class}")
    print(f"Predicted Classification: {pred_class}")
    print("\nTrue Label Probabilities:")
    for cls, prob in zip(class_names, true_label):
        print(f"  {cls}: {prob:.4f}")
    print("\nPredicted Label Probabilities:")
    for cls, prob in zip(class_names, pred_label):
        print(f"  {cls}: {prob:.4f}")


# In[28]:


def plot_multiple_test_samples_with_prediction(generator, model, device, num_examples=5, save_dir="example_label"):
    """Plot multiple test sample waveforms with both true and predicted labels, and save them in a folder."""

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    class_names = ['P-Phase', 'S-Phase', 'Noise']
    channel_names = ["Channel E", "Channel N", "Channel Z"]
    waveform_colors = ['#a3b18a', '#588157', '#344e41']

    for i in range(num_examples):
        # Get a random sample from test generator
        sample = generator[np.random.randint(len(generator))]
        waveform = sample["X"]  # Shape: (3, N)
        true_label = sample["y"]

        # Predict with model
        model.eval()
        with torch.no_grad():
            waveform_tensor = torch.from_numpy(waveform).float()
            if waveform_tensor.dim() == 2:
                waveform_tensor = waveform_tensor.unsqueeze(0)
            waveform_tensor = waveform_tensor.to(device)
            pred_logits = model(waveform_tensor)
            raw_logits = pred_logits.cpu().numpy()[0]
            pred_label = raw_logits

        # Metadata
        true_class = class_names[np.argmax(true_label)]
        pred_class = class_names[np.argmax(pred_label)]
        mid_index = waveform.shape[1] // 2
        time_axis = np.arange(waveform.shape[1])

        # Plotting
        fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=False,
                                gridspec_kw={'height_ratios': [1, 1, 1, 1, 1], 'hspace': 0.3})

        for j in range(3):
            axs[j].plot(time_axis, waveform[j], color=waveform_colors[j], linewidth=1.5)
            axs[j].axvline(x=mid_index, color='red', linestyle='--', linewidth=1.2)
            axs[j].set_title(f"{channel_names[j]} - Seismic Waveform", fontsize=12, fontweight='bold')
            axs[j].set_ylabel("Amplitude", fontsize=10)
            axs[j].grid(True, linestyle='--', alpha=0.6)

        axs[3].bar(class_names, true_label, color=['#6c757d', '#adb5bd', '#ced4da'])
        axs[3].set_ylim(0, 1.2)
        axs[3].set_ylabel("True Label", fontsize=10)
        axs[3].set_title(f"True Classification: {true_class}", fontsize=12, fontweight='bold')
        axs[3].grid(axis='y', linestyle='--', alpha=0.4)

        axs[4].bar(class_names, pred_label, color=['#6c757d', '#adb5bd', '#ced4da'])
        axs[4].set_ylim(0, 1.2)
        axs[4].set_ylabel("Predicted Label", fontsize=10)
        axs[4].set_title(f"Predicted Classification: {pred_class}", fontsize=12, fontweight='bold')
        axs[4].grid(axis='y', linestyle='--', alpha=0.4)

        # Save figure
        save_path = os.path.join(save_dir, f"example_{i+1:02d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print prediction summary
        print(f"\nExample {i+1}:")
        print(f"True Classification: {true_class}")
        print(f"Predicted Classification: {pred_class}")
        print("True Label Probabilities:")
        for cls, prob in zip(class_names, true_label):
            print(f"  {cls}: {prob:.4f}")
        print("Predicted Label Probabilities:")
        for cls, prob in zip(class_names, pred_label):
            print(f"  {cls}: {prob:.4f}")


# In[25]:


def plot_confusion_matrix(confusion_matrix, class_names, save_path="confusion_matrix.png"):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(generator, n_samples=2000, save_path="2000_Data_Trace_Distribution.png"):
    """Plot class distribution from n_samples with logging"""
    logger = logging.getLogger(__name__)
    class_counts = {"P-Phase": 0, "S-Phase": 0, "Noise": 0}
    class_names = ['P-Phase', 'S-Phase', 'Noise']
    
    logger.info(f"Analyzing class distribution from {n_samples} samples...")

    #random_indices = np.random.choice(0, len(generator), n_samples)

    #for idx in random_indices:
    #for idx in range(n_samples):
    for _ in range(n_samples):
        #sample = generator[idx]
        sample = generator[np.random.randint(len(generator))]
        label_vector = sample["y"]
        label_class = class_names[np.argmax(label_vector)]
        class_counts[label_class] += 1

    # Plot histogram of label counts
    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), 
                   color=['#6c757d', '#adb5bd', '#ced4da'])
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.title(f"Class Frequency from {n_samples} Samples", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(class_counts.values()),
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log distribution
    total = sum(class_counts.values())
    logger.info("Sample class distribution:")
    for cls, count in class_counts.items():
        percentage = (count / total) * 100
        logger.info(f"  {cls}: {count} ({percentage:.1f}%)")

    return class_counts

def plot_full_dataset_distribution(generator, save_path="All_Data_Trace_Distribution.png"):
    """Plot class distribution for the entire dataset with enhanced logging"""
    logger = logging.getLogger(__name__)
    class_counts = {"P-Phase": 0, "S-Phase": 0, "Noise": 0}
    class_names = ['P-Phase', 'S-Phase', 'Noise']
    
    total_samples = len(generator)
    logger.info(f"Analyzing full dataset distribution ({total_samples} samples)...")

    for idx in range(total_samples):
        if idx % 10000 == 0:  # Progress logging
            logger.info(f"Processed {idx}/{total_samples} samples ({idx/total_samples*100:.1f}%)")
            
        sample = generator[idx]
        label_vector = sample["y"]
        label_class = class_names[np.argmax(label_vector)]
        class_counts[label_class] += 1

    # Plot histogram of label counts
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), 
                   color=['#6c757d', '#adb5bd', '#ced4da'])
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.title("Class Frequency in Full Dataset", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(class_counts.values()),
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log comprehensive distribution analysis
    total_samples = sum(class_counts.values())
    logger.info("Full Dataset Class Distribution Summary:")
    for cls in class_names:
        count = class_counts[cls]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"  {cls}: {count:,} samples ({percentage:.2f}%)")
    logger.info(f"Total samples: {total_samples:,}")
    
    # Check for class imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    logger.info(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 3:
        logger.warning("Significant class imbalance detected! Consider using weighted loss or resampling.")

    return class_counts

def save_detailed_metrics(metrics_tracker, model_analyzer, save_path="detailed_metrics.json"):
    """Save comprehensive metrics to JSON file"""
    logger = logging.getLogger(__name__)
    
    # Prepare metrics for JSON serialization
    serializable_metrics = {}
    for key, values in metrics_tracker.metrics.items():
        if isinstance(values, list):
            # Convert numpy arrays and tensors to lists
            serializable_values = []
            for v in values:
                if hasattr(v, 'tolist'):  # numpy arrays
                    serializable_values.append(v.tolist())
                elif torch.is_tensor(v):
                    serializable_values.append(v.cpu().numpy().tolist())
                else:
                    serializable_values.append(v)
            serializable_metrics[key] = serializable_values
        else:
            serializable_metrics[key] = values
    
    # Add model analysis
    model_analysis = model_analyzer.analyze_model()
    
    # Combine all metrics
    comprehensive_metrics = {
        'training_metrics': serializable_metrics,
        'model_analysis': model_analysis,
        'final_performance': {
            'best_val_loss': min(serializable_metrics.get('val_loss', [float('inf')])),
            'best_val_accuracy': max(serializable_metrics.get('val_accuracy', [0])),
            'best_val_f1': max(serializable_metrics.get('val_f1_macro', [0])),
            'total_epochs': len(serializable_metrics.get('train_loss', [])),
            'convergence_epoch': serializable_metrics.get('val_loss', []).index(
                min(serializable_metrics.get('val_loss', [float('inf')]))
            ) + 1 if serializable_metrics.get('val_loss') else 0
        }
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(comprehensive_metrics, f, indent=2)
    
    logger.info(f"Detailed metrics saved to {save_path}")
    return comprehensive_metrics


# In[26]:


# In[ ]:


def main():
    # Setup logging
    train_logger, performance_logger, debug_logger = setup_logging()
    
    # Initialize system monitoring
    system_monitor = SystemMonitor()
    
    # Log system information
    initial_stats = system_monitor.get_system_stats()
    performance_logger.info(f"Initial system stats: {initial_stats}")
    
    try:
        # Load configuration
        config = load_config()
        
        # Setup device
        # device = setup_device(config)
        
        # Load model with detailed analysis
        debug_logger.info("Loading GPD model...")
       	model = sbm.GPD.from_pretrained("original")
        
        # Setup device
        device,device_ids = setup_device(config)
        if device_ids:
            model = torch.nn.DataParallel(model,device_ids=device_ids)
        model.to(device)
        
        # Analyze model
        model_analyzer = ModelAnalyzer(model)
        model_analysis = model_analyzer.analyze_model()
        debug_logger.info(f"Model Analysis: {json.dumps(model_analysis, indent=2)}")
        
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
        
        # Load data
        debug_logger.info("Loading OKLA_1Mil_120s_Ver_3 dataset...")
        start_time = time.time()
        data = sbd.OKLA_1Mil_120s_Ver_3(
            sampling_rate=config['peak_detection']['sampling_rate'], 
            force=True, 
            component_order="ENZ"
        )
        load_time = time.time() - start_time
        debug_logger.info(f"Dataset loaded in {load_time:.2f} seconds")
        
        # Create generators and loaders
        train_generator, dev_generator, test_generator = create_data_generators(data, config)
        train_loader, dev_loader, test_loader = create_data_loaders(
            train_generator, dev_generator, test_generator, config
        )
        
        # Evaluate initial performance
        initial_loss, initial_metrics = evaluate_initial_performance(
            model, dev_loader, device, performance_logger, metrics_tracker
        )
        
        # Plot dataset descriptions
        debug_logger.info("Generating dataset visualizations...")
        plot_sample_waveform(train_generator)
        plot_class_distribution(train_generator)
        #plot_full_dataset_distribution(train_generator)
        
        plot_multiple_test_samples_with_prediction(test_generator, model, device)
        
        # Setup optimizer with detailed logging
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        debug_logger.info(f"Optimizer: {optimizer}")
        debug_logger.info(f"Learning rate: {config['training']['learning_rate']}")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            verbose=True
        )
        debug_logger.info(f"Learning rate scheduler: {scheduler}")
        
        # Initialize early stopping with logging
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            verbose=True,
            delta=config['training']['early_stopping']['delta'],
            logger=train_logger
        )
        
        # Training loop with comprehensive monitoring
        debug_logger.info(f"Starting training for {config['training']['epochs']} epochs...")
        training_start_time = time.time()
        
        for epoch in range(1, config['training']['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_metrics = train_loop(
                train_loader, model, optimizer, device, epoch, 
                system_monitor, train_logger, performance_logger, metrics_tracker
            )
            
            # Validation
            val_loss, val_metrics = test_loop(
                dev_loader, model, device, epoch, "Validation",
                system_monitor, performance_logger, metrics_tracker
            )
            
            epoch_time = time.time() - epoch_start_time
            
            # Update learning rate scheduler
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                train_logger.info(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            
            # Log epoch summary
            train_logger.info(f"Epoch {epoch} Complete:")
            train_logger.info(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            train_logger.info(f"  Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            train_logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            
            # System monitoring
            current_stats = system_monitor.get_system_stats()
            performance_logger.info(f"Post-epoch {epoch} system stats: {current_stats}")
            
            # Check early stopping
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                train_logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': dict(metrics_tracker.metrics)
                }, checkpoint_path)
                debug_logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        total_training_time = time.time() - training_start_time
        train_logger.info(f"Training completed in {total_training_time:.2f} seconds")
        
        # Plot training history
        plot_training_history(metrics_tracker)
        
        # Load the best model for final evaluation
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth'))
            debug_logger.info("Loaded best model for final evaluation")
        
        # Final evaluation on test set
        performance_logger.info("Starting final evaluation on test set...")
        test_loss, test_metrics = test_loop(
            test_loader, model, device, "Final Test", "Final Test",
            system_monitor, performance_logger, metrics_tracker
        )
        
        # Plot final confusion matrix
        plot_confusion_matrix(
            test_metrics['confusion_matrix'], 
            ['P-Phase', 'S-Phase', 'Noise'],
            "final_confusion_matrix.png"
        )
        
        # Generate test sample visualization
        debug_logger.info("Generating test sample visualization...")
        plot_test_sample_with_prediction(test_generator, model, device)
        
        # Save comprehensive metrics
        final_metrics = save_detailed_metrics(metrics_tracker, model_analyzer)
        
        # Final summary
        performance_logger.info("="*50)
        performance_logger.info("FINAL TRAINING SUMMARY")
        performance_logger.info("="*50)
        performance_logger.info(f"Initial Loss: {initial_loss:.6f}")
        performance_logger.info(f"Final Test Loss: {test_loss:.6f}")
        performance_logger.info(f"Best Validation Loss: {min(metrics_tracker.metrics['val_loss']):.6f}")
        performance_logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        performance_logger.info(f"Final Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        performance_logger.info(f"Total Training Time: {total_training_time:.2f} seconds")
        performance_logger.info(f"Total Epochs: {len(metrics_tracker.metrics['train_loss'])}")
        performance_logger.info("="*50)
        
    except Exception as e:
        debug_logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Final system stats
        final_stats = system_monitor.get_system_stats()
        performance_logger.info(f"Final system stats: {final_stats}")



# In[11]:


def main():
    # Load configuration
    config = load_config()

    # Setup device
    device, device_ids = setup_device(config)

    # Load model
    model = sbm.GPD.from_pretrained("original")
    if device_ids:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    # Load the best model if available
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Best model not found, using pretrained model")

    # Load dataset
    data = sbd.OKLA_1Mil_120s_Ver_3(
        sampling_rate=config['peak_detection']['sampling_rate'],
        force=True,
        component_order="ENZ"
    )

    # Create data generators
    _, _, test_generator = create_data_generators(data, config)

    # Plot multiple test samples with predictions
    plot_multiple_test_samples_with_prediction(test_generator, model, device)


# In[12]:


if __name__ == "__main__":
    main()


# In[ ]:




