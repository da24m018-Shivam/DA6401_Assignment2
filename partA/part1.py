import os
import random
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import gc
import traceback

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

class CNNModel(pl.LightningModule):
    def __init__(self,
                 input_channels=3,  # RGB images
                 num_classes=10,
                 filters=[32, 64, 128, 256, 512],  # Number of filters in each conv layer
                 filter_size=3,     # Size of filters (k×k)
                 activation='relu', # Activation function
                 dense_neurons=512, # Number of neurons in dense layer
                 dropout_rate=0.5,  # Dropout rate
                 use_batchnorm=True, # Whether to use batch normalization
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 optimizer='adam',
                 lr_decay_factor=0.1,
                 lr_patience=3
                ):
        super(CNNModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.ReLU()  # Default to ReLU

        # Create convolution blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels

        for i, out_channels in enumerate(filters):
            block = nn.Sequential()

            # Convolution layer
            block.add_module(f"conv{i+1}", nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2))

            # Batch normalization (optional)
            if use_batchnorm:
                block.add_module(f"batchnorm{i+1}", nn.BatchNorm2d(out_channels))

            # Activation function
            block.add_module(f"activation{i+1}", self.activation)

            # Max pooling
            block.add_module(f"maxpool{i+1}", nn.MaxPool2d(kernel_size=2, stride=2))

            # Add dropout after each block except the last one
            if i < len(filters) - 1 and dropout_rate > 0:
                block.add_module(f"dropout{i+1}", nn.Dropout(dropout_rate))

            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate output size of last conv layer
        # Assuming input image size is 256×256
        output_size = 256 // (2 ** len(filters))  # Due to max pooling with stride 2
        flattened_size = filters[-1] * output_size * output_size

        # Dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flattened_size, dense_neurons)
        self.dense_activation = self.activation
        self.dense_dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_neurons, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Pass through convolution blocks
        for block in self.conv_blocks:
            x = block(x)

        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.dense_dropout(x)
        x = self.output(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_test_epoch_start(self):
        self.test_step_outputs = []
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        # Save output for later use
        output = {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'targets': y}
        self.test_step_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self):
        # Get outputs from the test_step
        outputs = self.test_step_outputs
        
        # Concatenate all predictions and targets for confusion matrix
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        self.test_predictions = all_preds
        self.test_targets = all_targets
    
    def configure_optimizers(self):
        # Set up optimizer based on configuration
        if self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.hparams.learning_rate, 
                momentum=0.9, 
                weight_decay=self.hparams.weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay
            )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.lr_decay_factor,
            patience=self.hparams.lr_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }
    
    def calculate_num_params(self):
        """Calculate the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())

def get_transforms(augment=True):
    """Create image transformations pipeline"""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CNNLightningDataModule(pl.LightningDataModule):
    """Lightning data module for handling data loading and preparation"""
    def __init__(self, data_dir, batch_size=64, val_ratio=0.2, augment=True, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.augment = augment
        self.num_workers = num_workers
        
    def prepare_data(self):
        # This method is used to download or prepare data, called once
        # For this use case, we already have data downloaded
        pass
    
    def setup(self, stage=None):
        # Define transforms
        train_transform = get_transforms(augment=self.augment)
        val_test_transform = get_transforms(augment=False)
        
        if stage == 'fit' or stage is None:
            # Load full training dataset
            full_train_dataset = ImageFolder(root=os.path.join(self.data_dir, "train"), transform=None)
            self.class_names = full_train_dataset.classes
            
            # Group indices by class
            class_indices = defaultdict(list)
            for idx, (_, class_idx) in enumerate(full_train_dataset.samples):
                class_indices[class_idx].append(idx)
            
            train_indices = []
            val_indices = []
            
            # Split each class to maintain class balance
            for class_idx, indices in class_indices.items():
                # Calculate number of validation samples for this class
                n_val = int(len(indices) * self.val_ratio)
                
                # Random split for this class
                val_idx_for_class = np.random.choice(indices, size=n_val, replace=False)
                train_idx_for_class = np.array([idx for idx in indices if idx not in val_idx_for_class])
                
                # Add to our lists
                train_indices.extend(train_idx_for_class)
                val_indices.extend(val_idx_for_class)
            
            # Create datasets with appropriate transforms
            self.train_data = Subset(full_train_dataset, train_indices)
            self.val_data = Subset(full_train_dataset, val_indices)
            
            # Apply transforms
            self.train_data.dataset.transform = train_transform
            self.val_data.dataset.transform = val_test_transform
        
        if stage == 'test' or stage is None:
            self.test_data = ImageFolder(root=os.path.join(self.data_dir, "val"), 
                                         transform=val_test_transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_class_names(self):
        return self.class_names

def get_model_predictions(model, dataloader, device=None):
    """
    Run inference on a model and return accuracy, predictions, targets, and images
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    all_preds = []
    all_targets = []
    all_images = []  
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_images.extend(inputs.cpu().numpy())
    
    accuracy = correct / total
    class_names = dataloader.dataset.dataset.classes if hasattr(dataloader.dataset, 'dataset') else None
    
    return accuracy, np.array(all_preds), np.array(all_targets), np.array(all_images), class_names

def visualize_predictions(images, preds, labels, class_names, num_samples=30, log_wandb=True):
    """Visualize model predictions on test set"""
    
    try:
        # Select random samples
        num_samples = min(num_samples, len(preds))
        indices = np.random.choice(len(preds), num_samples, replace=False)
        
        # Calculate grid dimensions
        cols = 3
        rows = (num_samples + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        fig.tight_layout(pad=3.0)
        
        # Make axes 2D even if there's only one row
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        count = 0
        for i, idx in enumerate(indices):
            if count >= num_samples:
                break
                
            # Get image, prediction and true label
            img = images[idx]
            pred = preds[idx]
            label = labels[idx]
            
            # Convert numpy array to tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
                
            # Unnormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Plot in grid
            row = count // cols
            col = count % cols
            
            axes[row, col].imshow(img.permute(1, 2, 0))
            color = 'green' if pred == label else 'red'
            axes[row, col].set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[label]}", color=color)
            axes[row, col].axis('off')
            
            count += 1
        
        # Hide unused subplots
        for i in range(count, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # Save figure
        plt.savefig('test_predictions.png', bbox_inches='tight')
        
        # Log to wandb if enabled
        if log_wandb:
            try:
                import wandb
                wandb.log({"test_predictions": wandb.Image('test_predictions.png')})
            except Exception as wandb_error:
                print(f"Error logging to wandb: {wandb_error}")
        
        plt.show()
        plt.close(fig)  # Close the figure to free memory
        
        return fig
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()
        return None

def generate_confusion_matrix(preds, labels, class_names):
    """Generate and visualize confusion matrix"""
    try:
        # Check if class_names is None or empty
        if class_names is None or len(class_names) == 0:
            # Get unique classes from labels and create generic class names
            unique_classes = np.unique(np.concatenate([labels, preds]))
            class_names = [f"Class {i}" for i in unique_classes]
            print(f"Warning: Using generic class names {class_names}")

        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save locally
        plt.savefig('confusion_matrix.png')
        plt.show()
        plt.close()  # Close the figure to free memory

        # Calculate per-class metrics
        precision = np.zeros(len(class_names))
        recall = np.zeros(len(class_names))
        f1_score = np.zeros(len(class_names))

        for i in range(len(class_names)):
            # True positives: diagonal elements
            tp = cm[i, i]
            # False positives: sum of column i - true positives
            fp = np.sum(cm[:, i]) - tp
            # False negatives: sum of row i - true positives
            fn = np.sum(cm[i, :]) - tp

            # Calculate metrics (avoiding division by zero)
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        # Print per-class metrics
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_score[i]:.4f}")

    except Exception as e:
        print(f"Error in generate_confusion_matrix: {e}")
        traceback.print_exc()

def train_model(config, data_dir, project_name="PyTorch_CNN", use_wandb=False):
    """Train model with provided configuration"""
    try:
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        print(f"\nStarting training with config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Initialize wandb if requested
        if use_wandb:
            import wandb
            run = wandb.init(project=project_name, config=config)
            wandb_logger = WandbLogger(project=project_name, log_model=True)
        else:
            wandb_logger = None
        
        # Create data module
        data_module = CNNLightningDataModule(
            data_dir=data_dir,
            batch_size=config['batch_size'],
            val_ratio=0.2,
            augment=config['augment'],
            num_workers=4
        )
        
        # Set up data module
        data_module.setup()
        
        # Get number of classes
        num_classes = len(data_module.get_class_names())
        
        # Define filter organization
        if config['filter_organization'] == 'same':
            filters = [config['base_filters']] * 5
        elif config['filter_organization'] == 'double':
            filters = [config['base_filters'] * (2**i) for i in range(5)]
        elif config['filter_organization'] == 'half':
            filters = [config['base_filters'] * (2**(4-i)) for i in range(5)]
        else:  # Custom
            filters = [32, 64, 128, 256, 512]  # Default
        
        # Create model
        model = CNNModel(
            input_channels=3,
            num_classes=num_classes,
            filters=filters,
            filter_size=config['filter_size'],
            activation=config['activation'],
            dense_neurons=config['dense_neurons'],
            dropout_rate=config['dropout_rate'],
            use_batchnorm=config['use_batchnorm'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            optimizer=config['optimizer'],
            lr_decay_factor=config['lr_decay_factor'],
            lr_patience=config['lr_patience']
        )
        
        # Calculate and print model stats
        total_params = model.calculate_num_params()
        print(f"\nModel architecture created with {total_params:,} parameters")
        
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoints/',
            filename='best-model-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            logger=wandb_logger,
            callbacks=callbacks,
            precision='16-mixed',  # Use mixed precision for memory efficiency
            accelerator='auto',
            devices=1,
            gradient_clip_val=1.0
        )
        
        # Train model
        print("\nStarting model training...")
        trainer.fit(model, data_module)
        
        # Test the model
        print("\nEvaluating model on test set...")
        test_results = trainer.test(model, data_module, ckpt_path='best')
        test_loss = test_results[0]['test_loss']
        test_acc = test_results[0]['test_acc']
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        
        # Log test results to wandb if enabled
        if use_wandb:
            wandb.log({
                'test_loss': test_loss,
                'test_accuracy': test_acc
            })
        
        # Get predictions for visualization
        print("\nGenerating predictions for visualization...")
        accuracy, all_preds, all_labels, all_images, class_names = get_model_predictions(
            model, data_module.test_dataloader(), model.device
        )
        
        # Visualize predictions
        print("\nVisualizing predictions...")
        visualize_predictions(all_images, all_preds, all_labels, data_module.get_class_names())
        
        # Generate confusion matrix
        print("\nGenerating confusion matrix...")
        generate_confusion_matrix(all_preds, all_labels, data_module.get_class_names())
        
        # Save the model
        try:
            model_save_path = 'trained_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'classes': data_module.get_class_names(),
                'test_accuracy': test_acc,
                'test_loss': test_loss
            }, model_save_path)
            print(f"\nModel saved to {model_save_path}")
        except Exception as save_error:
            print(f"Warning: Failed to save model: {save_error}")
        
        # Close wandb run if active
        if use_wandb:
            wandb.finish()
        
        return model, test_acc
    
    except Exception as e:
        print(f"Error in train_model: {e}")
        traceback.print_exc()
        
        # Try to close wandb run if it was opened
        if use_wandb:
            try:
                wandb.finish()
            except:
                pass
        
        return None, 0
    
    finally:
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as cuda_error:
                print(f"Error clearing CUDA cache: {cuda_error}")

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning CNN Trainer')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='inaturalist_12K',
                        help='Directory containing the dataset')
    
    # Model configuration
    parser.add_argument('--base_filters', type=int, default=32,
                        help='Base number of filters')
    parser.add_argument('--filter_organization', type=str, default='double', 
                        choices=['same', 'double', 'half'],
                        help='How filters scale across layers')
    parser.add_argument('--filter_size', type=int, default=3,
                        help='Size of convolutional filters')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'silu', 'mish'],
                        help='Activation function')
    parser.add_argument('--dense_neurons', type=int, default=512,
                        help='Number of neurons in dense layer')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--use_batchnorm', type=bool, default=True,
                        help='Whether to use batch normalization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--augment', type=bool, default=True,
                        help='Whether to use data augmentation')
    
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to log to Weights & Biases')
    parser.add_argument('--project_name', type=str, default='PyTorch_CNN',
                        help='Project name for Weights & Biases')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create config dictionary from args
    config = {
        'base_filters': args.base_filters,
        'filter_organization': args.filter_organization,
        'filter_size': args.filter_size,
        'activation': args.activation,
        'dense_neurons': args.dense_neurons,
        'dropout_rate': args.dropout_rate,
        'use_batchnorm': args.use_batchnorm,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'lr_decay_factor': args.lr_decay_factor,
        'lr_patience': args.lr_patience,
        'augment': args.augment
    }
    
    # Print system information
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print("="*50)
    
    # Train the model
    model, accuracy = train_model(
        config=config,
        data_dir=args.data_dir,
        project_name=args.project_name,
        use_wandb=args.use_wandb
    )
    
    print("\n" + "="*50)
    print(f"Training completed with test accuracy: {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
