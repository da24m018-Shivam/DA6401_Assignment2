# DA6401_Assignment2

# PartA :CNN Image Classification with PyTorch Lightning

## Features

- **Modular CNN Architecture**: Customizable depth, filter organization, and activation functions
- **PyTorch Lightning Integration**: Well-structured code with minimal boilerplate
- **Data Augmentation**: Configurable image transformations to improve model generalization
- **Advanced Training Features**:
  - Early stopping
  - Learning rate scheduling
  - Mixed precision training
  - Model checkpointing
- **Comprehensive Visualization**:
  - Confusion matrix
  - Sample predictions
  - Per-class metrics (precision, recall, F1-score)
- **Weights & Biases Integration**: Optional experiment tracking and visualizations
- **Balanced Dataset Handling**: Maintains class distribution in train/validation splits

## Usage

### Basic Usage

```bash
python part1.py --data_dir path/to/dataset
```

### Advanced Configuration

```bash
python part1.py --data_dir path/to/dataset \
                --base_filters 64 \
                --filter_organization double \
                --activation gelu \
                --batch_size 64 \
                --epochs 20 \
                --learning_rate 0.0005 \
                --optimizer adam \
                --use_wandb
```

### Command Line Arguments

#### Model Configuration
- `--base_filters`: Base number of filters (default: 32)
- `--filter_organization`: How filters scale across layers (choices: same, double, half; default: double)
- `--filter_size`: Size of convolutional filters (default: 3)
- `--activation`: Activation function (choices: relu, gelu, silu, mish; default: relu)
- `--dense_neurons`: Number of neurons in dense layer (default: 512)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--use_batchnorm`: Whether to use batch normalization (default: True)

#### Training Parameters
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 10)
- `--learning_rate`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer (choices: adam, sgd; default: adam)
- `--weight_decay`: Weight decay for optimizer (default: 0.0001)
- `--lr_decay_factor`: Learning rate decay factor (default: 0.1)
- `--lr_patience`: Patience for learning rate scheduler (default: 3)
- `--augment`: Whether to use data augmentation (default: True)

#### Weights & Biases Integration
- `--use_wandb`: Whether to log to Weights & Biases
- `--project_name`: Project name for Weights & Biases (default: PyTorch_CNN)

## Model Architecture

The CNN model consists of:
1. Configurable convolutional blocks with optional batch normalization
2. MaxPooling layers for downsampling
3. Dropout layers for regularization
4. Dense layers for classification output

The model is implemented using PyTorch Lightning for clean and organized code structure.

## Performance Visualization

After training, the model automatically:
1. Generates a confusion matrix to visualize class-wise performance
2. Displays sample predictions on test images

# Part B: Fine-Tuning InceptionV3 for Naturalist Dataset

This repository contains code for fine-tuning the InceptionV3 model on a naturalist dataset, allowing for different freezing strategies and optimization configurations.

### Advanced Fine-Tuning with Freezing Strategies

The second part of this project focuses on more sophisticated fine-tuning approaches with different layer freezing strategies and detailed visualization.

### Features

- Customizable freezing strategies for transfer learning:
  - No freezing (train all layers)
  - Freeze all layers except classification layer
  - Freeze only convolutional layers
  - Freeze only fully connected layers
  - Freeze first N layers
  - Freeze last N layers
- Multiple optimizer options (SGD, Adam, RMSprop, AdaGrad)
- Learning rate scheduling
- Comprehensive metrics and visualization:
  - Training/validation curves
  - Confusion matrices
  - Model prediction visualization
  - Per-class metrics
- Integration with Weights & Biases (W&B) for experiment tracking



### Usage

The script can be run from the command line with various parameters:

```bash
python part2.py --data_path /path/to/dataset --batch_size 32 --epochs 10 --freeze_strategy allconv
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to dataset root directory | (Required) |
| `--batch_size` | Batch size for training | 32 |
| `--epochs` | Number of training epochs | 10 |
| `--image_size` | Image size for resizing | 299 |
| `--optimizer` | Optimizer to use (sgd, adam, rmsprop, adagrad) | adam |
| `--learning_rate` | Learning rate | 0.001 |
| `--momentum` | Momentum for SGD optimizer | 0.9 |
| `--weight_decay` | Weight decay for regularization | 1e-4 |
| `--use_scheduler` | Use learning rate scheduler | False |
| `--freeze_strategy` | Strategy for freezing layers (No, all, allconv, allfc, first, last) | No |
| `--freeze_layers` | Number of layers to freeze (for first/last strategies) | 0 |
| `--project_name` | W&B project name | Fine_Tune_Naturalist |
| `--run_name` | W&B run name | InceptionV3_Fine_Tune |
| `--output_dir` | Directory to save outputs | output |
| `--seed` | Random seed | 42 |

### Freezing Strategies

- `No`: No freezing, train all parameters
- `all`: Freeze all layers except the final classification layer
- `allconv`: Freeze all convolutional layers
- `allfc`: Freeze all fully connected layers except the last one
- `first`: Freeze first N layers (specified by `--freeze_layers`)
- `last`: Freeze last N layers (specified by `--freeze_layers`)



### Outputs

The script generates several outputs in the specified output directory:

- Best model checkpoint (`best_model_epoch_X.pt`)
- Final model (`final_model.pt`)
- Training history plot
- Confusion matrix visualization
- Model prediction visualization

All results are also logged to Weights & Biases for experiment tracking.

### Example

```bash
python part2.py --data_path ./naturalist_data --batch_size 64 --epochs 15 --optimizer adam --learning_rate 0.0005 --freeze_strategy allconv --use_scheduler --output_dir results/allconv_run
```

This will fine-tune InceptionV3 on the naturalist dataset, freezing all convolutional layers and using Adam optimizer with learning rate scheduling.


