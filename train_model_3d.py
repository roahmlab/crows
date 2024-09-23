import torch
import numpy as np
import random
import argparse
import wandb 
from datetime import datetime
import json 

from data_processing.data_loader import set_dataloaders
from training.models import MLP_Joint_Occ, MLP_Grad
from training.train import train_CROWS
from uncertainty_quantification.conformal_prediction import split_conformal_prediction

import os 

def set_random_seed(seed):
    """Set random seed for reproducibility."""    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    '''
    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    '''
    
def read_params():
    """Parse command-line arguments to configure the training process.
    
    Provides flexibility for users to set hyperparameters, device options, 
    architecture choices, and other settings via the command line.
    """
    parser = argparse.ArgumentParser(description="CROWS Training")

    # General settings
    parser.add_argument('--wandb', type=str, default='')  # Weights & Biases project name
    parser.add_argument('--not_save_file_in_wandb', action='store_true')  # Option to not save model in wandb
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else -1, choices=range(-1, torch.cuda.device_count()))  # Device selection (GPU/CPU)
    parser.add_argument('--seed', type=int, default=0)  # Random seed for reproducibility

    # Hyperparameters for training
    parser.add_argument('--batch_size', type=int, default=1000)  # Batch size for DataLoader
    parser.add_argument('--num_epochs', type=int, default=30)  # Number of training epochs
    parser.add_argument("--beta1", type=float, default=0.9)  # Beta1 for AdamW optimizer
    parser.add_argument("--beta2", type=float, default=0.999)  # Beta2 for AdamW optimizer
    parser.add_argument('--weight_decay', type=float, default=0.0001)  # Weight decay for AdamW optimizer
    parser.add_argument("--lr", type=float, default=0.0003)  # Learning rate

    # Network architecture options
    parser.add_argument('--radius_net_arch', nargs='+', type=int, default=[1024]*3)  # Radius net layers
    parser.add_argument('--center_net_arch', nargs='+', type=int, default=[1024]*9)  # Center net layers
    parser.add_argument('--center_jac_net_arch', nargs='+', type=int, default=[1024]*12)  # Center jacobian net layers
    parser.add_argument('--radius_activation_fn', type=str, choices=['relu', 'gelu'], default='relu')  # Activation function for radius net
    parser.add_argument('--center_activation_fn', type=str, choices=['relu', 'gelu'], default='gelu')  # Activation function for center net
    parser.add_argument('--center_jac_activation_fn', type=str, choices=['relu', 'gelu'], default='gelu')  # Activation for center jacobian net

    # Data settings
    parser.add_argument('--training_portion', type=float, default=0.8)  # Training dataset portion
    parser.add_argument('--validation_portion', type=float, default=0.1)  # Validation dataset portion
    parser.add_argument('--train_eval_batch_ratio', type=int, default=10)  # Training vs evaluation batch size ratio
    parser.add_argument('--num_workers', type=int, default=0)  # Number of workers for DataLoader
    
    # Training settings
    parser.add_argument('--validation_freq', type=int, default=5)  # Frequency of validation during training
    parser.add_argument('--model_save_freq', type=int, default=0)  # Frequency of saving models
    parser.add_argument('--train_grad', action='store_true')  # Option to train the Jacobian (gradient) model

    return parser.parse_args()

def init_wandb(params):
    """Initialize Weights & Biases (W&B) logging system with the training parameters.
    
    W&B is used to track experiments, hyperparameters, and training metrics. 
    It automatically logs metrics and saves model artifacts.
    """
    config = vars(params).copy()  # Convert argparse namespace to dictionary
    del config['wandb']  # Remove the W&B project name from config
    if params.train_grad:
        # Exclude non-relevant architecture settings if training Jacobians
        del config['center_net_arch'], config['center_activation_fn'], config['radius_net_arch'], config['radius_activation_fn']   
    else:
        # Exclude Jacobian-related settings if not training gradients
        del config['center_jac_net_arch'], config['center_jac_activation_fn']
        
    # Define run tags for W&B experiment tracking
    tags = [
        'kinova_gen3',
        f'{params.batch_size}-batch_size',
        f'{params.num_epochs}-num_epochs',
        f'{params.lr:.6f}-lr',
    ]

    run = wandb.init(project=params.wandb, config=config, tags=tags)  # Initialize W&B run
    run_id = run.name.split('-')[-1]  # Extract run ID from W&B run name
    run.name = f'{params.batch_size}b_' + ('grad_' if params.train_grad else '') + run_id  # Name the run based on batch size and gradient training
    wandb.config.update({'trial_name': params.wandb + run_id + datetime.now().strftime("-%d-%m-%Y-%H-%M-%S")})  # Update W&B with trial details

    return run._settings['files_dir']  # Return directory for W&B files

def save_model_config(model_config, model_save_dir, train_grad=False):
    """Save the model configuration as a JSON file for future reference.
    
    Saves model architecture, training settings, and other configurations to a file.
    Also, it saves the configuration to W&B if enabled.
    """
    if model_save_dir is not None and not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)  # Create directory if it doesn't exist
    
    # Save model configuration to a JSON file
    with open(os.path.join(model_save_dir, 'model_config' + ('_grad' if train_grad else '') + '.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save the configuration file to W&B if the W&B run is active
    if wandb.run is not None:
        wandb.save(os.path.join(model_save_dir, 'model_config' + ('_grad' if train_grad else '') + '.json'), base_path=model_save_dir)



if __name__ == '__main__':
    # List of epsilon values for conformal prediction -> 0: 99.999%, 1: 99.99%, 2: 99.9%, 3: 99% 4: 90% 5:80%
    epsilon_hat = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2] 

    # Ensure certain CUDA features are disabled for consistency
    torch.backends.cuda.matmul.allow_tf32 = False

    # Read the command-line parameters  
    params = read_params()
    # Initialize W&B if the W&B project name is provided
    if params.wandb:
        wandb_files_dir = init_wandb(params)

    # Set random seed for reproducibility
    set_random_seed(params.seed)
    
    # Set the device (GPU or CPU) for model training
    device = torch.device(f'cuda:{params.device}' if params.device >= 0 else 'cpu')
    
    # Define base directory and paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_dir = os.path.join(base_dir, 'trained_models')
    nonconformity_save_dir = os.path.join(model_save_dir, 'nonconformity')
    path_to_dataset = os.path.join(base_dir, 'data_processing/so_dataset/kinova_gen3_so' + ('_grad' if params.train_grad else '') + '.hdf5')
    
    # Load dataset and dataloaders (train, validation, and calibration sets)
    dataloaders, robot_config = set_dataloaders(path_to_dataset, batch_size=params.batch_size, training_portion=params.training_portion, validation_portion=params.validation_portion, train_eval_batch_ratio=params.train_eval_batch_ratio, num_workers=params.num_workers, device=device)

    # Model selection: MLP_Grad if training gradients, otherwise MLP_Joint_Occ
    if params.train_grad:
        kwargs_for_model = {'dimension': 3, 'dof': robot_config.dof, 'center_jac_net_arch': params.center_jac_net_arch, 'center_jac_activation_fn': params.center_jac_activation_fn, 'scale': 1e-2, 'exclude_last_el': True}
        model = MLP_Grad(**kwargs_for_model)
    else:
        kwargs_for_model = {'dimension': 3, 'dof': robot_config.dof, 'center_net_arch': params.center_net_arch, 'radius_net_arch': params.radius_net_arch, 'center_activation_fn': params.center_activation_fn, 'radius_activation_fn': params.radius_activation_fn}
        model = MLP_Joint_Occ(**kwargs_for_model)

    # Set up robot parameters for the model based on the loaded robot configuration
    kwargs_for_robot_params = {"g_ka": robot_config.g_ka, "pos_max": robot_config.pos_max, "pos_min": robot_config.pos_min, "vel_max": robot_config.vel_max, "n_timesteps": robot_config.n_timesteps}
    model._setup_robot_params(**kwargs_for_robot_params)

    # Define model configuration for saving later
    model_config = {"model":'MLP_Grad' if params.train_grad else 'MLP_Joint_Occ', "kwargs_for_model": kwargs_for_model, "kwargs_for_robot_params": kwargs_for_robot_params, "kwargs_for_base_joint_occ": {"base_center": robot_config.base_center, "base_radius": robot_config.base_radius}, "kwargs_for_dataloader":{'training_portion':params.training_portion, 'validation_portion': params.validation_portion}}

    # Initialize the AdamW optimizer with specified learning rate, betas, and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.weight_decay)  
    # Move model to the selected device (GPU or CPU)      
    model.to(device)
    # Save model config before training (just in case you need this before everything is done)
    save_model_config(model_config, model_save_dir, params.train_grad)    

    # Train the model using the CROWS framework
    train_CROWS(model, dataloaders, optimizer, torch.nn.MSELoss(), params.num_epochs, validation_freq=params.validation_freq, model_save_dir = model_save_dir, model_save_freq=params.model_save_freq, train_grad=params.train_grad)

    # Perform conformal prediction on the calibration set to quantify uncertainty
    quantile = split_conformal_prediction(model, dataloaders['calibration'], epsilon_hat, nonconformity_save_dir = nonconformity_save_dir, train_grad=params.train_grad)
    
    # If not training gradients, save additional conformal prediction results
    if not params.train_grad:
        model_config['kwargs_for_conformal_prediction'] = {'epsilon_hat': epsilon_hat, 'quantile_for_each_joint': quantile.tolist()}
        # Update and save the model configuration after conformal prediction
        save_model_config(model_config, model_save_dir)

    # Finalize the W&B run if W&B logging is enabled
    if wandb.run is not None:
        wandb.finish()