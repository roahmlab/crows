import os 
import numpy as np
import torch
import wandb 
from tqdm import tqdm

# Convert an integer into its ordinal representation (e.g., 1 -> 1st, 2 -> 2nd)
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def save_model(model, model_save_dir, train_grad, epoch=None):
    """
    Save the model's state dictionary to a file.

    Args:
        model: The model to be saved.
        model_save_dir (str): Directory where the model will be saved.
        train_grad (bool): If True, saves the model with grad-related naming.
        epoch (int or None): If provided, saves the model with an epoch number in the file name.
    """
    # Get the device of the model's parameters
    device = next(model.parameters()).device
    
    # Generate the model's file name based on whether training gradients and epoch are provided
    model_suffix = '_grad' if train_grad else ''
    model_file_name = 'model'+model_suffix+'.pth' if epoch is None else 'model'+model_suffix+f'_epoch{epoch+1}.pth'
    path_to_latest_file = os.path.join(model_save_dir, model_file_name)
    
    # Save the model's state dictionary to CPU, excluding unnecessary keys (e.g., buffers)
    model_state_dict = model.cpu().state_dict()
    for k in ['pos_max', 'pos_min', 'vel_max', 'quantile_for_each_joint', 'alpha', 'base_center', 'base_radius']:
        if k in model_state_dict:
            del model_state_dict[k]

    # Save the model to the specified file
    torch.save(model_state_dict, path_to_latest_file)

    # Optionally log the model using Weights and Biases (WandB) if an active run exists
    if wandb.run is not None:
        wandb.save(path_to_latest_file, base_path=model_save_dir)
    
    # Move the model back to its original device
    model.to(device)


def train_CROWS(model, dataloaders, optimizer, criterion, num_epochs, validation_freq=5, model_save_dir=None, model_save_freq=0, train_grad=False):
    """
    Train and validate the CROWS model.

    Args:
        model: The PyTorch model to be trained.
        dataloaders (dict): Dictionary containing 'train', 'validation', and 'calibration' DataLoaders.
        optimizer: The optimizer used for training.
        criterion: The loss function used for training.
        num_epochs (int): Number of epochs to train the model.
        validation_freq (int): Frequency (in epochs) to run validation.
        model_save_dir (str or None): Directory to save the model.
        model_save_freq (int): Frequency (in epochs) to save the model.
        train_grad (bool): If True, train the model for gradient prediction.
    """
    # Create directory for saving models if it doesn't exist
    if model_save_dir is not None and not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    
    # Get the device from the model parameters
    device = next(model.parameters()).device
    dof = model.dof  # Degrees of freedom (e.g., number of joints)

    # Loop over the number of epochs
    for epoch in tqdm(range(num_epochs)):
        
        # Determine if the current epoch should include validation
        phases = ['train', 'validation'] if epoch % validation_freq == validation_freq-1 or epoch == num_epochs-1 else ['train']

        # Iterate over each phase (train or validation)
        for phase in phases: 
            # Set model to training or evaluation mode based on the phase
            model.train() if phase == 'train' else model.eval()

            # Get the corresponding dataloader for the current phase
            dataloader = dataloaders[phase]

            # Initialize logging dictionary and loss tracking
            log = {}
            epoch_loss = 0.0
            
            if train_grad:
                # If training gradients, initialize gradient error tracking
                all_grad_err = torch.zeros(len(dataloader.dataset), device=device)
                batch_count = 0
                n_inf = 0  # To count the number of infinite gradient errors
            else:
                # Buffers to store mean and max radii statistics if not training gradients
                mean_rad_buff = 0.0
                max_rad_buff = torch.zeros(dof, device=device)

            # Iterate through each batch of data
            for inputs, outputs in dataloader:
                # Move inputs to the appropriate device (GPU/CPU)
                inputs = [i.to(device) for i in inputs]
                
                if train_grad:
                    # For gradient prediction, scale the outputs accordingly
                    centers_jac_flat = outputs.to(device) / model.scale
                else:
                    # For non-gradient training, split the outputs into centers and radii
                    centers, radii = [o.to(device) for o in outputs] 

                # Zero the parameter gradients before the forward pass
                optimizer.zero_grad()

                # Forward pass: differentiate between training and evaluation
                if train_grad:
                    if phase == 'train':
                        pred_centers_jac_flat = model(*inputs)
                    else:
                        with torch.no_grad():
                            pred_centers_jac_flat = model(*inputs)
                    # Compute the loss for gradient prediction
                    loss = criterion(centers_jac_flat, pred_centers_jac_flat)

                else:
                    if phase == 'train':
                        pred_centers, pred_radii = model(*inputs)
                    else:
                        with torch.no_grad():
                            pred_centers, pred_radii = model(*inputs)
                    # Compute the loss for centers and radii prediction
                    loss = criterion(centers, pred_centers) + criterion(radii, pred_radii)
                    
                # Backward pass and optimize if training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Post-processing and logging metrics
                with torch.no_grad():
                    if train_grad:
                        # Compute the gradient error based on the norm difference
                        grad_err = torch.norm((centers_jac_flat - pred_centers_jac_flat), dim=[-2, -1]) / torch.norm(centers_jac_flat, dim=[-2, -1])
                        batch_size = grad_err.size(0)

                        # Store the gradient error, handling NaN and infinite values
                        all_grad_err[batch_count:batch_count+batch_size] = grad_err.nan_to_num(posinf=0, neginf=0)
                        n_inf += grad_err.isinf().sum().item()  # Count infinite errors
                        batch_count += batch_size
                    else:
                        # Compute radial buffer (distance between predicted and actual centers and radii)
                        rad_buff = torch.clamp(torch.norm(pred_centers - centers, dim=-1) + pred_radii - radii, min=0)

                        # Accumulate metrics
                        epoch_loss += loss.item()
                        mean_rad_buff += rad_buff.sum(dim=0)
                        max_rad_buff = torch.maximum(max_rad_buff, rad_buff.max(dim=0).values) 

            # Calculate average loss over the number of samples
            epoch_loss /= len(dataloader)
            
            if not train_grad:
                # Normalize mean radius buffer over the dataset size
                mean_rad_buff /= len(dataloader.dataset)

            # Log metrics to wandb if run is active
            if wandb.run is not None:
                if train_grad:
                    # Log metrics related to gradient error and loss
                    log[f"{phase}/loss"] = epoch_loss 
                    log[f"{phase}/med_grad_err"] = np.median(all_grad_err.cpu().numpy()).item()
                    log[f"{phase}/max_grad_err"] = all_grad_err.max().item()
                else:
                    # Log joint-specific max radial buffer values
                    for i in range(dof):
                        log[f"{phase}_joint_stat/{ordinal(i+1)}_joint_max_rad_buff"] = max_rad_buff[i].item()
                    log[f"{phase}/loss"] = epoch_loss 
                    log[f"{phase}/mean_rad_buff"] = mean_rad_buff.mean().item()
                    log[f"{phase}/max_rad_buff"] = max_rad_buff.max().item()

                wandb.log(log, step=epoch+1)

        # Save model at specified frequency (except for final epoch)
        if model_save_freq != 0 and epoch % model_save_freq == model_save_freq-1 and epoch != num_epochs - 1 and model_save_dir is not None:
            save_model(model, model_save_dir, train_grad, epoch)

    print("Training is completed.")

    # Save the final model at the end of training
    if model_save_dir is not None:
        save_model(model, model_save_dir, train_grad)
        print("The latest trained model is saved.")