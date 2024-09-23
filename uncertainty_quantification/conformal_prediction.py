import torch 
import numpy as np
import wandb 
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import h5py

# Convert an integer into its ordinal representation (e.g., 1 -> 1st, 2 -> 2nd)
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


def split_conformal_prediction(model, dataloader, epsilon_hat, nonconformity_save_dir = None, train_grad=False):
    """
    Perform conformal prediction for a given model and dataset.

    Args:
        model: The trained PyTorch model for predictions.
        dataloader: DataLoader providing batches of data for evaluation.
        epsilon_hat: Significance levels for conformal prediction.
        nonconformity_save_dir: Directory for saving nonconformity scroes (if provided).
        
    Returns:
        quantile: The quantile value(s) used for conformal prediction.
    """
    # Create model save directory if it doesn't exist
    if nonconformity_save_dir is not None and not os.path.exists(nonconformity_save_dir):
        os.mkdir(nonconformity_save_dir)

    # Determine device and degrees of freedom (dof) for the model
    device = next(model.parameters()).device
    dof = model.dof

    # Convert epsilon_hat to a numpy array if necessary
    if isinstance(epsilon_hat,list) or isinstance(epsilon_hat,tuple):
        epsilon_hat = np.array(epsilon_hat)
    elif isinstance(epsilon_hat, torch.Tensor):
        epsilon_hat = epsilon_hat.cpu().numpy()
    assert isinstance(epsilon_hat, np.ndarray), f"epsilon_hat has to be numpy.ndarray, torch.Tensor, list, or tuple, instead of {type(epsilon_hat)}."

    # Initialize tensors for nonconformity scores
    nonconformity_scores = torch.zeros(len(dataloader.dataset), device=device)
    nonconformity_for_each_joint = torch.zeros((len(dataloader.dataset), dof), device=device)
    # Set model to evaluation mode
    model.eval()

    batch_count = 0
    with torch.no_grad():  # Disable gradient calculations
        for inputs, outputs in tqdm(dataloader):
            inputs = [i.to(device) for i in inputs]

            if train_grad:
                centers_jac_flat = outputs.to(device)/model.scale
            else:
                centers, radii = [o.to(device) for o in outputs]                

                
            batch_size = inputs[0].size(0)

            # Forward pass: Get model predictions
            if train_grad:
                pred_centers_jac_flat = model(*inputs)
                nonconformity = torch.norm((centers_jac_flat-pred_centers_jac_flat), dim=[-2,-1])/torch.norm(centers_jac_flat, dim=[-2,-1])
                nonconformity_scores[batch_count:batch_count+batch_size] = nonconformity.nan_to_num(posinf=0,neginf=0)
            else:
                pred_centers, pred_radii = model(*inputs) # Predicted centers and radii
                nonconformity = torch.clamp(torch.norm(pred_centers - centers, dim=-1) + pred_radii - radii, min=0)
                nonconformity_for_each_joint[batch_count:batch_count+batch_size] = nonconformity
                nonconformity_scores[batch_count:batch_count+batch_size] = nonconformity.max(dim=-1).values

            batch_count += batch_size

    # Compute the quantile based on nonconformity scores and epsilon_hat
    nonconformity_scores = nonconformity_scores.cpu().numpy()
    nonconformity_for_each_joint = nonconformity_for_each_joint.cpu().numpy()
    
    quantile = np.array([np.nanquantile(nonconformity_scores, 1.-e, method='higher') for e in epsilon_hat])


    if not train_grad:
        quantile_for_each_joint = np.empty((len(epsilon_hat), dof))
        for i in range(dof):
            quantile_for_each_joint[:,i] = np.array([np.nanquantile(nonconformity_for_each_joint[:,i], 1.-e, method='higher') for e in epsilon_hat])

    # Save scores and generate figures
    model_suffix = '_grad' if train_grad else ''
    save_scores(nonconformity_scores if train_grad else nonconformity_for_each_joint, nonconformity_save_dir, model_suffix)
    draw_fig(nonconformity_scores, quantile, epsilon_hat, nonconformity_save_dir, model_suffix)

    if not train_grad:
        for j in range(dof):
            draw_fig(nonconformity_for_each_joint[:,j], quantile, epsilon_hat, nonconformity_save_dir, model_suffix, j+1)

    print("Conformal prediction is completed.")

    return quantile if train_grad else quantile_for_each_joint


def draw_fig(nonconformity_scores, quantile, epsilon_hat, nonconformity_save_dir, model_suffix, j=0):
    """Draw and save a histogram of nonconformity scores."""
    counts, _, _ = plt.hist(nonconformity_scores, bins=50, color='skyblue')
    
    # Plot quantiles
    for i, (q, e) in enumerate(zip(quantile, epsilon_hat)):
        h = max(counts) * (0.95 - 0.11 * i)
        plt.vlines(q.item(), 0, h, color='indianred', linestyle='--')
        percent_clean = format((1 - e.item()) * 100, 'f').rstrip('0').rstrip('.') + ' %'
        plt.text(q.item(), h + 0.54, percent_clean, color='darkred', ha='center')

    title = 'Nonconformity Scores ' + ('(Grad.)' if model_suffix == '_grad' else '(Occ.)') + ('' if j == 0 else f' for {ordinal(j)} Joint')
    plt.title(title, color='black')    
    plt.xlabel('Scores (Relative Errors)' if model_suffix == '_grad' else 'Scores, $\max(||c-\hat{c}||+r-\hat{r}, 0)$', color='black')
    plt.ylabel('Frequency', color='black')
    
    plt.savefig(os.path.join(nonconformity_save_dir, f'nonconformity_hist{model_suffix}' + ('' if j == 0 else f'_joint{j}') + '.png'), dpi=200, format='png')
    plt.close()

    if wandb.run is not None:
        wandb.log({"nonconformity_scores/figure": wandb.Image(os.path.join(nonconformity_save_dir, f'nonconformity_hist{model_suffix}' + ('' if j == 0 else f'_joint{j}') + '.png'), caption=("" if j == 0 else f"{ordinal(j)} Joint ") + "Nonconformity Scores from Calibration Dataset")})


def save_scores(nonconformity_scores, nonconformity_save_dir, model_suffix):
    """Save nonconformity scores to an HDF5 file."""
    file_path = os.path.join(nonconformity_save_dir, f"nonconformity_scores{model_suffix}.hdf5")
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('nonconformity_scores', data=nonconformity_scores)

    if wandb.run is not None:
        wandb.save(file_path, base_path=nonconformity_save_dir)

