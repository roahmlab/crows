import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class RobotConfig():
    """
    A class to store and represent the robot's configuration loaded from the dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize RobotConfig with keyword arguments representing configuration parameters.
        """
        self.__dict__.update(kwargs)  # Dynamically add all configuration parameters as attributes

    def __repr__(self):
        """
        Provide a string representation of the robot configuration for debugging or logging.
        """
        return f"{self.__class__.__name__}({self.__dict__})"
        
class ReachabilityDataset(Dataset):
    """
    Dataset for loading robot reachable set data from an HDF5 file.
    """
    def __init__(self, 
                 file_path:str, 
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 load_data: bool = True,
                 **kwargs,
                 ):
        """
        Initialize the dataset by loading data from the HDF5 file.

        Parameters:
        - file_path (str): Path to the HDF5 file containing the reachability data.
        - dtype (torch.dtype): Data type to use for tensors (default: torch.float).
        - device (torch.device): Device for tensor computations (default: CPU).
        - load_data (bool): Whether to load data from the file at initialization.
        """
        super().__init__()

        self.file_path = file_path
        self.dtype, self.device = dtype, device  # Store dtype and device

        if load_data:
            # Load data from HDF5 file
            with h5py.File(self.file_path, 'r') as f:
                # Read different data attributes from the HDF5 file
                self.data_class = f['data_class'][()].astype(str)
                self.qpos = torch.from_numpy(f['joint_positions'][:]).to(dtype=dtype, device=device)
                self.qvel = torch.from_numpy(f['joint_velocities'][:]).to(dtype=dtype, device=device)
                self.ka = torch.from_numpy(f['trajectory_parameters'][:]).to(dtype=dtype, device=device)

                # Handle occupancy or gradient data based on the class type
                if self.data_class == 'occupancy':
                    # Load occupancy centers and radii for reachability sets
                    base_center = f['base_joint_occupancy_center'][:].tolist()
                    base_radius = f['base_joint_occupancy_radius'][:].tolist()
                    self.centers = torch.from_numpy(f['joint_occupancy_centers'][:]).to(dtype=dtype, device=device)
                    self.radii = torch.from_numpy(f['joint_occupancy_radii'][:]).to(dtype=dtype, device=device)
                    self.centers_grad = None  # No gradient data in 'occupancy' class
                else:
                    # Load gradient data for reachability sets
                    base_center = None
                    base_radius = None
                    self.centers = None
                    self.radii = None
                    self.centers_grad = torch.from_numpy(f['joint_occupancy_centers_gradient'][:]).to(dtype=dtype, device=device)

                # Load dataset meta information (number of reachable sets and timesteps)
                self.n_reachable_sets = f['data_size'][:].item()
                self.n_timesteps = f['n_timesteps'][:].item()

                # Store robot configuration from the HDF5 file
                self.__robot_config__ = RobotConfig(
                    name=f['robot_name'][()].astype(str),
                    dof=f['dof'][:].item(),
                    g_ka=f['g_ka'][:].item(),
                    n_timesteps=f['n_timesteps'][:].item(),
                    pos_max=f['pos_max'][:].tolist(),
                    pos_min=f['pos_min'][:].tolist(),
                    vel_max=f['vel_max'][:].tolist(),
                    base_center=base_center,
                    base_radius=base_radius
                )
        else:
            # If not loading data, manually set attributes from kwargs
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])

    @property
    def robot_config(self):
        """
        Get the robot configuration.
        """
        return self.__robot_config__

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        The total is the number of reachable sets multiplied by the number of timesteps.
        """
        return self.n_reachable_sets * self.n_timesteps

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset by index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - A tuple: (joint positions, joint velocities, trajectory parameters, timestep), 
                   (occupancy centers and radii) or (gradient data).
        """
        # Calculate the reachability set and timestep indices
        idx_r, idx_t = idx // self.n_timesteps, idx % self.n_timesteps

        # Get joint positions, velocities, and trajectory parameters for the set
        qpos = self.qpos[idx_r]
        qvel = self.qvel[idx_r]
        ka = self.ka[idx_r]

        # Create a tensor representing the timestep
        timesteps = torch.tensor([idx_t], dtype=self.dtype, device=self.device)

        if self.data_class == 'occupancy':
            # Return centers and radii for 'occupancy' class data
            centers = self.centers[idx_r, idx_t]
            radii = self.radii[idx_r, idx_t]
            return (qpos, qvel, ka, timesteps), (centers, radii)
        else:
            # Return gradient data for other classes
            centers_grad = self.centers_grad[idx_r, idx_t]
            return (qpos, qvel, ka, timesteps), centers_grad

    def split(self, rs_lengths):
        """
        Split the dataset into multiple parts based on the provided sizes.

        Parameters:
        - rs_lengths (list): List of sizes to split the dataset into.

        Returns:
        - A list of ReachabilityDataset instances representing the splits.
        """
        # List of main attributes to split across datasets
        main_attr_list = ['data_class', '__robot_config__', 'qpos', 'qvel', 'ka', 
                          'centers', 'radii', 'centers_grad', 'n_timesteps']

        kwargs = {}
        # Split tensor attributes by reachability set length
        for key in main_attr_list:
            item = getattr(self, key)
            kwargs[key] = torch.split(item, rs_lengths) if isinstance(item, torch.Tensor) else item

        # Create a list of datasets, each with a portion of the original data
        splited_datasets = []
        for i in range(len(rs_lengths)):
            kwargs_tmp = {key: kwargs[key][i] if isinstance(kwargs[key], tuple) else kwargs[key] for key in kwargs}
            kwargs_tmp['n_reachable_sets'] = rs_lengths[i]
            dataset = ReachabilityDataset(self.file_path, self.dtype, self.device, False, **kwargs_tmp)
            splited_datasets.append(dataset)
        return splited_datasets

def set_dataloaders(path_to_dataset, batch_size, training_portion=0.8, validation_portion=0.1, train_eval_batch_ratio = 10, num_workers = 0, device=torch.device('cpu')):
    """
    Splits a dataset into training, validation, and calibration sets, and returns DataLoaders for each.

    Args:
        path_to_dataset (str): Path to the dataset (HDF5 file).
        batch_size (int): Batch size for the training DataLoader.
        training_portion (float): Proportion of the dataset to allocate to training. Default is 0.8.
        validation_portion (float): Proportion of the dataset to allocate to validation. Default is 0.1.
        train_eval_batch_ratio (int): Ratio of training to validation/calibration batch sizes. Default is 10.
        num_workers (int): Number of workers for DataLoader. Default is 0.
        device (torch.device or str): Device to return data (CPU or GPU). Default is CPU.

    Returns:
        dict: A dictionary containing DataLoaders for training, validation, and calibration (if applicable).
        RobotConfig: Robot configuration from the dataset.
    """
    # Ensure the sum of training and validation portions does not exceed 1.0
    assert training_portion + validation_portion <= 1.0, 'The sum of training and validation portions cannot exceed 1.'

    # Calculate the batch size for validation/calibration based on the train_eval_batch_ratio
    eval_batch_size = int(train_eval_batch_ratio * batch_size)

    # Load the full dataset
    dataset = ReachabilityDataset(path_to_dataset)

    # If device is provided as a string, convert it to a torch.device object
    if isinstance(device, str):
        device = torch.device(device)

    # Pin memory for faster transfer to GPU, if using CUDA
    pin_memory = device.type == 'cuda'
    print(f'Pin Memory: {pin_memory}')  # Useful for debugging memory usage

    # Calculate sizes for training, validation, and calibration sets
    training_rs_size = int(dataset.n_reachable_sets * training_portion)
    validation_rs_size = int(dataset.n_reachable_sets * validation_portion)
    calibration_rs_size = dataset.n_reachable_sets - training_rs_size - validation_rs_size
    use_calibration = calibration_rs_size > 0  # Only use calibration if there's data left

    # Split the dataset into training, validation, and calibration sets
    if use_calibration:
        split_rs_size = [training_rs_size, validation_rs_size, calibration_rs_size]
        training_set, validation_set, calibration_set = dataset.split(split_rs_size)
    else:
        split_rs_size = [training_rs_size, validation_rs_size]
        training_set, validation_set = dataset.split(split_rs_size)

    # Create DataLoaders for training, validation, and (if applicable) calibration sets
    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    validation_dataloader = DataLoader(validation_set, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Store DataLoaders in a dictionary for easy access
    dataloaders = {
        'train': training_dataloader,
        'validation': validation_dataloader,
    }

    # Optionally, add the calibration DataLoader if the calibration set exists
    if use_calibration:
        calibration_dataloader = DataLoader(calibration_set, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=pin_memory)
        dataloaders['calibration'] = calibration_dataloader

    # Print the sizes of the training, validation, and calibration sets
    print_txt = f"Data size: training {len(training_set)}, validation {len(validation_set)}"
    if use_calibration:
        print_txt += f", calibration {len(calibration_set)}"
    print(print_txt)

    # Return the DataLoader dictionary and the robot configuration
    return dataloaders, dataset.robot_config


# Example usage
if __name__ == "__main__":
    import os
    base_dirname = os.path.dirname(__file__)
    save_dirname = os.path.join(base_dirname,'so_dataset')
    print(base_dirname)
    print(save_dirname)
    file_name = 'kinova_gen3_so.hdf5'

    batch_size = 4  # Number of samples per batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Load the dataset using the DataLoader
    dataset = ReachabilityDataset(os.path.join(save_dirname, file_name))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the dataset
    for batch_idx, ((qpos, qvel, ka, timesteps), (centers, radii)) in enumerate(dataloader):

        # Example: Print the first batch's positions
        if batch_idx == 0:
            print("qpos: ", qpos)
            print("qvel: ", qvel)
            print("ka: ", ka)
            print("timesteps: ", timesteps)
            print("centers: ", centers)
            print("radii: ", radii)