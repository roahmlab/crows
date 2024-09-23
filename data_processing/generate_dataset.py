import torch
import zonopy as zp
import zonopyrobots as zpr 
import h5py
import argparse
from tqdm import tqdm
import numpy as np
import sys, os

# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from forward_occupancy.JRS import OfflineJRS
from forward_occupancy.SO import sphere_occupancy


class ReachableSetGenerator():
    """
    Class to generate reachable sets (spherical occupancies) for a robot's joints
    using zonotopic approximations and random sampling of joint states and trajectory parameters.
    """
    def __init__(self, 
                 robot: zpr.ZonoArmRobot, 
                 robot_name: str,
                 joint_radius_override: dict = {},
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 zono_order: int = 2,
                 ):
        """
        Initialize the Reachable Set Generator for a robot.

        Parameters:
        - robot (zpr.ZonoArmRobot): The robot model for which reachable sets are generated.
        - robot_name (str): Name of the robot model.
        - joint_radius_override (dict): A dictionary to override joint radii if necessary.
        - dtype (torch.dtype): Data type to use for tensors (default: torch.float).
        - device (torch.device): Device on which to perform calculations (default: CPU).
        - zono_order (int): The order of zonotope approximation used in calculations.
        """
        
        self.robot_name = robot_name
        self.dimension = 3  # The workspace dimension (3D space)
        self.dtype, self.device = dtype, device

       # Load the precomputed Joint Reachable Set (JRS) model
        self.JRS = OfflineJRS(dtype=self.dtype, device=self.device)
        self.g_ka = self.JRS.g_ka
        self.n_timesteps = self.JRS.jrs_tensor.shape[1] # Number of time intervals in the JRS model

        self.robot = robot
        self._setup_robot(robot)  # Configure the robot's properties
        
        self.zono_order = zono_order
        self.joint_radius_override = joint_radius_override


    def _setup_robot(self, robot: zpr.ZonoArmRobot):
        """
        Setup robot-specific parameters including degrees of freedom (DOF),
        joint axes, position limits, velocity limits, and continuous joint properties.
        
        Parameters:
        - robot (zpr.ZonoArmRobot): The robot model being configured.
        """
        self.dof = robot.dof  # Number of degrees of freedom (DOF) for the robot
        self.joint_axis = robot.joint_axis.to(dtype=self.dtype, device=self.device)  # Joint axes
        
        # Handling position limits, replacing NaN with ±π where necessary
        pos_lim = robot.pos_lim.nan_to_num(posinf=torch.pi, neginf=-torch.pi).to(dtype=self.dtype, device=self.device)
        self.pos_max = pos_lim[1]  # Upper position limits
        self.pos_min = pos_lim[0]  # Lower position limits
        self.vel_max = robot.vel_lim.to(dtype=self.dtype, device=self.device)  # Velocity limits


    def sample_inputs(self, batch_size:int=1):
        """
        Generate random joint positions, velocities, and trajectory parameters 
        within the specified limits.

        Parameters:
        - batch_size (int): Number of samples to generate.

        Returns:
        - qpos (torch.Tensor): Random joint positions within the position limits.
        - qvel (torch.Tensor): Random joint velocities within the velocity limits.
        - traj_param (torch.Tensor): Random trajectory parameters in the range [-1, 1].
        """
        # Sample random joint positions within the position limits
        qpos = (self.pos_max - self.pos_min) * torch.rand((batch_size, self.dof), dtype=self.dtype, device=self.device) + self.pos_min 
        # Sample random joint velocities within the velocity limits
        qvel = 2 * self.vel_max * torch.rand((batch_size, self.dof), dtype=self.dtype, device=self.device) - self.vel_max 
        # Sample random trajectory parameters in the range [-1, 1]
        traj_param = 2 * torch.rand((batch_size, self.dof), dtype=self.dtype, device=self.device) - 1

        return qpos, qvel, traj_param

    def get_base_joint_occ(self):
        """
        Get the base joint's spherical occupancy center and radius.

        Returns:
        - centers (torch.Tensor): The base joint occupancy center.
        - radii (torch.Tensor): The base joint occupancy radius.
        """
        # Sample random joint inputs
        qpos, qvel, traj_param = self.sample_inputs()
        _, JRS_R = self.JRS(qpos, qvel, self.joint_axis)  # Obtain joint reachable set for these inputs
        joint_occ, _, _ = sphere_occupancy(JRS_R, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)

        # Extract the occupancy centers and radii at the base joint
        centers_bpz, radii = next(iter(joint_occ.values()))

        # Expand trajectory parameters across time steps
        batched_traj_param = traj_param.unsqueeze(1).expand((-1, self.n_timesteps, self.dof))
        centers = centers_bpz.center_slice_all_dep(batched_traj_param).squeeze(0)

        return centers[0].reshape(1, 1, self.dimension), radii[0].reshape(1, 1)  # (1, 1, dimension), (1, 1)

    def compute_joint_occ(self, qpos:torch.Tensor, qvel:torch.Tensor, traj_param:torch.Tensor, generate_grad:bool = False):
        """
        Compute the joint spherical occupancy based on positions, velocities, and trajectory parameters.

        Parameters:
        - qpos (torch.Tensor): Joint positions.
        - qvel (torch.Tensor): Joint velocities.
        - traj_param (torch.Tensor): Trajectory parameters.
        - generate_grad (bool): Whether to compute the gradient of the centers.

        Returns:
        - centers (torch.Tensor): Centers of the spherical occupancy for each joint.
        - radii (torch.Tensor): Radii of the spherical occupancy for each joint.
        """
        _, JRS_R = self.JRS(qpos, qvel, self.joint_axis)  # Get the reachable set for the joints

        # Compute the spherical occupancy for each joint
        joint_occ, _, _ = sphere_occupancy(JRS_R, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)

        # Stack the occupancy centers and radii for all joints
        centers_bpz = zp.stack([pz for pz, _ in joint_occ.values()])
        radii = torch.stack([r for _, r in joint_occ.values()])

        # Expand trajectory parameters across time steps
        batched_traj_param = traj_param.unsqueeze(1).expand((-1, self.n_timesteps, self.dof))

        if generate_grad:
            # Compute gradient of the centers with respect to the trajectory parameters
            centers_jac = centers_bpz.grad_center_slice_all_dep(batched_traj_param)
            return centers_jac.permute(1, 2, 0, 3, 4)[:, :, 1:], None  # Shape: (batch_size, timesteps, dof, dimension, dof), None
        else:
            # Slice centers based on the trajectory parameters
            centers = centers_bpz.center_slice_all_dep(batched_traj_param)
            return centers.permute(1, 2, 0, 3)[:, :, 1:], radii.permute(1, 2, 0)[:, :, 1:] # Shape: (batch_size, timesteps, dof, dimension), (batch_size, timesteps, dof)
        # [:,:,1:] is for excluding a base joint, which is constant.

    def generate_dataset(self, file_path: str, data_size: int = int(1e6), batch_size: int = 200, random_seed: int = 0, generate_grad: bool = False, exclude_last_el: bool = True):
        """
        Generate and optionally save a dataset for a robot's spherical occupancy.

        Parameters:
        - file_path (str): Path to save the generated dataset.
        - data_size (int): Total number of samples to generate.
        - batch_size (int): Number of samples per batch.
        - random_seed (int): Seed for random number generation.
        - generate_grad (bool): If True, generates gradient of joint centers w.r.t trajectory parameters.
        - exclude_last_el (bool): If True, excludes the last element in the lower triangular indices during gradient generation.
        """

        # Set the random seed for reproducibility
        torch.manual_seed(random_seed)

        # Ensure data_size is a multiple of batch_size
        if data_size > batch_size:
            # Calculate the number of full batches
            n_batches = data_size // batch_size
            # Adjust data_size to be a multiple of batch_size
            data_size = n_batches * batch_size
        else:
            # If data_size is less than batch_size, handle all data in one batch
            n_batches = 1
            batch_size = data_size

        # Preallocate tensors for storing the dataset
        data_to_save = {
            "robot_name": self.robot_name,  # Robot name for reference
            "data_class": 'gradient' if generate_grad else 'occupancy',  # Specify if the dataset includes gradients
            "joint_positions": torch.zeros((data_size, self.dof), dtype=self.dtype, device='cpu'),  # Joint positions data
            "joint_velocities": torch.zeros((data_size, self.dof), dtype=self.dtype, device='cpu'),  # Joint velocities data
            "trajectory_parameters": torch.zeros((data_size, self.dof), dtype=self.dtype, device='cpu'),  # Trajectory parameters data
            "g_ka": self.g_ka,  # Example parameter for dataset
            "pos_max": self.pos_max,  # Max position limit for joints
            "pos_min": self.pos_min,  # Min position limit for joints
            "vel_max": self.vel_max,  # Max velocity limit for joints
            "dof": self.dof,  # Degrees of freedom of the robot
            "data_size": data_size,  # Total number of samples
            "n_timesteps": self.n_timesteps,  # Number of timesteps in trajectories
        }


        # Generate data depending on whether gradient or occupancy is needed
        if generate_grad:
            print('Generating the gradient of joint centers w.r.t. trajectory parameters.')
            # Get lower triangular indices for the gradient (Jacobian matrix)
            idx1, idx2 = torch.tril_indices(self.dof, self.dof)
            
            # Option to exclude the last element in the lower triangular matrix
            if exclude_last_el:
                idx1, idx2 = idx1[:-1].tolist(), idx2[:-1].tolist()
            else:
                idx1, idx2 = idx1.tolist(), idx2.tolist()

            n_nonzero_el_in_jac = len(idx1)  # Number of non-zero elements in the Jacobian

            # Preallocate space for storing the gradient data
            data_to_save["joint_occupancy_centers_gradient"] = torch.zeros(
                (data_size, self.n_timesteps, n_nonzero_el_in_jac, self.dimension), 
                dtype=self.dtype, device='cpu'
            )
        else:
            print('Generating spherical joint occupancy data.')
            # Preallocate space for storing occupancy center and radii data
            data_to_save["joint_occupancy_centers"] = torch.zeros(
                (data_size, self.n_timesteps, self.dof, self.dimension), 
                dtype=self.dtype, device='cpu'
            )
            data_to_save["joint_occupancy_radii"] = torch.zeros(
                (data_size, self.n_timesteps, self.dof), 
                dtype=self.dtype, device='cpu'
            )

            # Get base joint occupancy centers and radii
            base_center, base_radius = self.get_base_joint_occ()
            # Store the base joint occupancy centers and radii in the dataset
            data_to_save["base_joint_occupancy_center"] = base_center.cpu()
            data_to_save["base_joint_occupancy_radius"] = base_radius.cpu()

        # Display information about the dataset generation process
        print(f'Generating dataset with {data_size} samples, batch size: {batch_size}, seed: {random_seed}.')
        print(f'Using dtype: {self.dtype} and device: {self.device}.')

        # Loop through all batches and generate the dataset
        for batch_idx in tqdm(range(n_batches)):
            # Slice for the current batch (for indexing)
            batch_slice = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)

            # Sample inputs: joint positions, velocities, and trajectory parameters
            qpos, qvel, traj_param = self.sample_inputs(batch_size)

            # Compute spherical occupancy for the given inputs (joint positions, velocities, and trajectory parameters)
            centers, radii = self.compute_joint_occ(qpos, qvel, traj_param, generate_grad)

            if generate_grad:
                # If generating gradients, flatten the centers for the Jacobian matrix and store in the dataset
                center_jac_flat = centers[..., idx1, :, idx2].permute(1, 2, 0, 3)
                data_to_save["joint_occupancy_centers_gradient"][batch_slice] = center_jac_flat.cpu()
            else:
                # Store the computed occupancy centers and radii in the dataset
                data_to_save["joint_occupancy_centers"][batch_slice] = centers.cpu()
                data_to_save["joint_occupancy_radii"][batch_slice] = radii.cpu()

            # Store joint positions, velocities, and trajectory parameters in the dataset
            data_to_save["joint_positions"][batch_slice] = qpos.cpu()
            data_to_save["joint_velocities"][batch_slice] = qvel.cpu()
            data_to_save["trajectory_parameters"][batch_slice] = traj_param.cpu()

        # Save the generated dataset to an HDF5 file
        self.save_to_hdf5(file_path, data_to_save)
        print('Dataset successfully saved!')

    def save_to_hdf5(self, file_path: str, data: dict):
        """
        Save the dataset to an HDF5 file

        Parameters:
        - file_path (str): Path to save the file.
        - data (dict): Data to save, containing tensors and metadata.
        """
        with h5py.File(file_path, 'w') as f:
            for key, value in data.items():
                if isinstance(value, str):
                    # Save string data
                    f.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding='utf-8', length=20))
                elif isinstance(value, torch.Tensor):
                    # Save tensor data
                    f.create_dataset(key, data=value.cpu().numpy())
                else:
                    # Save scalar or non-tensor data
                    f.create_dataset(key, data=np.array([value]))        


def read_params():
    """
    Parses command-line arguments for the data generation script.
    """
    parser = argparse.ArgumentParser(description="Spherical Occupancy Dataset Generator")
    parser.add_argument('--batch_size', type=int, default=500, help="Number of samples per batch")
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else -1, 
                        choices=range(-1, torch.cuda.device_count()), help="Device to use: GPU or CPU")
    parser.add_argument('--dtype', type=int, default=32, help="Data type: 32 for float32, 64 for float64")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for data generation")
    parser.add_argument('--robot_name', type=str, default='kinova_gen3', help="Name of the robot")
    parser.add_argument('--grad', action='store_true', help="Flag to generate gradient data")
    parser.add_argument('--data_size', type=int, default=int(1e6), help="Total number of samples to generate")
    return parser.parse_args()

def main():
    """
    Main function to generate and save a dataset for a robot's spherical occupancy.
    """
    # Read parameters from command-line arguments
    params = read_params()

    # Set the computation device: either CPU or a specified CUDA GPU device
    device = torch.device('cpu') if params.device < 0 else torch.device(f'cuda:{params.device}')

    # Set the data type (float32 or float64) for the computations
    dtype = torch.float32 if params.dtype == 32 else torch.float64
    
    # Load the robot model and set the visualization flag
    zpr.DEBUG_VIZ = False # Disable debugging visualization
    zpr_dirname = os.path.dirname(zpr.__file__)  # Get the directory of the zpr module
    if params.robot_name == 'kinova_gen3':
        robot_path = 'robots/assets/robots/kinova_arm/gen3.urdf'
    
    # Load the robot model, specifying data type and device, and initialize joint occupancy
    robot = zpr.ZonoArmRobot.load(
        os.path.join(zpr_dirname, robot_path), 
        dtype=dtype, 
        device=device, 
        create_joint_occupancy=True
    )

    # Set up the directory for saving the dataset
    base_dirname = os.path.dirname(__file__)
    save_dirname = os.path.join(base_dirname,'so_dataset')
    if not os.path.exists(save_dirname):
        os.mkdir(save_dirname)

    # Define the file path to save the dataset
    file_path = os.path.join(save_dirname,f'{params.robot_name}_so'+('_grad' if params.grad else '')+'.hdf5')

    # Set Joint Radius Override
    joint_radius_override = {
            'joint_1': torch.tensor(0.0503305, dtype=dtype, device=device),
            'joint_2': torch.tensor(0.0630855, dtype=dtype, device=device),
            'joint_3': torch.tensor(0.0463565, dtype=dtype, device=device),
            'joint_4': torch.tensor(0.0634475, dtype=dtype, device=device),
            'joint_5': torch.tensor(0.0352165, dtype=dtype, device=device),
            'joint_6': torch.tensor(0.0542545, dtype=dtype, device=device),
            'joint_7': torch.tensor(0.0364255, dtype=dtype, device=device),
            'end_effector': torch.tensor(0.0394685, dtype=dtype, device=device),
        }

    # Initialize the ReachableSetGenerator
    so_generator = ReachableSetGenerator(
        robot=robot, 
        robot_name = params.robot_name,
        zono_order=2,  # Default zono_order, you can change this as needed
        joint_radius_override=joint_radius_override, 
        dtype=dtype, 
        device=device
    )
    
    # Generate the dataset
    so_generator.generate_dataset(
        file_path=file_path, 
        data_size=params.data_size, 
        batch_size=params.batch_size, 
        random_seed=params.seed,
        generate_grad=params.grad,
    )

if __name__ == '__main__':
    main()


