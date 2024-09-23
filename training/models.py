import torch 
import torch.nn as nn

class MLP_Joint_Occ(nn.Module):
    def __init__(self,
                 dimension = 3,
                 dof = 7,
                 center_net_arch = [256, 256, 256],
                 radius_net_arch = [256, 256], 
                 center_activation_fn = 'gelu',
                 radius_activation_fn = 'relu'
                 ):
        """
        Initialize the MLP model for predicting joint occupancy centers and radii.

        Args:
        - dimension (int): The spatial dimension of the joint centers (typically 3 for 3D space).
        - n_joints (int): The number of joints in the robot.
        - center_net_arch (list of int): List specifying the sizes of a center network.
        - radius_net_arch (list of int): List specifying the sizes of a radius network.
        """
        super().__init__()

        # Check that the provided parameters are valid
        assert dimension > 0 and dof > 0, "Dimension and degrees of freedom should be greater than 0."
        assert center_activation_fn in ['relu', 'gelu'], 'Activation function for center network should be either ReLU or GELU.'
        assert radius_activation_fn in ['relu', 'gelu'], 'Activation function for radius network should be either ReLU or GELU.'

        # Define model parameters
        self.dimension = dimension
        self.dof = dof

        # Define the input and output sizes for the two networks
        self.centers_input_size = dof * 3 + 1  # Joint positions, velocities, and time (including ka)
        self.radii_input_size = dof * 2 + 1    # Joint positions, velocities, and time (excluding ka)
        self.centers_output_size = dof * dimension  # Output size: predicted centers
        self.radii_output_size = dof            # Output size: predicted radii

        # Activation functions
        self.center_activation_fn = nn.ReLU() if center_activation_fn == 'relu' else nn.GELU()
        self.radius_activation_fn = nn.ReLU() if radius_activation_fn == 'relu' else nn.GELU()

        # Build center network architecture
        center_layers = [nn.Linear(self.centers_input_size, center_net_arch[0]), self.center_activation_fn]
        for i in range(1, len(center_net_arch)):
            center_layers.append(nn.Linear(center_net_arch[i - 1], center_net_arch[i]))
            center_layers.append(self.center_activation_fn)
        center_layers.append(nn.Linear(center_net_arch[-1], self.centers_output_size))  # Output layer for centers

        # Build radius network architecture
        radius_layers = [nn.Linear(self.radii_input_size, radius_net_arch[0]), self.radius_activation_fn]
        for i in range(1, len(radius_net_arch)):
            radius_layers.append(nn.Linear(radius_net_arch[i - 1], radius_net_arch[i]))
            radius_layers.append(self.radius_activation_fn)
        radius_layers.append(nn.Linear(radius_net_arch[-1], self.radii_output_size))  # Output layer for radii

        # Combine layers into nn.Sequential models
        self.center_network = nn.Sequential(*center_layers)
        self.radius_network = nn.Sequential(*radius_layers)
        

    def _setup_robot_params(self, g_ka, pos_max, pos_min, vel_max, n_timesteps,):
        """
        Set up robot-related parameters as buffers for normalization.

        Args:
        - g_ka (Tensor): Ka parameter (trajectory parameter).
        - pos_max (Tensor or list): Maximum joint positions.
        - pos_min (Tensor or list): Minimum joint positions.
        - vel_max (Tensor or list): Maximum joint velocities.
        - n_timesteps (int): Number of timesteps.
        """
        self.g_ka = g_ka
        self.register_buffer('pos_max', pos_max if isinstance(pos_max, torch.Tensor) else torch.tensor(pos_max)) 
        self.register_buffer('pos_min', pos_min if isinstance(pos_min, torch.Tensor) else torch.tensor(pos_min))
        self.register_buffer('vel_max', vel_max if isinstance(vel_max, torch.Tensor) else torch.tensor(vel_max))
        self.n_timesteps = n_timesteps

    def _conformalize(self, quantile_for_each_joint, alpha):
        """
        Store calibration-related parameters as buffers for conformal prediction.

        Args:
        - quantile_for_each_joint (Tensor or list): Quantiles for each joint.
        - alpha (Tensor or float): Confidence level for conformal prediction.
        """
        self.register_buffer('quantile_for_each_joint', quantile_for_each_joint if isinstance(quantile_for_each_joint, torch.Tensor) else torch.tensor(quantile_for_each_joint)) # (N, dof)
        self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)) # (N,)

    def _setup_base_joint_occ(self, base_center, base_radius):
        """
        Set up base joint occupancy for centers and radii as buffers.

        Args:
        - base_center (Tensor or list): Base center position for joint occupancy.
        - base_radius (Tensor or float): Base radius for joint occupancy.
        """
        self.register_buffer('base_center', base_center if isinstance(base_center, torch.Tensor) else torch.tensor(base_center)) 
        self.register_buffer('base_radius', base_radius if isinstance(base_radius, torch.Tensor) else torch.tensor(base_radius))    

    def normalize_inputs(self, qpos, qvel, t):
        """
        Normalize the inputs (joint positions, velocities, and time) based on pre-set robot parameters.
        """
        qpos = (2 * qpos - self.pos_max - self.pos_min) / (self.pos_max - self.pos_min)  # Normalize joint positions
        qvel = qvel / self.vel_max  # Normalize joint velocities
        t = 2 * t / self.n_timesteps - 1  # Normalize time
        return qpos, qvel, t

    def predict_centers(self, qpos, qvel, ka, t):
        """
        Predict the centers using qpos, qvel, ka, and t.
        """
        centers = self.predict_centers_without_base(qpos, qvel, ka, t)
        return torch.cat((self.base_center.expand(centers.size(0),1,self.dimension),centers),dim=-2)

       
    def predict_radii(self, qpos, qvel, t):
        """
        Predict the radii using qpos, qvel, and t (excluding ka).
        """
        radii = self.predict_radii_without_base(qpos, qvel, t)
        return torch.cat((self.base_radius.expand(radii.size(0),1),radii),dim=-1)


    def predict_centers_without_base(self, qpos, qvel, ka, t):
        """
        Predict the centers using qpos, qvel, ka, and t.
        """
        x = torch.cat([qpos, qvel, ka, t], dim=-1)  # Exclude ka
        centers = self.center_network(x)
        return centers.view(-1, self.dof, self.dimension)
        #return torch.einsum('i,...ij->...ij',self.center_scaler,centers)

       
    def predict_radii_without_base(self, qpos, qvel, t):
        """
        Predict the radii using qpos, qvel, and t (excluding ka).
        """
        x = torch.cat([qpos, qvel, t], dim=-1) 
        radii = self.radius_network(x)
        return radii.view(-1, self.dof)

    def forward(self, qpos, qvel, ka, t):
        """
        Predict both centers and radii.

        Args:
        - qpos (Tensor): Joint positions.
        - qvel (Tensor): Joint velocities.
        - ka (Tensor): Trajectory parameter.
        - t (Tensor): Timesteps.

        Returns:
        - centers (Tensor): Predicted centers of spherical occupancy.
        - radii (Tensor): Predicted radii of spherical occupancy.
        """
        qpos, qvel, t = self.normalize_inputs(qpos, qvel, t)
        centers = self.predict_centers_without_base(qpos, qvel, ka, t)
        radii = self.predict_radii_without_base(qpos, qvel, t)
        return centers, radii


class MLP_Grad(nn.Module):
    def __init__(self,
                 dimension=3,
                 dof=7,
                 center_jac_net_arch=[256, 256, 256, 256],
                 center_jac_activation_fn='gelu',
                 scale=1e-2,
                 exclude_last_el=True):
        """
        Initialize the MLP model for predicting the Jacobian of joint occupancy centers.

        Args:
        - dimension (int): Spatial dimension of the joint centers (typically 3 for 3D space).
        - dof (int): Degrees of freedom (number of joints in the robot).
        - center_jac_net_arch (list of int): List specifying the hidden layer sizes for the Jacobian network.
        - center_jac_activation_fn (str): Activation function for the Jacobian network ('relu' or 'gelu').
        - scale (float): Scaling factor for the Jacobian outputs.
        - exclude_last_el (bool): If True, excludes the last element of the lower triangular indices.
        """
        super().__init__()

        # Check validity of provided parameters
        assert dimension > 0 and dof > 0, "Dimension and degrees of freedom should be greater than 0."
        assert center_jac_activation_fn in ['relu', 'gelu'], "Activation function should be either 'relu' or 'gelu'."

        # Model parameters
        self.dimension = dimension
        self.dof = dof
        self.scale = scale

        # Generate lower triangular indices of a matrix for the Jacobian
        idx1, idx2 = torch.tril_indices(dof, dof)
        if exclude_last_el:
            self.idx1, self.idx2 = idx1[:-1].tolist(), idx2[:-1].tolist()  # Exclude last element if specified
        else:
            self.idx1, self.idx2 = idx1.tolist(), idx2.tolist()  # Include all elements

        # Define the input and output sizes for the Jacobian prediction network
        self.centers_jac_input_size = dof * 3 + 1  # Joint positions, velocities, and time (including ka)
        self.n_nonzero_el = len(self.idx1)  # Number of non-zero elements in the lower triangular matrix
        self.centers_jac_output_size = self.n_nonzero_el * dimension  # Jacobian output size (depends on dimension)

        # Set the activation function for the Jacobian network
        self.center_jac_activation_fn = nn.ReLU() if center_jac_activation_fn == 'relu' else nn.GELU()

        # Build the center Jacobian network architecture
        center_jac_layers = [nn.Linear(self.centers_jac_input_size, center_jac_net_arch[0]), self.center_jac_activation_fn]

        # Add hidden layers to the Jacobian network
        for i in range(1, len(center_jac_net_arch)):
            center_jac_layers.append(nn.Linear(center_jac_net_arch[i - 1], center_jac_net_arch[i]))
            center_jac_layers.append(self.center_jac_activation_fn)
        
        # Add output layer for the Jacobian
        center_jac_layers.append(nn.Linear(center_jac_net_arch[-1], self.centers_jac_output_size))
        
        # Combine the layers into a sequential model
        self.center_jac_network = nn.Sequential(*center_jac_layers)

    def _setup_robot_params(self, g_ka, pos_max, pos_min, vel_max, n_timesteps,):
        """
        Set up robot-related parameters as buffers for normalization.

        Args:
        - g_ka (Tensor): Ka parameter (trajectory parameter).
        - pos_max (Tensor or list): Maximum joint positions.
        - pos_min (Tensor or list): Minimum joint positions.
        - vel_max (Tensor or list): Maximum joint velocities.
        - n_timesteps (int): Number of timesteps.
        """
        self.g_ka = g_ka
        self.register_buffer('pos_max', pos_max if isinstance(pos_max, torch.Tensor) else torch.tensor(pos_max)) 
        self.register_buffer('pos_min', pos_min if isinstance(pos_min, torch.Tensor) else torch.tensor(pos_min))
        self.register_buffer('vel_max', vel_max if isinstance(vel_max, torch.Tensor) else torch.tensor(vel_max))
        self.n_timesteps = n_timesteps

    def normalize_inputs(self, qpos, qvel, t):
        """
        Normalize the inputs (joint positions, velocities, and time) based on pre-set robot parameters.
        """
        qpos = (2 * qpos - self.pos_max - self.pos_min) / (self.pos_max - self.pos_min)
        qvel = qvel  / self.vel_max
        t = 2 * t / self.n_timesteps - 1
        return qpos, qvel, t

    def predict_centers_jac_flat_without_base(self, qpos, qvel, ka, t):
        """
        Predict the flattened Jacobian of the joint occupancy centers.
        """
        # Concatenate normalized inputs for the prediction
        x = torch.cat([qpos, qvel, ka, t], dim=-1)
        
        # Forward pass through the center Jacobian network
        centers_jac = self.center_jac_network(x)
        
        # Reshape the output to (batch_size, n_nonzero_el, dimension)
        return centers_jac.view(-1, self.n_nonzero_el, self.dimension)

    def forward(self, qpos, qvel, ka, t):
        """
        Forward pass: Predict the flattened Jacobian of the centers.

        Args:
        - qpos (Tensor): Joint positions.
        - qvel (Tensor): Joint velocities.
        - ka (Tensor): Trajectory parameter.
        - t (Tensor): Timesteps.

        Returns:
        - centers_jac (Tensor): Predicted Jacobian matrix for joint centers.
        """
        # Normalize inputs before feeding them to the network
        qpos, qvel, t = self.normalize_inputs(qpos, qvel, t)
        
        # Predict the Jacobian of the centers
        centers_jac = self.predict_centers_jac_flat_without_base(qpos, qvel, ka, t)
        
        return centers_jac









