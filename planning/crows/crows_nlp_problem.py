import torch
import numpy as np
import sys, os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from distance_net.batched_distance_and_gradient_net import BatchedDistanceGradientNet
from forward_occupancy.SO import make_spheres
from training.models import MLP_Joint_Occ, MLP_Grad

class NeuralCrowsConstraints:
    def __init__(self, 
                 dimension = 3, 
                 dtype = torch.float, 
                 device=None, 
                 max_spheres=5000, 
                 n_params=7,
                 model_dir= os.path.join(os.path.dirname(__file__), '../../trained_models'),
                 use_learned_grad = True,
                 confidence_idx = 0,
                 ):
        self.dimension = dimension
        self.n_params = n_params
        self.dtype = dtype
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
        if device is None:
            device = torch.empty(0,dtype=dtype).device
        self.device = device

        self.distance_net = BatchedDistanceGradientNet().to(device).eval()
        # self.distance_net = torch.jit.script(self.distance_net)
        # self.distance_net = torch.jit.optimize_for_inference(self.distance_net)

        # Load Model 
        self.use_learned_grad = use_learned_grad
        self.confidence_idx = confidence_idx
        self.__load_model(model_dir)
        self.__allocate_base_params(max_spheres)
        
        time_ = torch.arange(self.n_time, dtype=self.dtype, device=self.device).view(self.n_time, 1) 
        self.time_nn = (2 * time_ / self.n_time - 1) #.expand(-1,self.n_robots,1)

    def __load_model(self, model_dir):
        with open(os.path.join(model_dir, 'model_config.json')) as f:
            model_config = json.load(f)

        self.model = MLP_Joint_Occ(**model_config['kwargs_for_model'])
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=self.device, weights_only=True))
        self.model._setup_robot_params(**model_config['kwargs_for_robot_params'])
        self.model._setup_base_joint_occ(**model_config['kwargs_for_base_joint_occ'])
        self.model._conformalize(**model_config['kwargs_for_conformal_prediction'])
        
        self.model.to(dtype=self.dtype, device=self.device)
        self.model.eval()
        self.confidence_bound = self.model.quantile_for_each_joint[self.confidence_idx] #.view(1, dof) # NOTE 0: 99.999%, 1: 99.99%, 2: 99.9%, 3: 99% 4:90% 5:80%

        self.dof = self.model.dof
        self.g_ka = self.model.g_ka * np.ones((self.dof),dtype=self.np_dtype)
        self.n_time = self.model.n_timesteps
        self.n_joints = self.dof + 1 # include end effectors

        if self.use_learned_grad:
            with open(os.path.join(model_dir, 'model_config_grad.json')) as f:
                model_grad_config = json.load(f)
            self.model_grad = MLP_Grad(**model_grad_config['kwargs_for_model'])
            self.model_grad.load_state_dict(torch.load(os.path.join(model_dir, 'model_grad.pth'), map_location=self.device, weights_only=True))
            self.model_grad.to(dtype=self.dtype, device=self.device)
            self.model_grad.eval()
            self.idx1, self.idx2 = [i+1 for i in self.model_grad.idx1], self.model_grad.idx2 
            self.grad_scale = self.model_grad.scale


    def __allocate_base_params(self, max_spheres):
        self.max_spheres = max_spheres
        self.__centers = torch.empty((self.max_spheres, self.dimension), dtype=self.dtype, device=self.device)
        self.__radii = torch.empty((self.max_spheres), dtype=self.dtype, device=self.device)
        self.__center_jac = torch.empty((self.max_spheres, self.dimension, self.n_params), dtype=self.dtype, device=self.device)
        self.__radii_jac = torch.empty((self.max_spheres, self.n_params), dtype=self.dtype, device=self.device)

        # For base joint occupancy,
        self.__centers[:self.n_time] = self.model.base_center
        self.__radii[:self.n_time] = self.model.base_radius

        # Fill only lower triangular elements
        self.__center_jac[:self.n_time*self.n_joints] = 0
        # For all joint occupancy, radii Jacobian is zero
        self.__radii_jac[:self.n_time*self.n_joints] = 0

    def set_params(self, qpos, qvel, p_idx, n_spheres_per_link, obs_tuple):
        
        qpos = torch.as_tensor(qpos, dtype=self.dtype, device=self.device) #.view(self.n_robots, self.dof)
        qvel = torch.as_tensor(qvel, dtype=self.dtype, device=self.device) #.view(self.n_robots, self.dof)
        
        # Normalize Inputs for Neural Network
        qpos = (2 * qpos - self.model.pos_max - self.model.pos_min) / (self.model.pos_max - self.model.pos_min)
        qvel = qvel  / self.model.vel_max

        self.qpos_nn = qpos.expand(self.n_time, -1)
        self.qvel_nn = qvel.expand(self.n_time, -1)

        # Add confidence bound in radii
        with torch.no_grad():
            radii = self.model.predict_radii_without_base(self.qpos_nn, self.qvel_nn, self.time_nn) + self.confidence_bound
        
        self.p_idx = p_idx
        n_pairs = p_idx.shape[-1]

        self.n_spheres = n_spheres_per_link
        self.total_spheres = self.n_joints*self.n_time + self.n_spheres*n_pairs*self.n_time
        
        if self.total_spheres > self.max_spheres:
            # reallocate new tensors
            self.__allocate_base_params(self.total_spheres)

        ### num constraints
        self.M = self.total_spheres  
        ###

        ## Utilize underlying storage and update initial radii values
        self.centers = self.__centers[:self.total_spheres]
        self.radii = self.__radii[:self.total_spheres]
        self.center_jac = self.__center_jac[:self.total_spheres]
        self.radii_jac = self.__radii_jac[:self.total_spheres]

        self.radii.view(-1, self.n_time)[1:self.n_joints] = radii.transpose(0,1) # Exclud base joint
        #self.radii_jac[:self.n_time*self.n_joints] = 0

        ## Obstacle data
        self.obs_tuple = obs_tuple

    def NN_fun(self, points, obs_tuple):
        '''This function is used to compute the distance and gradient of the distance to the nearest obstacle for each point in points.
        
        Args:
            points: A tensor of shape (n_points, 3) representing the points to compute the distance and gradient of the distance to the nearest obstacle for.
            obs_tuple: A tuple of tensors representing the obstacles
            
        Returns:
            A tuple of tensors representing the distance and gradient of the distance to the nearest obstacle for each point in points.
        '''
        dist_out, grad_out = self.distance_net(points, *obs_tuple)
        min_dists, idxs = dist_out.min(dim=-1)
        return min_dists, grad_out.gather(1, idxs.reshape(-1, 1, 1).expand(-1, -1, 3)).squeeze(1)


    def __call__(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if Cons_out is None:
            Cons_out = np.empty(self.M, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M, self.n_params), dtype=self.np_dtype)
        Cons_out = torch.from_numpy(Cons_out)
        Jac_out = torch.from_numpy(Jac_out)


        # Batch form of batch construction
        centers = self.centers.view(-1, self.n_time, self.dimension)
        center_jac = self.center_jac.view(-1, self.n_time, self.dimension, self.n_params)
        radii = self.radii.view(-1, self.n_time)

        with torch.no_grad():
            if self.use_learned_grad:
                x_nn = x.expand(self.n_time, -1)
                # n_time, n_joints, dimension -> n_joints, n_time, dimension
                centers_pred = self.model.predict_centers_without_base(self.qpos_nn, self.qvel_nn, x_nn, self.time_nn).transpose(0,1) 
                center_jac_flat_pred = self.model_grad.predict_centers_jac_flat_without_base(self.qpos_nn, self.qvel_nn, x_nn, self.time_nn).transpose(0,1) 
                centers[1:self.n_joints] = centers_pred
                center_jac[self.idx1, :,:,self.idx2] = self.grad_scale * center_jac_flat_pred
            else:
                def aux_fun(aux_input):
                    aux_input_nn = aux_input.expand(self.n_time,-1)
                    aux_centers = self.model.predict_centers_without_base(self.qpos_nn, self.qvel_nn, aux_input_nn, self.time_nn).transpose(0,1) 
                    return aux_centers, aux_centers
                center_jac_pred, centers_pred = torch.func.jacfwd(aux_fun, argnums=0, has_aux=True)(x)
                centers[1:self.n_joints] = centers_pred
                center_jac[1:self.n_joints] = center_jac_pred
        ###
        joint_centers = centers[self.p_idx]
        joint_radii = radii[self.p_idx]
        joint_jacs = center_jac[self.p_idx]
        spheres = make_spheres(joint_centers[0], joint_centers[1], joint_radii[0], joint_radii[1], joint_jacs[0], joint_jacs[1], self.n_spheres)
        sidx = self.n_joints * self.n_time

        self.centers[sidx:] = spheres[0].reshape(-1,self.dimension)
        self.radii[sidx:] = spheres[1].reshape(-1)
        self.center_jac[sidx:] = spheres[2].reshape(-1,self.dimension,self.n_params)
        self.radii_jac[sidx:] = spheres[3].reshape(-1,self.n_params)
        
        # Do what you need with the centers and radii
        # NN(centers) - r > 0
        # D_NN(c(k)) -> D_c(NN) * D_k(c) - D_k(r)
        # D_c(NN) should have shape (n_spheres, 3, 1)
        # D_k(c) has shape (n_spheres, 3, n_params)
        # D_k(r) has shape (n_spheres, n_params)
        
        dist, dist_jac = self.NN_fun(self.centers, self.obs_tuple)
        cons_dists_out = Cons_out[:self.total_spheres]
        cons_dists_jac_out = Jac_out[:self.total_spheres]
        cons_dists_out.copy_(-(dist - self.radii))
        cons_dists_jac_out.copy_(-((dist_jac.unsqueeze(-1) * self.center_jac).sum(1) - self.radii_jac))

        return Cons_out, Jac_out
