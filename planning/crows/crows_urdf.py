import torch
import numpy as np
import zonopy as zp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from forward_occupancy.SO import sphere_occupancy
import cyipopt

import time
import json 

T_PLAN, T_FIN = 0.5, 1.0

from planning.crows.crows_nlp_problem import NeuralCrowsConstraints
from planning.common.base_armtd_nlp_problem import BaseArmtdNlpProblem

from zonopyrobots import ZonoArmRobot

from distance_net.compute_vertices_from_generators import compute_edges_from_generators

class CROWS_3D_planner():
    def __init__(self,
                 robot: ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 100,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 sphere_device: torch.device = torch.device('cpu'),
                 spheres_per_link: int = 5,
                 use_weighted_cost: bool = False,
                 joint_radius_override: dict = {},
                 linear_solver: str = 'ma27',
                 model_dir: str = os.path.join(os.path.dirname(__file__), '../../trained_models'), # NOTE: new 
                 use_learned_grad: bool = True,
                 confidence_idx: int = 0,
                 ):
        '''CROWS 3D receding horizon arm robot planner
        
        Args:
            robot: ZonoArmRobot
                The robot to plan for.
            dtype: torch.dtype, Optional
                The datatype to use for the calculations. Default is torch.float.
                This should match the datatype of the robot.
            device: torch.device, Optional
                The device to use for the calculations. Default is torch.device('cpu').
            sphere_device: torch.device, Optional
                The device to use for the sphere calculations. Default is torch.device('cpu').
            spheres_per_link: int, Optional
                The number of spheres to use per link for the sphere calculations. Default is 5.
            use_weighted_cost: bool, Optional
                Whether to use a weighted cost for the NLP. Default is False.
            joint_radius_override: dict, Optional
                A dictionary of joint radius overrides. Default is {}.
                This is used to override the joint radius for the forward occupancy calculations.
            linear_solver: str, Optional
                The linear solver to use for the NLP. Default is 'ma27'.

        '''
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.linear_solver = linear_solver
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)

        self._setup_robot(robot)
        self.sphere_device = sphere_device
        self.spheres_per_link = spheres_per_link
        self.CROWS_constraint = NeuralCrowsConstraints(dtype=dtype, device=sphere_device, n_params=self.dof, model_dir=model_dir, use_learned_grad=use_learned_grad, confidence_idx=confidence_idx)
        self.joint_radius_override = joint_radius_override
        self.__preprocess_SO_constants()
        # Prepare the nlp
        self.g_ka = np.ones((self.dof), dtype=self.np_dtype)*self.CROWS_constraint.model.g_ka # load g_ka from json file
        # Radius of maximum steady state oscillation that can occur for the given g_ka
        self.osc_rad = self.g_ka*(T_PLAN**2)/8
        self.nlp_problem_obj = BaseArmtdNlpProblem(self.dof,
                                         self.g_ka,
                                         self.pos_lim, 
                                         self.vel_lim,
                                         self.continuous_joints,
                                         self.pos_lim_mask,
                                         self.dtype,
                                         T_PLAN,
                                         T_FIN,
                                         weight_joint_costs = use_weighted_cost)
    
    def _setup_robot(self, robot: ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.vel_lim = np.clip(robot.np.vel_lim, a_min=None, a_max=self.JRS.g_ka * T_PLAN)
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass
    
    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
    
    def __preprocess_SO_constants(self):
        ## Preprocess some constraint constants
        ## Use this only for preprocessing indices
        # empty rotatotope will return joint occupancy at statinary
        joint_occ, link_joint_pairs, _ = sphere_occupancy({}, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)
        # Flatten out all link joint pairs so we can just process that
        joint_pairs = []
        for pairs in link_joint_pairs.values():
            joint_pairs.extend(pairs)
        joints_idxs = {name: i for i, name in enumerate(joint_occ.keys())}
        # convert to indices
        p_idx = torch.empty((2, len(joint_pairs)), dtype=int, device=self.sphere_device)
        for i, (joint1, joint2) in enumerate(joint_pairs):
            p_idx[0][i] = joints_idxs[joint1]
            p_idx[1][i] = joints_idxs[joint2]
        self.p_idx = p_idx.to(device=self.sphere_device)

    def _prepare_CROWS_constraints(self, qpos, qvel, obs_zono):
        
        ### Process the obstacles
        dist_net_time = time.perf_counter()

        # Compute hyperplanes from buffered obstacles generators
        hyperplanes_A, hyperplanes_b = obs_zono.to(device=self.sphere_device).polytope(self.combs)
        # Compute vertices from buffered obstacles generators
        v1, v2 = compute_edges_from_generators(obs_zono.Z[...,0:1,:], obs_zono.Z[...,1:,:], hyperplanes_A, hyperplanes_b.unsqueeze(-1))
        # combine to one input for the NN
        obs_tuple = (hyperplanes_A, hyperplanes_b, v1, v2)

        CROWS_prep_time = time.perf_counter()
        # Build the constraint
        self.CROWS_constraint.set_params(qpos, qvel, self.p_idx, self.spheres_per_link, obs_tuple)

        final_time = time.perf_counter()
        out_times = {
            'CROWS_prep': final_time - CROWS_prep_time,
            'distance_prep_net': CROWS_prep_time - dist_net_time,
        }
        return self.CROWS_constraint, out_times

    def trajopt(self, qpos, qvel, waypoint, ka_0, CROWS_constraint, time_limit=None, t_final_thereshold=0., tol=1e-5):
        self.nlp_problem_obj.reset(qpos, qvel, waypoint.pos, CROWS_constraint, t_final_thereshold=t_final_thereshold, qdgoal=waypoint.vel)
        n_constraints = self.nlp_problem_obj.M

        nlp = cyipopt.Problem(
        n = self.dof,
        m = n_constraints,
        problem_obj=self.nlp_problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*n_constraints,
        cu = [-1e-6]*n_constraints,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', tol)
        nlp.add_option('linear_solver', self.linear_solver)
        # nlp.add_option('derivative_test', 'first-order')
        # nlp.add_option('derivative_test_perturbation', 1e-7)
        

        if time_limit is not None:
            nlp.add_option('max_wall_time', max(time_limit, 0.01))

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)
        if self.info['status'] == -12:
            raise ValueError(f"Invalid option or solver specified. Perhaps {self.linear_solver} isn't available?")
        self.final_cost = self.info['obj_val'] if self.info['status'] == 0 else None      
        return CROWS_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
    def plan(self,qpos, qvel, waypoint, obs, ka_0 = None, time_limit=None, t_final_thereshold=0., tol=1e-5):
        '''Plan a trajectory for the robot

        Args:
            qpos: np.ndarray
                The current joint positions of the robot
            qvel: np.ndarray
                The current joint velocities of the robot
            waypoint: np.ndarray
                The waypoint to plan to
            obs: tuple
                The obstacles in the environment
                The first element is the positions of the obstacles in rows of xyz
                The second element is the sizes of the obstacles in rows of xyz
                The third element (optional) is the 3x3 rotation matrices representing the orientation of each obstacle
            ka_0: np.ndarray, Optional
                The initial guess for the joint accelerations. Default is None.
            time_limit: float, Optional
                The time limit for the optimization. Default is None.
            t_final_thereshold: float, Optional
                The final time threshold for the optimization. Default is 0.
            tol: float, Optional
                The desired convergence tolerance for IPOPT solver. Default is 1e-5.
        Returns:
            np.ndarray: The joint accelerations
            int: The status of the optimization (0 is success)
            dict: The timing statistics
        '''
        # Create obs zonotopes
        preparation_time_start = time.perf_counter()
        if len(obs) == 2:
            obs_Z = torch.cat((
                torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
                torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2.
                ), dim=-2)
        elif len(obs) > 2:
            Gen =  torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2. @ torch.as_tensor(obs[2], dtype=self.dtype, device=self.device).transpose(-1,-2)
            obs_Z = torch.cat((
                torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
                Gen
                ), dim=-2)
                
        obs_zono = zp.batchZonotope(obs_Z)

        # Compute FO
        CROWS_constraint, CROWS_times = self._prepare_CROWS_constraints(qpos, qvel, obs_zono)

        preparation_time = time.perf_counter() - preparation_time_start
        if time_limit is not None:
            time_limit -= preparation_time
            
        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, waypoint, ka_0, CROWS_constraint, time_limit=time_limit, t_final_thereshold=t_final_thereshold, tol=tol)
        trajopt_time = time.perf_counter() - trajopt_time
        
        timing_stats = {
            'cost': self.final_cost,
            'nlp': trajopt_time, 
            'total_prepartion_time': preparation_time,
            'constraint_times': constraint_times,
            'num_constraint_evaluations': self.nlp_problem_obj.num_constraint_evaluations,
            'num_jacobian_evaluations': self.nlp_problem_obj.num_jacobian_evaluations,
        }
        timing_stats.update(CROWS_times)
        
        return k_opt, flag, timing_stats
