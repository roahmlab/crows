import torch
import numpy as np
import time


class BaseMultiStepNlpProblem():
    """
    Base for the ArmTD NLP problem. This class is used to define the optimization problem for the ArmTD.
    The class is designed to be used with the `ipopt` solver, and is based on the `cyipopt` interface.

    Constraints are defined as classes like in planning.armtd.armtd_nlp_problem or in planning.sparrows.sphere_nlp_problem.
    The constraints are called in the `__call__` method of the class, and Cons_out and Jac_out are updated in place.
    The M attribute is the number of constraints in a given constraint class
    """
    __slots__ = [
        't_plan',
        't_full',
        'dtype',
        'n_joints',
        'g_ka',
        'pos_lim',
        'vel_lim',
        'continuous_joints',
        'pos_lim_mask',
        'M_limits',
        '_g_ka_masked',
        '_pos_lim_masked',
        '_masked_eye',
        '_grad_qpos_peak',
        '_grad_qvel_peak',
        '_grad_qpos_brake',
        'qpos',
        'qvel',
        'qgoal',
        'qdgoal',
        '_additional_constraints',
        '_cons_val',
        'M',
        '_qpos_masked',
        '_qvel_masked',
        '_Cons',
        '_Jac',
        '_x_prev',
        '_constraint_times',
        'num_constraint_evaluations',
        'num_jacobian_evaluations',
        '_joint_cost_weights',
        'use_t_final'
        ]

    def __init__(
            self,
            n_joints: int,
            g_ka: np.ndarray, #[object],
            pos_lim: np.ndarray, #[float],
            vel_lim: np.ndarray, #[float],
            continuous_joints: np.ndarray, #[int],
            pos_lim_mask: np.ndarray, #[bool],
            dtype: torch.dtype = torch.float,
            t_plan: float = 0.5,
            t_full: float = 1.0,
            weight_joint_costs: bool = False,
            H_pred: int = 5, 
            ):

        # Core constants
        self.t_plan = t_plan
        self.t_full = t_full
        self.use_t_final = False
        # convert from torch dtype to np dtype
        self.dtype = torch.empty(0,dtype=dtype).numpy().dtype

        # Optimization parameters and range, which is known a priori for armtd
        self.n_joints = n_joints
        self.g_ka = g_ka

        # Joint limit constraints
        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.continuous_joints = continuous_joints
        self.pos_lim_mask = pos_lim_mask
        self.M_limits = int(2*self.n_joints + 6*self.pos_lim_mask.sum()) # NOTE

        # Extra usefuls precomputed for joint limit
        self._g_ka_masked = self.g_ka[self.pos_lim_mask]
        self._pos_lim_masked = self.pos_lim[:,self.pos_lim_mask]
        self._masked_eye = np.eye(self.n_joints, dtype=self.dtype)[self.pos_lim_mask]

        # Precompute some joint limit gradients
        self._grad_qpos_peak = (0.5 * self._g_ka_masked * self.t_plan**2).reshape(-1,1) * self._masked_eye
        self._grad_qvel_peak = np.diag(self.g_ka * self.t_plan)
        self._grad_qpos_brake = (0.5 * self._g_ka_masked * self.t_plan * self.t_full).reshape(-1,1) * self._masked_eye

        # Basic cost weighting per joint, but maybe calculate this using actual distances
        if weight_joint_costs:
            self._joint_cost_weights = np.linspace(1, 0, num=n_joints, endpoint=False)
        else:
            self._joint_cost_weights = 1

        # NOTE:
        self.H_pred = H_pred # prediction hrizon 

    def reset(self, qpos, qvel, qgoal, additional_constraints = [], cons_val = 0, t_final_thereshold=0., qdgoal = None):
        # Key parameters for optimization
        self.p_init = qpos
        self.v_init = qvel
        self.qgoal = qgoal
        self.qdgoal = qdgoal
        self.use_t_final = np.linalg.norm(self.qpos - self.qgoal) < t_final_thereshold
        # FO constraint holder & functional
        if not isinstance(additional_constraints, list):
            additional_constraints = [additional_constraints]
        self._additional_constraints = additional_constraints
        self._cons_val = cons_val

        self.M = self.M_limits
        for constraints in additional_constraints:
            self.M += int(constraints.M)
        
        # Prepare the rest of the joint limit constraints
        self._qpos_masked = self.qpos[self.pos_lim_mask]
        self._qvel_masked = self.qvel[self.pos_lim_mask]

        # Constraints and Jacobians
        self._Cons = np.zeros(self.M, dtype=self.dtype)
        self._Jac = np.zeros((self.M, self.n_joints), dtype=self.dtype)

        # Internal
        self._Z_prev = np.zeros(self.n_joints)*np.nan
        self._constraint_times = []
        self.P_next, self.V_next, self.Ka, self.P, self.V = None  

        # IPOPT stats
        self.num_constraint_evaluations = 0
        self.num_jacobian_evaluations = 0

    def objective(self, Z):        
        P_next = Z[:self.n_joints*self.H_pred].reshape(self.H_pred, self.n_joints)
        p_obj = np.sum(self._joint_cost_weights*self._wrap_cont_joints(P_next - self.qgoal)**2)
        return p_obj 

    def gradient(self, Z):
        P_next = Z[:self.n_joints*self.H_pred].reshape(self.H_pred, self.n_joints)
        p_obj_grad = np.zeros_like(Z)
        p_obj_grad[:self.n_joints*self.H_pred] = 2*self._joint_cost_weights*self._wrap_cont_joints(P_next - self.qgoal).flatten()
        return p_obj_grad 

    def constraints(self, Z):
        self.num_constraint_evaluations += 1
        self.compute_constraints(Z)
        return self._Cons

    def jacobian(self, Z):
        self.num_jacobian_evaluations += 1
        self.compute_constraints(Z)
        return self._Jac

    def compute_constraints(self,Z):
        if (self._Z_prev!=Z).any():
            start = time.perf_counter()
            self._Z_prev = np.copy(Z)

            # zero out the underlying constraints and jacobians
            self._Cons[...] = self._cons_val
            self._Jac[...] = 0

            # Parse Decision Varaibles
            self._parse_decision_variables(Z)

            # Joint limits
            self._constraints_limits(Z, Cons_out=self._Cons[:self.M_limits], Jac_out=self._Jac[:self.M_limits])

            # Additional constraints as desired
            M_start = self.M_limits
            for constraints in self._additional_constraints:
                if constraints.M <= 0:
                    continue
                M_end = M_start + constraints.M
                constraints(Z, Cons_out=self._Cons[M_start:M_end], Jac_out=self._Jac[M_start:M_end])
                M_start = M_end
            
            # Timing
            self._constraint_times.append(time.perf_counter() - start)

    def _constraints_state_transition(self, Z, Cons_out=None, Jac_out=None):
        # assume perfect traking 


        hnj = self.H_pred*self.n_joints


        if Cons_out is None:
            Cons_out = np.empty(2*hnj, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.zeros((2*hnj, 3*hnj), dtype=self.dtype)        

        '''
        # Alternative to write down joint position transition
        (P + V*self.t_plan + 0.5*self.g_ka*self.t_plan**2*Ka - P_next).flatten()
        
        '''
        
        # Transiton for Joint Position
        Cons_out[:hnj] = (self.P + 0.5*self.t_plan*(self.V+self.V_next) - self.P_next).flatten()
        Jac_out[range(hnj),range(hnj)] = -1 
        Jac_out[range(self.n_joints,hnj),range(hnj-self.n_joints)] = 1
        Jac_out[range(hnj),range(hnj,2*hnj)] = 0.5*self.t_plan
        Jac_out[range(self.n_joints,hnj),range(hnj,2*hnj-self.n_joints)] = 0.5*self.t_plan 

        # Transiton for Joint Velocity
        Cons_out[hnj:2*hnj] = (self.V + self.g_ka * self.Ka * self.t_plan - self.V_next).flatten()
        Jac_out[range(hnj,2*hnj),range(hnj,2*hnj)] = -1
        Jac_out[range(hnj+self.n_joints,2*hnj),range(hnj,2*hnj-self.n_joints)] = 1
        Jac_out[range(hnj,2*hnj),range(2*hnj,3*hnj)] = self.g_ka * self.t_plan   
    
        return Cons_out, Jac_out 
    


    def _constraints_limits(self, x, Cons_out=None, Jac_out=None):
        ka = x # is this numpy? we will see
        if Cons_out is None:
            Cons_out = np.empty(self.M_limits, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M_limits, self.n_joints), dtype=self.dtype)

        ## position and velocity constraints
        # disable warnings for this section
        settings = np.seterr('ignore')
        # scake k and get the part relevant to the constrained positions
        scaled_k = self.g_ka*ka
        scaled_k_masked = scaled_k[self.pos_lim_mask]
        # time to optimum of first half traj.
        t_peak_optimum = -self._qvel_masked/scaled_k_masked
        # if t_peak_optimum is in the time, qpos_peak_optimum has a value
        t_peak_in_range = (t_peak_optimum > 0) * (t_peak_optimum < self.t_plan)
        # Get the position and gradient at the peak
        qpos_peak_optimum = np.nan_to_num(t_peak_in_range * (self._qpos_masked + self._qvel_masked * t_peak_optimum + 0.5 * scaled_k_masked * t_peak_optimum**2))
        grad_qpos_peak_optimum = np.nan_to_num(t_peak_in_range * 0.5*self._qvel_masked**2/(scaled_k_masked**2)).reshape(-1,1) * self._masked_eye
        # restore
        np.seterr(**settings)

        ## Position and velocity at velocity peak of trajectory
        qpos_peak = self._qpos_masked + self._qvel_masked * self.t_plan + 0.5 * scaled_k_masked * self.t_plan**2
        qvel_peak = self.qvel + scaled_k * self.t_plan

        ## Position at braking
        # braking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
        # qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*braking_accel*(T_FULL-T_PLAN)**2
        # NOTE: swapped to simplified form
        qpos_brake = self._qpos_masked + 0.5 * self._qvel_masked * (self.t_full + self.t_plan) + 0.5 * scaled_k_masked * self.t_plan * self.t_full
        
        ## compute the final constraint values and store them to the desired output arrays
        qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))
        qpos_ub = (qpos_possible_max_min - self._pos_lim_masked[1]).flatten()
        qpos_lb = (self._pos_lim_masked[0] - qpos_possible_max_min).flatten()
        qvel_ub = qvel_peak - self.vel_lim
        qvel_lb = (-self.vel_lim) - qvel_peak
        np.concatenate((qpos_ub, qpos_lb, qvel_ub, qvel_lb), out=Cons_out)

        ## Do the same for the gradients
        grad_qpos_ub = np.vstack((grad_qpos_peak_optimum,self._grad_qpos_peak,self._grad_qpos_brake))
        grad_qpos_lb = -grad_qpos_ub
        grad_qvel_ub = self._grad_qvel_peak
        grad_qvel_lb = -self._grad_qvel_peak
        np.concatenate((grad_qpos_ub, grad_qpos_lb, grad_qvel_ub, grad_qvel_lb), out=Jac_out)

        return Cons_out, Jac_out
    
    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    def _parse_decision_variables(self,Z):
        # Z = [ p1,p2,...,pH_pred | v1,v2,...,vH_pred | u1,u2,...,uH_pred] 
        #   = [ P_next | V_next | U_curr ]
        # shape (n_joint * H_pred * 3, )
        self.P_next, self.V_next, self.Ka = np.split(Z.reshape(2*self.H_pred, self.n_joints),[self.H_pred, 2*self.H_pred])
        self.P = np.vstack((self.p_init,self.P_next))[:-1]
        self.V = np.vstack((self.v_init,self.V_next))[:-1] 

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass

    @property
    def constraint_times(self):
        return self._constraint_times