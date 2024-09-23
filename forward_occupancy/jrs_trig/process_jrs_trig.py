# TODO VALIDATE

import torch
from zonopy import batchZonotope

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from load_jrs_trig import JRS_KEY

from zonopy import gen_rotatotope_from_jrs_trig, gen_batch_rotatotope_from_jrs_trig

T_fail_safe = 0.5

cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

def process_batch_JRS_trig(jrs_tensor, q_0,qd_0,joint_axes):
    dtype, device = jrs_tensor.dtype, jrs_tensor.device 
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        JRS_batch_zono = batchZonotope(jrs_tensor[closest_idx])
        c_qpos = torch.cos(q_0[i])
        s_qpos = torch.sin(q_0[i])
        Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=dtype,device=device)
        A = torch.block_diag(Rot_qpos,torch.eye(4,dtype=dtype,device=device))
        JRS_batch_zono = A@JRS_batch_zono.slice(kv_dim,qd_0[i:i+1])
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim, id=i)
        '''
        delta_k = PZ_JRS.G[0,0,ka_dim]
        c_breaking = - qd_0[i]/T_fail_safe
        delta_breaking = - delta_k/T_fail_safe
        PZ_JRS.c[50:,acc_dim] = c_breaking
        PZ_JRS.G[50:,0,acc_dim] = delta_breaking
        '''
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])

        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)
    return PZ_JRS_batch, R_batch

# batched over time and initial condition
def process_batch_JRS_trig_ic(jrs_tensor,q_0,qd_0,joint_axes):
    dtype, device = jrs_tensor.dtype, jrs_tensor.device 
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[:,i:i+1]-jrs_key),dim=-1)
        JRS_batch_zono = batchZonotope(jrs_tensor[closest_idx])
        c_qpos = torch.cos(q_0[:,i:i+1]).unsqueeze(-1)
        s_qpos = torch.sin(q_0[:,i:i+1]).unsqueeze(-1)
        A = (c_qpos*torch.tensor([[1.0]+[0]*5,[0,1]+[0]*4]+[[0]*6]*4,dtype=dtype,device=device) 
            + s_qpos*torch.tensor([[0,-1]+[0]*4,[1]+[0]*5]+[[0]*6]*4,dtype=dtype,device=device) 
            + torch.tensor([[0.0]*6]*2+[[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=dtype,device=device))
        
        JRS_batch_zono = A.unsqueeze(1)@JRS_batch_zono.slice(kv_dim,qd_0[:,i:i+1].unsqueeze(1).repeat(1,100,1))
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim, id=i)
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])
        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)

    return PZ_JRS_batch, R_batch



if __name__ == '__main__':
    # Validate parallel reachable set 
    import zonopy as zp
    import zonopyrobots as zpr 

    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    from forward_occupancy.JRS import OfflineJRS
    from forward_occupancy.SO import sphere_occupancy

    # Set the random seed for reproducibility
    torch.manual_seed(0)
    # number of samples 
    n_samples = 500
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype =torch.float32

    # Disable debugging visualization
    zpr.DEBUG_VIZ = False

    # Load the robot model
    zpr_dirname = os.path.dirname(zpr.__file__)
    robot_path = 'robots/assets/robots/kinova_arm/gen3.urdf'
    robot = zpr.ZonoArmRobot.load(os.path.join(zpr_dirname,robot_path), create_joint_occupancy=True, dtype=dtype, device=device)

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

    # Setup robot-specific parameters including degrees of freedom (DOF), joint axes, position limits, velocity limits, and continuous joint properties.
    pos_lim = robot.pos_lim.nan_to_num(posinf=torch.pi, neginf=-torch.pi).to(dtype=dtype, device=device)
    pos_max = pos_lim[1]
    pos_min = pos_lim[0]
    vel_max = robot.vel_lim.to(dtype=dtype, device=device)

    # Generate random joint positions, velocities, and trajectory parameters 
    qpos = (pos_max - pos_min) * torch.rand((n_samples, robot.dof),dtype=dtype, device=device) + pos_min 
    qvel = vel_max * torch.rand((n_samples, robot.dof),dtype=dtype, device=device) 
    traj_param = 2 * torch.rand((n_samples, robot.dof),dtype=dtype, device=device) - 1 # Range [-1, 1]

    # Load OfflineJRS 
    JRS = OfflineJRS(dtype=dtype, device=device)
    n_timesteps = JRS.jrs_tensor.shape[1]

    ### Parallelized Occupancy Computation
    _, JRS_R = JRS(qpos, qvel, robot.joint_axis)
    # Compute spherical occupancy based on the joint reachable set
    joint_occ, _, _ = sphere_occupancy(JRS_R, robot, zono_order=2, joint_radius_override=joint_radius_override)
    
    # Stack the occupancy centers and radii
    batched_centers_bpz = zp.stack([pz for pz, _ in joint_occ.values()])
    batched_radii1 = torch.stack([r for _, r in joint_occ.values()]).permute(1,2,0) # n_joints, timesteps 
    
    # Slice centers based on the trajectory parameters
    batched_traj_param = traj_param.unsqueeze(1).repeat((1,n_timesteps,1))
    batched_centers1 = batched_centers_bpz.center_slice_all_dep(batched_traj_param).permute(1,2,0,3) # n_joints, timesteps, dimension


    ### Non-parallel Occupancy Computation
    batched_radii2 = torch.zeros_like(batched_radii1)
    batched_centers2 = torch.zeros_like(batched_centers1)
    for idx in range(n_samples):
        ### Parallelized Occupancy Computation
        _, jrs_r = JRS(qpos[idx], qvel[idx], robot.joint_axis)
        # Compute spherical occupancy based on the joint reachable set
        joint_occ, _, _ = sphere_occupancy(jrs_r, robot, zono_order=2, joint_radius_override=joint_radius_override)
        
        # Stack the occupancy centers and radii
        centers_bpz = zp.stack([pz for pz, _ in joint_occ.values()])
        radii = torch.stack([r for _, r in joint_occ.values()]) # n_joints, timesteps 
        
        # Slice centers based on the trajectory parameters
        centers = centers_bpz.center_slice_all_dep(traj_param[idx]) # n_joints, timesteps, dimension   

        batched_radii2[idx] = radii.transpose(0,1) 
        batched_centers2[idx] = centers.transpose(0,1)

    eps = 1e-6
    count = 0
    max_sphere_dev = 0.0
    for idx in range(n_samples):
        radii_diff = torch.abs(batched_radii1[idx] - batched_radii2[idx])
        centers_dist = torch.norm(batched_centers1[idx] - batched_centers2[idx],dim=-1)
        max_sphere_dev = max(max_sphere_dev,(centers_dist+radii_diff).max().item())
        if  max_sphere_dev > eps:
            print(f'{idx}-initial_condition: sphere deviation is too large!')
            count += 1

    print(f'Max. sphere dev. is {max_sphere_dev}')
    if count == 0: 
        print(f'All {n_samples} samples are correctly validated.')
        

