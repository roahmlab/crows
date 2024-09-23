import torch
import argparse
import numpy as np
import time
import csv
from scipy.io import loadmat, matlab
import json
import random
from tqdm import tqdm
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from environments.fullstep_recorder import FullStepRecorder
from planning.armtd.armtd_3d_urdf import ARMTD_3D_planner
from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner
from planning.crows.crows_urdf import CROWS_3D_planner
from planning.common.waypoints import GoalWaypointGenerator, CustomWaypointGenerator
from visualizations.fo_viz import FOViz
from visualizations.sphere_viz import SpherePlannerViz
import os
T_PLAN, T_FULL = 0.5, 1.0

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def evaluate_planner(planner, 
                     planner_name='sphere', 
                     env_indices=range(0,100), 
                     n_steps=150, 
                     n_links=7, 
                     n_obs=5, 
                     save_success_trial_id=False, 
                     video=False, 
                     reachset_viz=False, 
                     time_limit=0.5, 
                     detail=True, 
                     t_final_thereshold=0.,
                     check_self_collision=False,
                     use_hlp=False,
                     tol = 1e-5,
                    ):
    t_armtd = 0.0
    num_success = 0
    num_collision = 0
    num_stuck = 0
    num_no_solution = 0
    num_step = 0
    t_armtd_list = []
    t_success_list = []
    n_envs = len(env_indices)
    
    planner_stats = {}
    if save_success_trial_id:
        success_episodes = []
    if video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        video_folder = f'scenario_planning_videos/{planner_name}'
        if reachset_viz:
            video_folder += '_reachset'
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
    if detail:
        import pickle
        planning_details = {}
        trial_details = {}
    
    for i_env in tqdm(env_indices):
        dir = 'kinova_scenarios'
        scene_name = f'{dir}/scene_{str(i_env).zfill(3)}.csv'
        obstacle_centers = []
        with open(scene_name, mode ='r') as file:   
            csvFile = csv.reader(file)
            line_number = 0
            for line in csvFile:
                    if line_number == 0:
                        qstart = np.array([float(num) for num in line if num != 'NaN'])
                    elif line_number == 1:
                        qgoal = np.array([float(num) for num in line])
                    elif line_number > 2:
                        obstacle_centers.append([float(num) for num in line][:3])
                    line_number += 1
            n_obs = len(obstacle_centers)
        
        env_args = dict(
            step_type='integration',
            check_joint_limits=True,
            check_self_collision=check_self_collision,
            use_bb_collision=False,
            render_mesh=True,
            reopen_on_close=False,
            obs_size_min = [0.2,0.2,0.2],
            obs_size_max = [0.2,0.2,0.2],
            n_obs=n_obs,
            renderer = 'pyrender-offscreen',
            info_nearest_obstacle_dist = False,
            obs_gen_buffer = 0.01
        )
        env = KinematicUrdfWithObstacles(
                robot=rob.urdf,
                **env_args
            )
        if video and reachset_viz:
            if 'sphere' in planner_name or 'crows' in planner_name:
                viz = SpherePlannerViz(planner, plot_full_set=True, t_full=T_FULL)
            elif 'armtd' in planner_name:
                viz = FOViz(planner, plot_full_set=True, t_full=T_FULL)
            else:
                raise NotImplementedError(f"Visualizer for {planner_name} type has not been implemented yet.")
            env.add_render_callback('spheres', viz.render_callback, needs_time=False)
        
        obs = env.reset(
                qpos = qstart, 
                qvel = np.zeros_like(qstart), 
                qgoal = qgoal, 
                obs_pos = obstacle_centers,
        )
        
        ### Load waypoints
        if use_hlp:
            mat_filename = os.path.join(dir, 'waypoints', f'traj_data_{i_env}.mat')
            mat_data = loadmat(mat_filename)
            waypoints = mat_data['pos']        
            waypoint_generator = CustomWaypointGenerator(waypoints, qgoal, planner.osc_rad*3)
        else:
            waypoint_generator = GoalWaypointGenerator(qgoal, planner.osc_rad*3)
        if detail:
            planning_details[i_env] = {
                'initial': obs,
                'trajectory': {
                    'k': [],
                    'flag': [],
                    'nearest_distance': []
                }
            }
        t_curr_trial = []
        if video:
            video_path = os.path.join(video_folder, f'video{i_env}.mp4')
            video_recorder = FullStepRecorder(env, path=video_path)
        was_stuck = False
        force_fail_safe = False
        
        for i_step in range(n_steps):
            qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
            obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
            waypoint = waypoint_generator.get_waypoint(qpos, qvel)
            ts = time.time()
            ka, flag, planner_stat = planner.plan(qpos, qvel, waypoint, obstacles, time_limit=time_limit, t_final_thereshold=t_final_thereshold, tol=tol)   
            t_elasped = time.time()-ts
            t_armtd += t_elasped
            t_armtd_list.append(t_elasped)
            t_curr_trial.append(t_elasped)
            
            for key in planner_stat:
                if planner_stat[key] is None:
                    continue      
                if key in planner_stats:
                    if isinstance(planner_stat[key], list):
                        planner_stats[key] += planner_stat[key]
                    else:
                        planner_stats[key].append(planner_stat[key])
                else:
                    if isinstance(planner_stat[key], list):
                        planner_stats[key] = planner_stat[key]
                    else:
                        planner_stats[key] = [planner_stat[key]]

            if flag != 0:
                ka = (0 - qvel)/(T_FULL - T_PLAN)

            if force_fail_safe:
                ka = (0 - qvel)/(T_FULL - T_PLAN)
                force_fail_safe = False   
            else:
                force_fail_safe = (flag == 0) and planner.nlp_problem_obj.use_t_final and (np.sqrt(planner.final_cost) < env.goal_threshold)
                
            if video and reachset_viz:
                if flag == 0:
                    viz.set_ka(ka)
                else:
                    viz.set_ka(None)
            obs, reward, done, info = env.step(ka)
            if video:
                video_recorder.capture_frame()
            if detail:
                planning_details[i_env]['trajectory']['k'].append(ka)
                planning_details[i_env]['trajectory']['flag'].append(flag)
            
            num_step += 1
            if info['collision_info']['in_collision']:
                num_collision += 1
                break
            elif reward == 1:
                num_success += 1
                t_success_list += t_curr_trial
                if save_success_trial_id:
                    success_episodes.append(i_env)
                break
            elif done:
                break

            if flag != 0:
                if was_stuck:
                    num_step -= 1
                    num_stuck += 1
                    break
                else:
                    was_stuck = True
                    if flag > 0 or flag == -5:
                        num_no_solution += 1
            else:
                was_stuck = False
        if detail:
            trial_details[i_env] = {'success': reward == 1, 'length': i_step+1, 'collision': info['collision_info']['in_collision']}
            planning_details[i_env].update(
                trial_details[i_env]
            )
        if video:
            video_recorder.close()
        

    planner_stats_summary = {}
    for key in planner_stats:
        planner_stats_summary[key] = {
            'mean': np.mean(planner_stats[key]),
            'std': np.std(planner_stats[key]),
            'max': float(np.max(planner_stats[key]))
        }
    stats = {
        'planner': planner_name,
        'n_trials': n_envs,
        'n_links': n_links,
        'n_obs':n_obs,
        'time_limit': time_limit,
        't_final_thereshold': t_final_thereshold,
        'num_success': num_success,
        'num_collision': num_collision,
        'num_stuck': num_stuck, 
        'mean planning time': np.mean(np.array(t_armtd_list)),
        'std planning time': np.std(np.array(t_armtd_list)),
        "use_hlp": use_hlp,
        'mean planning time for success trials': np.mean(np.array(t_success_list)),
        'std planning time for success trials': np.std(np.array(t_success_list)),
        'total planning time': t_armtd,
        'num_no_solution': num_no_solution,
        'num_step': num_step,
        'planner_stats': planner_stats_summary,
    } 
    if detail:
        stats.update({'trial_details': trial_details})
    stats.update({'env_args': env_args})
        
    with open(f"scenario_planning_results/{planner_name}_stats_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.json", 'w') as f:
        if save_success_trial_id:
            stats['success_episodes'] = success_episodes
        json.dump(stats, f, indent=2)
    if detail:
        with open(f"scenario_planning_results/{planner_name}_stats_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.pkl", 'wb') as f:
            pickle.dump(planning_details, f)
        
    return stats
    
def read_params():
    parser = argparse.ArgumentParser(description="Scenario Planning")
    # general setting
    parser.add_argument("--planner", type=str, default="sphere") # "armtd", "sphere", "rdf"
    parser.add_argument('--robot_type', type=str, default="branched")
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_obs', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--device', type=int, default=0 if torch.cuda.is_available() else -1, choices=range(-1,torch.cuda.device_count())) # Designate which cuda to use, default: cpu
    parser.add_argument('--dtype', type=int, default=32)

    # visualization settings
    parser.add_argument('--video',  action='store_true')
    parser.add_argument('--reachset',  action='store_true')
    
    # optimization info
    parser.add_argument('--num_spheres', type=int, default=5)
    parser.add_argument('--time_limit',  type=float, default=1e20)
    parser.add_argument('--t_final_thereshold', type=float, default=0.2)
    parser.add_argument('--hlp', action='store_true')
    parser.add_argument('--solver', type=str, default="ma27")
    parser.add_argument('--tol', type=float, default=1e-3) # desired convergence tolerance for IPOPT solver

    # results info
    parser.add_argument('--save_success', action='store_true') # whether to save success trial id
    parser.add_argument('--detail', action='store_true') # whether to save trajetcory detail

    # CROWS
    parser.add_argument('--not_use_learned_grad', action='store_true') # whether to not use learned gradient for CROWS
    parser.add_argument('--confidence_idx', type=int, default=2) #option for confidence level of CROWS model uncertainty -> {idx:epsilon_hat}, 0: 99.999%, 1: 99.99%, 2: 99.9%, 3: 99% 4: 90% 5:80%
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    
    params = read_params()
    planner_name = params.planner
    # Set device
    device = torch.device('cpu') if params.device <0 else torch.device(f'cuda:{params.device}')
    # Set dtype
    assert params.dtype ==32 or params.dtype == 64
    dtype = torch.float32 if params.dtype == 32 else torch.float64

    print(f"Running {planner_name} 3D{params.n_links}Links with {params.n_steps} step limit and {params.time_limit}s time limit each step")
    print(f"Using device {device}")
    
    planning_result_dir = f'scenario_planning_results/'
    if not os.path.exists(planning_result_dir):
        os.makedirs(planning_result_dir)
    
    stats = {}
    import zonopyrobots as robots2
    robots2.DEBUG_VIZ = False
    basedirname = os.path.dirname(robots2.__file__)
    robot_path = 'robots/assets/robots/kinova_arm/gen3.urdf'
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, robot_path), dtype = dtype, device=device, create_joint_occupancy=True)

    if planner_name == 'armtd':
        planner = ARMTD_3D_planner(
            rob, 
            dtype = dtype,
            device=device,
            linear_solver=params.solver,
        )
        hlp_identifier = '_HLP' if params.hlp else ''
        stats['armtd'] = evaluate_planner(
            planner=planner,
            planner_name=f'armtd{hlp_identifier}_{params.robot_type}_t{params.time_limit}_tol{params.tol}', 
            env_indices=range(1,15), 
            n_steps=params.n_steps, 
            n_links=params.n_links, 
            n_obs=params.n_obs,
            save_success_trial_id=params.save_success, 
            video=params.video,
            reachset_viz=params.reachset,
            time_limit=params.time_limit,
            detail=params.detail,
            t_final_thereshold=params.t_final_thereshold,
            use_hlp=params.hlp,
            tol=params.tol,
        )
    if planner_name == 'sphere' or planner_name == 'crows':
        joint_radius_override = {
                'joint_1': torch.tensor(0.0503305, dtype=torch.float, device=device),
                'joint_2': torch.tensor(0.0630855, dtype=torch.float, device=device),
                'joint_3': torch.tensor(0.0463565, dtype=torch.float, device=device),
                'joint_4': torch.tensor(0.0634475, dtype=torch.float, device=device),
                'joint_5': torch.tensor(0.0352165, dtype=torch.float, device=device),
                'joint_6': torch.tensor(0.0542545, dtype=torch.float, device=device),
                'joint_7': torch.tensor(0.0364255, dtype=torch.float, device=device),
                'end_effector': torch.tensor(0.0394685, dtype=torch.float, device=device),
            }

        if planner_name == 'sphere':
            planner = SPARROWS_3D_planner(
                rob, 
                dtype = dtype,
                device=device, 
                sphere_device=device, 
                spheres_per_link=params.num_spheres,
                joint_radius_override=joint_radius_override,
                linear_solver=params.solver,
            )
            hlp_identifier = '_HLP' if params.hlp else ''
            stats['spheres_armtd'] = evaluate_planner(
                planner=planner,
                planner_name=f'sphere{hlp_identifier}_{params.robot_type}_t{params.time_limit}_tol{params.tol}', 
                env_indices=range(1,15), 
                n_steps=params.n_steps, 
                n_links=params.n_links, 
                n_obs=params.n_obs,
                save_success_trial_id=params.save_success, 
                video=params.video,
                reachset_viz=params.reachset,
                time_limit=params.time_limit,
                detail=params.detail,
                t_final_thereshold=params.t_final_thereshold,
                use_hlp=params.hlp,
                tol=params.tol,
            )
        else:
            model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')

            planner = CROWS_3D_planner(
                rob, 
                dtype = dtype,
                device=device, 
                sphere_device=device, 
                spheres_per_link=params.num_spheres,
                joint_radius_override=joint_radius_override,
                linear_solver=params.solver,
                model_dir = model_dir,
                use_learned_grad = not params.not_use_learned_grad,
                confidence_idx = params.confidence_idx
            )
            learned_grad_identifier = '' if params.not_use_learned_grad else '_LG'
            hlp_identifier = '_HLP' if params.hlp else ''
            stats['spheres_armtd'] = evaluate_planner(
                planner=planner,
                planner_name=f'crows_conf{params.confidence_idx}{learned_grad_identifier}{hlp_identifier}_{params.robot_type}_t{params.time_limit}_tol{params.tol}', 
                env_indices=range(1,15), 
                n_steps=params.n_steps, 
                n_links=params.n_links, 
                n_obs=params.n_obs,
                save_success_trial_id=params.save_success, 
                video=params.video,
                reachset_viz=params.reachset,
                time_limit=params.time_limit,
                detail=params.detail,
                t_final_thereshold=params.t_final_thereshold,
                use_hlp=params.hlp,
                tol=params.tol,
            )