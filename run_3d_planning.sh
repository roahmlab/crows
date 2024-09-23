time_limit=0.5 
n_obss=(10 20 40) 
n_envs=100
solver="ma27"

for n_obs in "${n_obss[@]}";do
    python run_statistics_planning_3d.py --planner crows --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envs --t_final_thereshold 0.2 --detail --solver $solver --tol 0.001 --confidence_idx 2
    python run_statistics_planning_3d.py --planner crows --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envs --t_final_thereshold 0.2 --detail --solver $solver --tol 0.001 --confidence_idx 2 --not_use_learned_grad
    python run_statistics_planning_3d.py --planner sphere --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envs --t_final_thereshold 0.2 --detail --solver $solver --tol 0.001
    python run_statistics_planning_3d.py --planner armtd --save_success --n_obs $n_obs --time_limit $time_limit --n_envs $n_envs  --t_final_thereshold 0.2 --detail --solver $solver --tol 0.001 
done

