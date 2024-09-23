time_limit=0.5
solver="ma27"

python run_scenario_planning_3d.py --planner crows --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver --hlp --tol 0.001 --confidence_idx 2
python run_scenario_planning_3d.py --planner crows --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver --hlp --tol 0.001 --confidence_idx 2 --not_use_learned_grad
python run_scenario_planning_3d.py --planner sphere --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver --hlp --tol 0.001 
python run_scenario_planning_3d.py --planner armtd --save_success --time_limit $time_limit --t_final_thereshold 0.2 --detail --solver $solver --hlp --tol 0.001 
