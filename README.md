# CROWS: Conformalized Reachable Sets for Obstacle Avoidance With Spheres

[Project Page](https://roahmlab.github.io/crows/) | Paper (Comming Soon!) | [Dataset](https://drive.google.com/drive/folders/1y82zpWuKaZmejr7AXxctpPKblB2tOSW9?usp=sharing)
## Introduction
This is the code for CROWS: Conformalized Reachable Sets for Obstacle Avoidance With Spheres

CROWS is a real-time, receding horizon trajectory planner that generates *probabilitically-safe* motion plans based on a neural network-based representation of a spherebased reachable set. 

We demonstrate that CROWS outperforms a variety of state-of-the-art methods in solving challenging motion planning tasks in cluttered environments while remaining collision-free.

## Dependency 
To set up the Python environment, you can install the required dependencies using [conda](https://www.anaconda.com) by following these steps:
```bash
conda env create --file environment.yaml
conda activate sparrows
```
Note: Solving the conda environment can take up to 15 minutes. If you don't have access to a CUDA-capable device, use the environment-cpu.yaml file to install the CPU-only version of PyTorch.

For MACOS, you will need to modify the cpu environment to remove `open3d`
and then build and install `python-fcl` and `open3d` from source manually after activating the environment.

Note that the environment includes:
- [zonopy](https://github.com/roahmlab/zonopy) which provides functionalities used for reachability analysis
- [zonopy-robots](https://github.com/roahmlab/zonopy-robots) which provides functionalities for specifying and loading robots

[MATLAB](https://matlab.mathworks.com) and [CORA 2021](https://tumcps.github.io/CORA/) are used to compute Joint Reachable Set in `forward_occupancy/jrs_trig/gen_jrs_trig` with the provided MATLAB scripts.

## Reproducing Results
### Device Allocation
The `--device` argument specifies the device used for [PyTorch](https://pytorch.org/) computations. If a GPU is available, the default setting is `--device 0`, which uses `cuda:0`. If only a CPU is available, the default is `--device -1`, which sets the computation to `cpu`. For systems with multiple GPUs, you can manually select the device, e.g., `--device 2` to use `cuda:2`.

### Generating Dataset
You can download the pre-generated dataset from [Google Drive](https://drive.google.com/drive/folders/1y82zpWuKaZmejr7AXxctpPKblB2tOSW9?usp=sharing), or you can generate the dataset yourself using the following commands:
```python
# Generate dataset for spherical occupancy
python data_processing/generate_dataset.py --batch_size 500  
# Generate dataset for gradients of centers w.r.t. trajectory parameters
python data_processing/generate_dataset.py --batch_size 500 --grad  
```
You may need to adjust the `--batch_size` depending on your machine's capacity. A `--batch_size` of 500 requires approximately 1.4 GB of GPU memory. Increasing the batch size may speed up data generation until you reach a size of 500.

### Training CROWS Models
Pretrained models are available in the `trained_models/` directory. If you prefer to train your own model, you can run the following commands:
```python
# Train CROWS model for spherical occupancy prediction
python train_model_3d.py --num_epochs 30 --wandb crows --num_workers 5  
# Train CROWS gradient model for center gradients w.r.t. trajectory parameters
python train_model_3d.py --num_epochs 30 --wandb crows --num_workers 5 --train_grad  
```
You would need to adjust the `--num_workers` option according to the number of available CPU threads on your machine. The `--wandb` option uses [Weights & Biases](https://wandb.ai/home) for tracking the training process. If you do not want to use it, simply omit the `--wandb` option.

### Running Planning with a random obstacle example
To run a single trial of CROWS with 20 obstacles and a 0.5-second time limit, use the following command:
```python
python run_statistics_planning_3d.py --planner crows --n_obs 20 --time_limit 0.5 --n_envs 1 --video
```
The `--video` argument enables video rendering, and the generated video will be saved in the `planning_videos/` directory.


### Running Planning Experiments for CROWS, SPARROWS, and ARMTD

CROWS uses [IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html), a nonlinear program solver, and the python iterface [cyipopt](https://cyipopt.readthedocs.io/en/stable/). 
Experiments in the paper are run with MA27 (`ma27`) linear solver, which requires to install [HSL](https://github.com/coin-or-tools/ThirdParty-HSL). 

- [IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html) and [cyipopt](https://cyipopt.readthedocs.io/en/stable/) will be install as part of the environment setup of `environment.yaml` or `environment-cpu.yaml`.
- For the MA27 (`ma27`) linear solver, install [HSL](https://github.com/coin-or-tools/ThirdParty-HSL), seperately. For detailed installation instructions, refer to [this guide](docs/how_to_install_HSL).
- If the MA27 (`ma27`) linear solver is unavailable, you can use MUMPS (`mumps`) but performance may be lower. You can manually specify the linear solver with the `--solver` option

To reproduce the single-arm planning experiments on random scenarios, run `bash run_3d_planning.sh`. The results will be in `planning_results/` as generated by the planning program.

To reproduce the single-arm planning experiments on hard scenarios, run `bash run_scenario_planning.sh`. The results will be in `scenario_planning_results/` as generated by the planning program.

### Recreating Figures

We generate our figures using a pipeline to export base Blender files which we then add cameras or materials to.
Since the Blender code adds quite a bit of bloat, we keep it separate.
Please clone the `blender-color` branch to use those visualization environments.

## Credits
- [`zonopy`](https://github.com/roahmlab/zonopy) referred some part of [CORA](https://tumcps.github.io/CORA/).
- This code is built upon [`sparrows`](https://github.com/roahmlab/sparrows), which provides the core architecture and planning framework that CROWS extends and improves upon.

## Citation
```bibtex
@article{kwon2024crows,
  title={Conformalized Reachable Sets for Obstacle Avoidance With Spheres},
  author={Yongseok Kwon and Jonathan Michaux and Ram Vasudevan},
  journal={},
  year={2024},
  volume={},
  url={}}
```