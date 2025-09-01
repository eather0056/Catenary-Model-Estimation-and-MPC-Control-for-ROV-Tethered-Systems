# Catenary_Dynamic

This repository contains code and scripts for the modeling, simulation, and validation of catenary dynamics in tethered underwater robot systems. It includes data processing pipelines, symbolic regression for dynamics learning, RK4 simulations, and visualization tools.

## 📁 Repository Structure

- `main_fun.py` — Core functions used throughout the project.
- `catenary.py` — Catenary model implementations.
- `simulate_theta_gamma.py` / `simulate_rk4_theta_gamma.py` — RK4-based simulators for angular dynamics.
- `dynamic_eq_*.py` — Cluster and local scripts for symbolic dynamics estimation.
- `dd_cluster.py`, `dd_test_cluster.py` — Cluster-based PySR training scripts.
- `PySRTrainingScript.py` — Central training script using PySR.
- `catenary_validation.py` — Model validation pipeline.
- `velocity_transform*.py` — Velocity frame transformation utilities.
- `batch_correct_velocity.py` — Applies batch velocity corrections.
- `Rov_traj_gen.py` — Generates ROV trajectory scenarios.
- `Experiment_Movements*.py` — Experimental movement scenarios and visualizations.
- `outputs/` — Symbolic models and logs saved from training runs.
- `Results/` — Model results, animations, and trajectory visualizations.
- `Data/` — Experimental and simulation datasets.
- `lagrangian/` — Contains scripts for hybrid symbolic Lagrangian modeling.
- `wandb/` — Weights & Biases experiment logs.
- `README.md`, `LICENSE` — Documentation and license.

## 🧪 Key Functionalities

- **Symbolic Regression (PySR):** Data-driven discovery of angular dynamics (\(\dot{\theta}, \dot{\gamma}, \ddot{\theta}, \ddot{\gamma}\)) using time-series data and regression pipelines.
- **Catenary Modeling:** Implements classical and augmented catenary models.
- **Simulation Tools:** RK4-based simulators to predict tethered robot trajectories using learned models.
- **Velocity Correction:** Coordinate transformation and Kabsch algorithm for aligning velocity data.
- **Data Visualization:** Includes animation and plotting tools for validation and experiment rendering.

## 🧠 Core Objectives

- Identify and validate interpretable equations for tether angular dynamics.
- Use symbolic models in predictive simulation pipelines.
- Integrate learned dynamics into control-oriented architectures.

## 🚀 How to Run

> Note: This repo is designed for a high-performance computing (HPC) cluster using conda environment `cad`.

1. **Setup Environment**:
   ```bash
   conda activate cad
   pip install -r requirements.txt  # if available
    ```

2. **Train Symbolic Models**:

   ```bash
   python dd_cluster.py  # or dynamic_eq_cluster.py
   ```

3. **Validate Models:**

   ```bash
   python catenary_validation.py
   ```

4. **Simulate Predicted Trajectories:**

   ```bash
   python simulate_theta_gamma.py
   ```

5. **Correct Velocities (Optional):**

   ```bash
   python batch_correct_velocity.py
   ```

## 📊 Output Format

All generated outputs (models, plots, logs) are saved in:

* `outputs/` — PySR model artifacts (pkl, txt, csv)
* `Results/` — Visualizations and animated simulations


## 📎 Acknowledgements

This work is conducted at COSMER Laboratory (EA 7398) and Laboratoire d’Informatique et Systemes (LIS, UMR CNRS 7020), Université de Toulon, supported by various cluster compute nodes and experimental setups.

**Author**: [MD Ether Deowan](mailto:mdeowan698@sms-cluster), Nicolas BOIZOT


