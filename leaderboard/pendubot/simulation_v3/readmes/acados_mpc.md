# Nonlinear Model Predictive Control using Acados 

This controller is implemented using the acados framework https://docs.acados.org/
for acados problem formulation see https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

## simple setup
1.initialize controller with model parameters AcadosMpcController(model_pars=mpar)

2. set controller parameter with controller.set_parameters(...)

3. optionally.set cosntraints on velocity with controller.set_velocity_constraints(v_max=vmax, v_final=vf)

4. set cost matrices on x, f_final and u error with controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat

5. controller.init()

## controller options

| Attribute | Type | Default | Description |
| -------- | ------- | -------- | ------- |
| N_horizon | int | 20 | number of shooting nodes |
| prediction_horizon | float | 0.5 | prediction horizon |
| Nlp_max_iter | int | 500 | maximum number of NLP iterations |
| lstinline | max_solve_time | float | 1.0 | Maximum time before solver timeout |
| solver_type | string | SQP_RTI |in ("SQP", "DDP", "SQP-RTI") |
| wrap_angle | bool | 0.5 | wether or not angles bigger than 360 deg are translated to $\theta \mod 360$ | 
| lstinline | warm_start | bool | True | solver does some initial iterations to find a good initial guess |
| scaling | int[] | np.full(N_horizon, 1) | scaling for the cost on nodes 1-N |
| nonuniform_grid | bool | False | Timesteps $t_n$ are growing in size with there distance from $t_0$ |
| use_energy_for_terminal_cost | bool | False | wether in the terminal cost the energy is used instead of the state x |
| fallback_on_solver_fail | bool | False | uses next $x$ of stored old solution if the NLP is not feasible |
| cheating_on_inactive_joint | float | 0.5 | inactive joint is set to be capable to exert a torque of 0.5 Nm as friction compensation |
| mpc_cycle_dt | float | 0.01 | frequency of the mpc |
| pd_tracking | bool | False | use PID Controller |
| outer_cycle_dt | float | 0.001 | timestep of the integrated PID controller |
| pd_KP | float | None | Gain for position error for the PID Controller |
| pd_KD | float | None | Gain for integrated error for the PID Controller |
| pd_KI | float | None | N_horizon |

