# Time-varying LQR (TVLQR) with iLQR trajectory

This controller is implemented using the acados framework https://docs.acados.org/
for acados problem formulation see https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

## simple setup
1.initialize controller with model parameters AcadosMpcController(
    model_pars=mpar
)

2. set controller parameter with controller.

3. optionally.set cosntraints on velocity with controller.set_velocity_constraints(v_max=vmax, v_final=vf)

4. set cost matrices on x, f_final and u error with controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat

5. controller.init()

## controller options

| Command | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |


| Attribute | Type | Default | Description |
| -------- | ------- |
| N_horizon | int | 20 | number of shooting nodes |
| prediction_horizon | float | 0.5 | prediction horizon |


