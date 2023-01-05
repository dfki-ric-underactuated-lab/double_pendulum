import pickle
import os
import multiprocessing
import time
import numpy as np

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface

robot = "acrobot"
filename = "heatmap_l1l2_"+robot+".pickle"

N_PROC = int(min(multiprocessing.cpu_count() - 1, 32))
evals = 100000
n = 32

torque_limit = 5.0
m1 = 0.6
m2 = 0.6
l1 = 0.3
l2 = 0.2
R_init = np.diag((1., 1.))
Q_init = np.diag((1., 1., 1., 1.))


save_dir = os.path.join("../../results", "design_optimization", robot, "lqr", "heatmaps")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, filename)

if robot == "pendubot":
    tau_max = [torque_limit, 0.0]
elif robot == "acrobot":
    tau_max =[0.0, torque_limit]
design_params = {
    "m": [m1, m2],
    "l": [l1, l2],
    "lc": [l1, l2],
    "b": [0.0, 0.0],
    "fc": [0.0, 0.0],
    "g": 9.81,
    "I": [m1*l1**2, m2*l2**2],
    "tau_max": tau_max,
}

backend = "sos_con" # or "najafi"
caprr_roaEst = caprr_coopt_interface(
    design_params, Q_init, R_init, backend=backend,
    najafi_evals=evals, estimate_clbk=None, robot=robot)


def roa(roa_interface, m2, l1, l2, idx1, idx2, array_size, array):
    vol = roa_interface.design_opt_obj([m2, l1, l2])
    array[idx1*array_size[1]+idx2] = vol


l1Vals = np.linspace(0.2, 0.4, n)
l2Vals = np.linspace(0.2, 0.4, n)

comp_list = []
for idx1, l1 in enumerate(l1Vals):
    for idx2, l2 in enumerate(l2Vals):
        comp_list.append([l1, l2, idx1, idx2])

comp_list2 = []
cc = []
for i, c in enumerate(comp_list):
    if i > 0 and i % N_PROC == 0:
        comp_list2.append(cc)
        cc = []
    cc.append(c)
if len(cc) > 0:
    comp_list2.append(cc)

i = 0
init = time.time()
vol_array = multiprocessing.Array('d', [0]*len(l1Vals)*len(l2Vals))
for c in comp_list2:
    manager = multiprocessing.Manager()
    jobs = []
    for cc in c:
        p = multiprocessing.Process(target=roa, args=(caprr_roaEst,
                                                      m2,
                                                      cc[0],
                                                      cc[1],
                                                      cc[2],
                                                      cc[3],
                                                      [len(l1Vals), len(l2Vals)],
                                                      vol_array))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
        i = i+1
        #print(f"{i} jobs done...")
end = time.time()

prob_vols = np.zeros((len(l1Vals), len(l2Vals)))
for idx1, q1 in enumerate(l1Vals):
    for idx2, q2 in enumerate(l2Vals):
        prob_vols[idx1][idx2] = vol_array[idx1*len(l2Vals)+idx2]

results = {"prob_vols": prob_vols,
           "yticks": l1Vals,
           "xticks": l2Vals,
           "backend": backend,
           "robot": robot,
           "execution_time": end-init}

print("execution_time: ",end-init)

outfile = open(save_file, 'wb')
pickle.dump(results, outfile)
outfile.close()
