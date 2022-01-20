# caprr-release-version

combinational active passive RR (CAPRR)
A repository for double pendulum, pendubot, and acrobot.

Suggestion for paper title: A(n Open Source) Dual Purpose Acrobot & Pendubot Platform (based on Quasi-Direct Drives) For Studying Learning and Control Algorithms for Underactuated Robotics


### Repository Overview

| Controller/Method     | Simulation Success (acrobot)   | Simulation Success (pendubot)   | Tested on real acrobot |   Migrated    |
|:----------------------|:------------------------------:|:-------------------------------:|:----------------------:|:-------------:|
|LQR (dependency free)  |yes                             |yes                              |                        |yes            |
|LQR (drake)            |yes                             |yes                              |                        |no             |
|Partial feedback lin   |yes                             |no                               |no                      |yes            |
|Direct Col. (drake)    |yes                             |yes                              |yes (failed with tvlqr) |no             |
|TVLQR (drake)          |yes                             |yes                              |yes (failed with dc traj)|no            |
|iLQR (drake )          |no                              |no                               |no                      |no             |
|iLQR (c++)             |no                              |no                               |no                      |no             |
|DDP (crocoddyl)        |yes                             |yes                              |no                      |no             |


|Other                                  | Implemented   | Migrated  |
|:--------------------------------------|:-------------:|:---------:|
|Plant + Simulator (python)             |yes            |yes        |
|Plant + Simulator (c++)                |yes            |no         |
|System Identification                  |yes            |no         |
|LQR Region of Attraction               |yes            |no         |
|Design Optimization                    |no             |no         |
|Parameter Optimization with CMA-ES     |yes            |yes        |
|Controller Benchmark Tool              |no             |no         |
|Hardware specification/documentation   |no             |no         |

# Todos
- Autodesk A360
    - Get a CAD from Heiner, get a link from the website and put it in repository
- Cleanup
    - An **Example folder** for test and simulation
    - **data folder** for the results and data of the system
- Software
    - Double pendulum 
        - Trajectory optimization
        - TVLQR
        - LQR
    - Design optimization
    - **unit test** folder

- Documentation
- Tests