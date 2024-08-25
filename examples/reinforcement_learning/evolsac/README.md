# EvolSAC training
To train the Evolutionary SAC Agent for both the pendubot and the acrobot, first ensure that the variable robot is consistently set to either acrobot or pendubot in all 3 `main.py` files contained inside the folders `SAC_main_training` and `SNES_finetuning`. 

The scripts below must be ran directly from the folders that contain them to ensure path integrity.

1. Run `python main.py 3.0 0 0 0` from `SAC_main_training`, which trains the agent according to the surrogate reward function defined in the same file
2. Run `python main.py 3.0 0 0 0 [acrobot/pendubot]` from `SNES_finetuning`, which loads the agent found in step 3, and further trains it based on the performance score defined by the competition's organizers