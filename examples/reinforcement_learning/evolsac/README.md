# EvoSAC training
To train the Evolutionary SAC Agent for both the pendubot and the acrobot
1. Ensure that the variable robot is consistently set to either acrobot or pendubot in all 3 `main.py` files contained inside the folders `SAC_main_training`, `SAC_finetuning` and `SNES_finetuning`
2. Run python `SAC_main_training/main.py`, which trains the agent according to the surrogate reward function defined in the same file
3. Run python `SAC_finetuning/main.py`, which loads the agent found in step 2, and further trains it according to a refined surrogate reward function defined in the same file
4. Run python `SNES_finetuning/main.py`, which loads the agent found in step 3, and further trains it based on the performance score defined by the competition's organizers