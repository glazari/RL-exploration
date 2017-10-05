# RL-exploration

This is the code for a few of the experiments I am doing for my undergraduate termination project.
So far it is not organized in any particular way. But hopfuly I will get around to organize it once
the project gets a little more momentum.

## Running the code

Running the code is as simple as running the "Run_Experiment.py" script.

    python3 Run_Experiment.py
    
It will ask you to name the experiment and show you the current parameters being used. To complete training the code takes around 6h to 12h depending on how many timesteps you set it up to run for.


## Install dependencies

This code depends on *gym* for the atari emulator, *tensorflow* for the neural networks and open-cv to save some videos. To install the dependencies simply copy the lines below.

    pip3 install gym[atari] dill opencv-python
    
    pip3 install tensorflow #Install thee gpu version for faster running code


## Disclaimer

This code is based largely on openai's baselines code for deepq. Credits for the original implementation goes to them. You can check it out at:             
                                        
https://github.com/openai/baselines 
