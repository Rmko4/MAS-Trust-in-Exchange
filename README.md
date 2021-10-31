# Trust in exchange

This repository contains the sources for running a multi-agent system that models trust in market exchanges.

## Running the model
The model can be run with:  
`python run.py [-h] [-a {MSAgent,WHAgent,RLAgent,GossipAgent,RLGossipAgent}] [-m {[0.0,1.0]}] [-N {[0,10000]}] [-n {[0,10000]}] [-l {[0.0,1.0]}] [-sl {[0.0,1.0]}] [-r {True,False}] [-ms {[0,10000]}] [-t1 {[0,1000000]}] [-t2 {[1,1000000]}] [--save-filename SAVE_FILENAME]`

  * `-h`, `--help` - Show the help message and exit
  * `-a`, `--agent-class` - {_MSAgent_, _WHAgent_, _RLAgent_, _GossipAgent_, _RLGossipAgent_} - Which type of agent to use.
  * `-m`, `--mobility-rate` - [0.0,1.0] - The probability of an agent moving to a new neighbourhood.
  * `-N`, `--number-of-agents` - [0,10000] - The total number of agents in the model.
  * `-n`, `--neighbourhood-size` - [0,10000] - The initial number of agents in each neighbourhood.
  * `-l`, `--learning-rate` - [0.0,1.0] - (_RLAgent_, _RLGossipAgent_ only) The discount factor with which probabilities updated.
  * `-sl`, `--social-learning-rate` - [0.0,1.0] - (_RLAgent_, _RLGossipAgent_ only) The probability of copying a propensity.
  * `-r`, `--relative-reward` - {_True_, _False_} - (_RLAgent_, _RLGossipAgent_ only) Whether to normalize rewards to a mean of zero.
  * `-ms`, `--memory-size` - [0,10000] - (_GossipAgent_, _RLGossipAgent_ only) The number of memories an agent can store.
  * `-t1`, `--T_onset` - [0,1000000] - The number of time steps to run before recording data.
  * `-t2`, `--T_record` - [1,1000000] - The number of time steps to run for recording the data.
  * `--save-filename` - _SAVE_FILENAME_ - Saves to /m\__SAVE-FILENAME_ and /a\__SAVE-FILENAME_

## Repository contents description
* The starting point for running the code is the file [`run.py`](run.py). [`runMultipleExperiments.py`](runMultipleExperiments.py) contains the code for running several experiments.  
* The model implementation can be found in the [_trust_](trust) folder. The [_utils_](utils) folder contains some utilties for use by the model and running scripts.  
* The data from running the model with [`run.py`](run.py) will be stored in the [_data_](data) folder.  
* Scripts for plotting and some resulting plots is to be found in the [_plotting_](plotting) folder.
