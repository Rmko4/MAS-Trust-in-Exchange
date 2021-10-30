# Trust in exchange

This repository contains the sources for running a multi-agent system that models trust in market exchanges.

The model can be run with:  
`python run.py [-h] [-a {MSAgent,WHAgent,RLAgent,GossipAgent}] [-m {[0.0,1.0]}] [-N {[0,10000]}] [-n {[0,10000]}]  
[-l {[0.0,1.0]}] [-r {True,False}] [-ms {[0,10000]}] [-t1 {[0,1000000]}] [-t2 {[0,1000000]}] [--save-filename SAVE_FILENAME]`

  * `-h`, `--help` - Show the help message and exit
  * `-a`, `--agent-class` - {_MSAgent_, _WHAgent_, _RLAgent_, _GossipAgent_} - Which type of agent to use.
  * `-m`, `--mobility-rate` - [0.0,1.0] - The probability of an agent moving to a new neighbourhood.
  * `-N`, `--number-of-agents` - [0,10000] - The total number of agents in the model.
  * `-n`, `--neighbourhood-size` - [0,10000] - The initial number of agents in each neighbourhood.
  * `-l`, `--learning-rate` - [0.0,1.0] - (_RLAgent_ only) The discount factor with which probabilities updated.
  * `-r`, `--relative-reward` - {_True_, _False_} - (_RLAgent_ only) Whether to normalize rewards to a mean of zero.
  * `-ms`, `--memory-size` - [0,10000] - (_GossipAgent_ only) The number of memories an agent can store.
  * `-t1`, `--T_onset` - [0,1000000] - The number of time steps to run before recording data.
  * `-t2`, `--T_record` - [1,1000000] - The number of time steps to run for recording the data.
  * `--save-filename` - _SAVE_FILENAME_ - Saves to /m\__SAVE-FILENAME_ and /a\__SAVE-FILENAME_
