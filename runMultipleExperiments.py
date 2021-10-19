import pandas as pd
import copy
from trust.model import PDTModel

import numpy as np
import sys

N = 1000
n_min = 10
n_max = 100
n_stepsize = 10
mob_rate_min = 0
mob_rate_max = 0.3
mob_rate_stepsize = 0.03

if (len(sys.argv) == 1):
    print("Please specify output file")
    sys.exit()

with open(str(sys.argv[1]), 'w') as f:
    f.write(str(n_min) +" " + str(n_max) +" " + str(n_stepsize) +" " + str(mob_rate_min) +" " + str(mob_rate_max) + " " +str(mob_rate_stepsize) + "\n")

    print("Number of agents: " + str(N))
    for n in np.arange(n_min,n_max + 0.001,n_stepsize):
        for mob_rate in np.arange(mob_rate_min,mob_rate_max + 0.0001,mob_rate_stepsize):
            print("Neighborhood size: " + str(n) + ", Mobility rate: " + str(mob_rate))

            model = PDTModel(N=N, neighbourhood_size=n, mobility_rate=mob_rate)
            model.run_model(100, 1000)
            df = model.datacollector.get_model_vars_dataframe()

            f.write(str(df["Market_Size"].mean()) + " ")
            f.write(str(df["Trust_in_Strangers"].mean()) + " ")
            f.write(str(df["Signal_Reading"].mean()) + " ")
            f.write(str(df["Trust_Rate"].mean()) + " ")
            f.write(str(df["Cooperating_Agents"].mean()) + " ")
            f.write(str(df["Trust_in_Neighbors"].mean()) + " ")
            f.write(str(df["Trust_in_Newcomers"].mean()) + "\n")






