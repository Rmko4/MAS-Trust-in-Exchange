import pandas as pd
import copy
from trust.model import PDTModel

DATA_PATH = 'data/'


N = 1000
n = 10
mob_rate = 0.1



print("Number of agents: " + str(N))
print("Neighbourhood size: " + str(n))
print("Mobility rate: " + str(mob_rate))
<<<<<<< HEAD
model = PDTModel(agentClass="WHAgent", N=N, neighbourhood_size=n, mobility_rate=mob_rate)
=======
model = PDTModel(N=N, neighbourhood_size=n, mobility_rate=mob_rate)


>>>>>>> 54bbddf09fd8d97fb76e767b5163fd296ed1d87b
# print([a.neighbourhood for a in model.schedule.agents])
model.run_model(100, 1000)
df = model.datacollector.get_model_vars_dataframe()

print(df.describe())
df.to_csv(DATA_PATH + 'data.csv')

