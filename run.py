import pandas as pd
from trust.model import PDTModel

DATA_PATH = 'data/'

N = 1000
n = 10
mob_rate = 0.3


Market_Size = []
Trust_in_Strangers = []
Signal_Reading = []
Trust_Rate = []
Cooperating_Agents = []


for k in range(10):



    print("Number of agents: " + str(N))
    print("Neighbourhood size: " + str(n))
    print("Mobility rate: " + str(mob_rate))
    model = PDTModel(N=N, neighbourhood_size=n, mobility_rate=mob_rate)
    # print([a.neighbourhood for a in model.schedule.agents])
    model.run_model(100, 1000)
    df = model.datacollector.get_model_vars_dataframe()

    Market_Size.append(df["Market_Size"].mean())
    Trust_in_Strangers.append(df["Trust_in_Strangers"].mean())
    Signal_Reading.append(df["Signal_Reading"].mean())
    Trust_Rate.append(df["Trust_Rate"].mean())
    Cooperating_Agents.append(df["Cooperating_Agents"].mean())



    print(df.describe())
    #df.to_csv(DATA_PATH + 'data.csv')
    n += 10


print(Market_Size)
print(Trust_in_Strangers)
print(Signal_Reading)
print(Trust_Rate)
print(Cooperating_Agents)
