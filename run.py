import pandas as pd
from trust.model import PDTModel

DATA_PATH = 'data/'

model = PDTModel(N=1000, neighbourhood_size=50, mobility_rate=0.4)
# print([a.neighbourhood for a in model.schedule.agents])
model.run_model(100, 1000)
df = model.datacollector.get_model_vars_dataframe()

print(df.describe())
df.to_csv(DATA_PATH + 'data.csv')