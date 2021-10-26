''' This runfile starts the multi-agent system model.
    Also, it prints the values for the mobility rate, neighbourhood size an total number of agents.
    After the model has run, the results will be printed.
'''
from trust.model import PDTModel

DATA_PATH = 'data/'

N = 1000
n = 10
mob_rate = 0.1

print("Number of agents: " + str(N))
print("Neighbourhood size: " + str(n))
print("Mobility rate: " + str(mob_rate))

agent_class = input("\nChoose agentclass (Default is RLAgent):\n 1: MSAgent, 2: WHAgent, 3: GossipAgent, 4: RLAgent.\n\n")
if agent_class == "1":
    agent_class = "MSAgent"
    model = PDTModel(AgentClass="MSAgent", N=N, neighbourhood_size=n, mobility_rate=mob_rate)
elif agent_class == "2":
    agent_class = "WHAgent"
    model = PDTModel(AgentClass="WHAgent", N=N, neighbourhood_size=n, mobility_rate=mob_rate)
elif agent_class == "3":
    agent_class = "GossipAgent"
    model = PDTModel(AgentClass="GossipAgent", N=N, neighbourhood_size=n, mobility_rate=mob_rate)
else:
    agent_class = "RLAgent"
    model = PDTModel(AgentClass="RLAgent", N=N, neighbourhood_size=n, mobility_rate=mob_rate, learning_rate=0.02, relative_reward=False)

print("Chosen agent type: " + agent_class)
# print([a.neighbourhood for a in model.schedule.agents])
model.run_model(100, 1000)
df = model.datacollector.get_model_vars_dataframe()

print(df.describe())
df.to_csv(DATA_PATH + 'data.csv')
