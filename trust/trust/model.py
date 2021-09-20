from trust.agent import PDTAgent
from mesa import Model
from mesa.time import RandomActivation


class PDTModel(Model):

    def __init__(self, N=1000, neighbourhood_size=50, mobility_rate=0.2) -> None:
        self.num_agents = N
        self.neighbourhood_size = neighbourhood_size
        self.num_neighbourhoods = self.num_agents / self.neighbourhood_size
        self.mobility_rate = mobility_rate
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            neighbourhood = i%self.num_neighbourhoods # TODO: Cluster agent location into neighborhoods of randomly varying size.
            a = PDTAgent(i, self, neighbourhood)
            self.schedule.add(a)

    def step():
        # Change neighbourhood involuntary
        pass