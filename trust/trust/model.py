from trust.agent import PDTAgent
from trust.network import Network
from mesa import Model
from mesa.time import SimultaneousActivation


class PDTModel(Model):

    def __init__(self, N=1000, neighbourhood_size=50, mobility_rate=0.2) -> None:
        self.num_agents = N
        self.num_neighbourhoods = int(self.num_agents / neighbourhood_size)

        self.network = Network(self, self.num_neighbourhoods)
        self.schedule = SimultaneousActivation(self)

        self.mobility_rate = mobility_rate

        for i in range(self.num_agents):
            neighbourhood = int(i % self.num_neighbourhoods)
            # TODO: Cluster agent location into neighborhoods of randomly varying size.
            a = PDTAgent(i, self, neighbourhood)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
        self.network.pair()

        self.schedule.advance()
        self.network.clean_up()
