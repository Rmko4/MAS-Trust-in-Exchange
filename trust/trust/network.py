from trust.model import PDTModel
from trust.agent import PDTAgent


class Network:
    def __init__(self, model: PDTModel, num_neighbourhoods) -> None:
        self.model = model
        self.num_neighbourhoods = num_neighbourhoods
        self.market = []
        self.neighbourhoods = [[] for _ in range(self.num_neighbourhoods)]

    def add_agent(self, agent: PDTAgent, neighbourhood: int):
        old_nbh = agent.neighbourhood
        self.neighbourhoods[old_nbh].remove(agent)

        self.neighbourhoods[neighbourhood].append(agent)
        agent.neighbourhood = neighbourhood

    def add_agent_to_market(self, agent: PDTAgent) -> None:
        self.market.append(agent)

    def pair(self) -> None:
        for nbh in self.neighbourhoods:
            agents = [a for a in nbh if a not in self.market]
            # Pairs


    def clean_up(self) -> None:
        self.market.clear()
