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

    def pair_and_play(self) -> None:
        for nbh in self.neighbourhoods:
            agents = [a for a in nbh if a not in self.market]
            self.play_PDT(agents)

        self.play_PDT(self.market)

    def play_PDT(self, agentList: list[PDTAgent]) -> None:
        self.model.random.shuffle(agentList)
        for i in range(int(len(agentList)/2)):
            a = agentList[i]
            b = agentList[i]
            
            a.decide_exchange()
            b.decide_exchange()

            if a.play and b.play:
                opportunity_cost = self.model.opportunity_cost(len(agentList))
                a_payoff = self.model.pdt_payoff((a.pdtchoice, b.pdtchoice), opportunity_cost)
                b_payoff = self.model.pdt_payoff((b.pdtchoice, a.pdtchoice), opportunity_cost)
                a.update_payoff(a_payoff)
                b.update_payoff(b_payoff)
            else:
                a.update_payoff(self.model.EXIT_PAYOFF)
                b.update_payoff(self.model.EXIT_PAYOFF)

    def get_role_model(self, neighbourhood: int) -> PDTAgent:
        return max(self.neighbourhoods[neighbourhood], key=lambda a: a.cumulative_payoff)
    
    def clean_up(self) -> None:
        self.market.clear()
