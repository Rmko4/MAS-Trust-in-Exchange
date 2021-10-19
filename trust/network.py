from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trust.agent import BaseAgent
    from trust.model import PDTModel


class Network:
    def __init__(self, model: 'PDTModel', num_neighbourhoods) -> None:
        self.model = model
        self.num_neighbourhoods = num_neighbourhoods
        self.market = set()
        self.neighbourhoods = [set() for _ in range(self.num_neighbourhoods)]

    def add_agent_to_neighbourhood(self, agent: 'BaseAgent', neighbourhood: int):
        old_nbh = agent.neighbourhood
        self.neighbourhoods[old_nbh].discard(agent)

        self.neighbourhoods[neighbourhood].add(agent)
        agent.neighbourhood = neighbourhood

    def add_agent_to_market(self, agent: 'BaseAgent') -> None:
        self.market.add(agent)

    def remove_agent_from_market(self, agent: 'BaseAgent') -> None:
        self.market.discard(agent)

    def pair_and_play(self) -> None:
        for nbh in self.neighbourhoods:
            agents = [a for a in nbh if a not in self.market]
            self.play_PDT(agents)

        self.play_PDT(self.market)

    def play_PDT(self, agentSet: 'set[BaseAgent]') -> None:
        agentList = list(agentSet)
        self.model.random.shuffle(agentList)

        for i in range(int(len(agentList)/2)):
            a = agentList[2*i]
            b = agentList[2*i + 1]

            a.decide_cooperation()
            b.decide_cooperation()

            a.decide_play(b)
            b.decide_play(a)

            if a.play and b.play:
                # Both agents trust so PD is played
                opportunity_cost = self.model.get_opportunity_cost(
                    len(agentList))
                a_payoff = self.model.get_pdt_payoff(
                    (a.pdtchoice, b.pdtchoice), opportunity_cost)
                b_payoff = self.model.get_pdt_payoff(
                    (b.pdtchoice, a.pdtchoice), opportunity_cost)
                a.receive_payoff(a_payoff)
                b.receive_payoff(b_payoff)
            else:
                # One agent does not trust so PD is not played and both agents receive exit payoff
                a.receive_payoff(self.model.exit_payoff)
                b.receive_payoff(self.model.exit_payoff)

    def get_role_model(self, neighbourhood: int) -> 'BaseAgent':
        return max(self.neighbourhoods[neighbourhood], key=lambda a: a.cumulative_payoff)
        # potential_models = [a for a in self.neighbourhoods[neighbourhood] if not a.newcomer]
        # if len(potential_models) == 0:
        #     return None
        # else:
        #     return max(potential_models, key=lambda a: a.cumulative_payoff)
