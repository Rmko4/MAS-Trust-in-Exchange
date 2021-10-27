""" This file contains the definition of the Network class, which holds all information
    regarding the agents and their location.
"""
from typing import TYPE_CHECKING

from trust.agent import GossipAgent

if TYPE_CHECKING:
    from trust.agent import BaseAgent
    from trust.model import PDTModel


class Network:
    """ Defines the Network.
    """

    def __init__(self, model: 'PDTModel', num_neighbourhoods) -> None:
        """ Initializes the Network. Takes the model and the amount of neighbourhoods as parameters.
            The market, as well as the list of neighbourhoods, is initialized as empty sets
            representing that no agents are currently present in the market and neighbourhoods
            respectively.
        """
        self.model = model
        self.num_neighbourhoods = num_neighbourhoods
        self.market = set()
        self.neighbourhoods = [set() for _ in range(self.num_neighbourhoods)]

    def add_agent_to_neighbourhood(self, agent: 'BaseAgent', neighbourhood: int):
        """ Removes an agent (specified in the passed agent parameter) from its current
            neighbourhood and adds it to the neighbouhood specified in the passed neighbourhood
            parameter.
        """
        old_nbh = agent.neighbourhood
        self.neighbourhoods[old_nbh].discard(agent)

        self.neighbourhoods[neighbourhood].add(agent)
        agent.neighbourhood = neighbourhood

    def add_agent_to_market(self, agent: 'BaseAgent') -> None:
        """ Adds the agent passed in the parameters to the global market. Please note that the
            agent remains in the same neighbourhood.
        """
        self.market.add(agent)

    def remove_agent_from_market(self, agent: 'BaseAgent') -> None:
        """ Removes the agent passed in the parameters to the global market. Please note that the
            agent remains in the same neighbourhood.
        """
        self.market.discard(agent)

    def pair_and_play(self) -> None:
        """ Calls the method to play the prisoners dilemma for all neigbhourhoods and the global
            market. Note that an agent can only play in either their neighbourhood, or on the global
            market and not both.
        """
        for nbh in self.neighbourhoods:
            agents = [a for a in nbh if a not in self.market]
            self.play_PDT(agents)

        self.play_PDT(self.market)

    def play_PDT(self, agentSet: 'set[BaseAgent]') -> None:
        """ Randomly pairs all agents in the given agentset and lets them play the prisoners'
            dilemma.

            Once the agents have been matched with an agent, both agents decide whether to play the
            game. After that, both agents decide

            TODO
        """
        agent_list = list(agentSet)
        self.model.random.shuffle(agent_list)

        for i in range(int(len(agent_list)/2)):
            agent_a = agent_list[2*i]
            agent_b = agent_list[2*i + 1]

            agent_a.decide_cooperation()
            agent_b.decide_cooperation()

            agent_a.decide_play(agent_b)
            agent_b.decide_play(agent_a)

            if agent_a.play and agent_b.play:
                # Both agents trust so PD is played
                opportunity_cost = self.model.get_opportunity_cost(
                    len(agent_list))
                a_payoff = self.model.get_pdt_payoff(
                    (agent_a.pdtchoice, agent_b.pdtchoice), opportunity_cost)
                b_payoff = self.model.get_pdt_payoff(
                    (agent_b.pdtchoice, agent_a.pdtchoice), opportunity_cost)
                agent_a.receive_payoff(a_payoff)
                agent_b.receive_payoff(b_payoff)
            else:
                # One agent does not trust so PD is not played and both agents receive exit payoff
                agent_a.receive_payoff(self.model.exit_payoff)
                agent_b.receive_payoff(self.model.exit_payoff)

    def get_role_model(self, neighbourhood: int) -> 'BaseAgent':
        """ Returns most successfull agent (considering the cumulative payoff) in the
            neighbourhood as passed in the parameters.
        """
        def cumulative_payoff(agent: 'BaseAgent') -> int:
            return agent.cumulative_payoff
        return max(self.neighbourhoods[neighbourhood], key=cumulative_payoff)
