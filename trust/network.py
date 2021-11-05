""" This file contains the definition of the Neighbourhood and Network class, which holds
    information regarding the neighbourhoods and agents and their location respectively.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trust.agent import BaseAgent
    from trust.model import PDTModel


class Neighbourhood(set):
    """ Defines the neighbourhood. 
    """
    def __init__(self) -> None:
        """ Initializes the neighbourhood as a set. Sets the role model to none. 
        """
        super().__init__()
        self.role_model = None

    def set_role_model(self):
        """ Updates the neighbourhoods role model. This role model is chosen
            to be the neighbourhoods most successfull agent considering their
            cumulative payoff.
        """
        def _cumulative_payoff(agent: 'BaseAgent') -> int:
            return agent.cumulative_payoff
        if len(self) > 0:
            self.role_model = max(self, key=_cumulative_payoff)

    def get_role_model(self):
        """ Returns the current role model of the neighbourhood, updates this
            if there is no role model yet.
        """
        if self.role_model is None:
            self.set_role_model()
        return self.role_model


class Network:
    """ Defines the Network, containing the model, global market and neighbourhoods.
    """
    def __init__(self, model: 'PDTModel', num_neighbourhoods) -> None:
        """ Initializes the Network. Takes the model and the amount of neighbourhoods as parameters.
            The market, is initialized as an empty neighbourhood. The networks neighbourhoods parameter,
            is initialized as a list of empty neighbourhoods.
        """
        self.model = model
        self.num_neighbourhoods = num_neighbourhoods
        self.market = Neighbourhood()
        self.neighbourhoods = [Neighbourhood()
                               for _ in range(self.num_neighbourhoods)]

    def add_agent_to_neighbourhood(self, agent: 'BaseAgent', neighbourhood: int):
        """ Removes an agent (as specified in the passed agent parameter) from its current
            neighbourhood and adds it to the new neighbouhood as specified in the passed
            neighbourhood parameter.
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
        """ For each neighbourhood, the role model is updated after which the prisoners'
            dilemma is played for all agents in that neighbourhood that have not decided
            to enter to global market. After this, the prisoners' dilemma is played for
            all agents that have decided to enter the global market. Note that an agent
            can only play in either their neighbourhood, or on the global market and not both.
        """
        for nbh in self.neighbourhoods:
            nbh.set_role_model()
            agents = [a for a in nbh if a not in self.market]
            self.play_PDT(agents)
        self.play_PDT(self.market)

    def play_PDT(self, agentSet: 'set[BaseAgent]') -> None:
        """ Randomly pairs all agents in the given agentset. Once the agents have been
            matched with an agent, both agents decide whether to cooporate or defect.
            After that, both agents decide whether to play the game, or exit. If both
            agents decide to play the game, the payoffs for both agents are calculated
            and given to both agents.
            
            If at least one of the agents decide to exit, both agents receive the exit
            payoff.
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
                opportunity_cost = self.model.get_opportunity_cost(
                    len(agent_list))
                a_payoff = self.model.get_pdt_payoff(
                    (agent_a.pdtchoice, agent_b.pdtchoice), opportunity_cost)
                b_payoff = self.model.get_pdt_payoff(
                    (agent_b.pdtchoice, agent_a.pdtchoice), opportunity_cost)
                agent_a.receive_payoff(a_payoff)
                agent_b.receive_payoff(b_payoff)
            else:
                agent_a.receive_payoff(self.model.exit_payoff)
                agent_b.receive_payoff(self.model.exit_payoff)

    def get_role_model(self, neighbourhood: int) -> 'BaseAgent':
        """ Returns role model of the neighbourhood as passed in the parameters.
        """
        return self.neighbourhoods[neighbourhood].get_role_model()
