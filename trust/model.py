""" This file contains the PDTModel and all its associated funtionality.
"""

from typing import Union

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

import trust.agent as agent_module
from trust.activation import TwoStepActivation
from trust.agent import *
from trust.choice import PDTChoice
from trust.network import Network


class PDTModel(Model):
    """ Defines the PDTModel. The associated payoffs are given as attributes of the model.
    """
    _PDT_PAYOFF = {(PDTChoice.DEFECT, PDTChoice.COOPERATE): 1,  # Without opportunity cost
                   (PDTChoice.COOPERATE, PDTChoice.COOPERATE): 0.7,
                   (PDTChoice.DEFECT, PDTChoice.DEFECT): -0.2,
                   (PDTChoice.COOPERATE, PDTChoice.DEFECT): -0.5,
                   }
    _EXIT_PAYOFF = -0.2

    def get_opportunity_cost(self, neighbourhood_size: int) -> float:
        """ Calculates and returns the opportunity costs (1 - (n-1)/(N-1)).
        """
        return 1 - (neighbourhood_size - 1) / (self.num_agents - 1)

    def get_pdt_payoff(self, choices: 'tuple[int, int]', opportunity_cost: int) -> float:
        """ Looks up the payoff based on the choices made by both agents.
            Returns the payoffs after the opportunity costs, if needed, have been subtracted.
        """
        payoff = PDTModel._PDT_PAYOFF[choices]
        if choices[1] == PDTChoice.COOPERATE:
            payoff -= 0.5 * opportunity_cost
        return payoff

    @property
    def exit_payoff(self):
        """ Returns the exit payoff of the model.
        """
        return PDTModel._EXIT_PAYOFF

    # Can pass any class of agent that implements BaseAgent to AgentClass. Passing a str of
    # the class also suffices: e.g. 'RLAgent'. kwargs are keyword arguments that are passed
    # on to the __init__ of RLAgent. Check implementation for available args.
    def __init__(self, AgentClass: Union[str, type] = MSAgent, number_of_agents: int = 1000,
                 neighbourhood_size: int = 50, mobility_rate: float = 0.2, **kwargs) -> None:
        """ Initializes the model. Can take parameters defining the agent type (default MSAgent),
            population size N (default 1000), neighbourhood size (default 50) and mobility rate
            (default 0.2). kwargs are keyword arguments that are passed on to the __init__ of
            RLAgent.

            The number of neighbourhoods n is calculated, after which a network with n
            neighbourhoods is created. Additionaly, a scheduler (as defined in activation.py)
            is created. All agents are distributed amongst the neighbourhoods and added to
            the scheduler. Besides this, a datacollector is set up and initialized.
        """

        self.num_agents = number_of_agents
        self.num_neighbourhoods = int(self.num_agents / neighbourhood_size)

        self.network = Network(self, self.num_neighbourhoods)
        self.schedule = TwoStepActivation(self)

        self.mobility_rate = mobility_rate

        if isinstance(AgentClass, str):
            AgentClass = getattr(agent_module, AgentClass)

        if AgentClass == GossipAgent:
            kwargs["num_agents"] = self.num_agents

        for i in range(self.num_agents):
            neighbourhood = int(i % self.num_neighbourhoods)
            # TODO: Cluster agent location into neighborhoods of randomly varying size.
            agent = AgentClass(i, self, neighbourhood, **kwargs)
            self.schedule.add(agent)
            self.network.add_agent_to_neighbourhood(agent, neighbourhood)

        self.datacollector = DataCollector(
            {
                "Market_Size": lambda m: m.market_size(),
                "Trust_in_Strangers": lambda m: m.trust_in_strangers(),
                "Signal_Reading": lambda m: m.signal_reading(),
                "Trust_Rate": lambda m: m.trust_rate(),
                "Cooperating_Agents": lambda m: m.cooperating_agents(),
                "Trust_in_Neighbors": lambda m: m.trust_in_neighbors(),
                "Trust_in_Newcomers": lambda m: m.trust_in_newcomers()
            }
        )

    def step(self):
        """ Lets the scheduler execute a step for all agents. Afterward, the nework pairs
            all agents to another agent (with the exception of odd amount of agents in a
            neighbourhood or globally) and lets them play the prisoners' game. If needed,
            the data are stored in the datacollector. Finally, it lets the scheduler execute
            the finalize method for all agents (and move to the next step).
        """
        self.schedule.step()
        self.network.pair_and_play()

        if self.record:
            self.datacollector.collect(self)
        self.schedule.finalize()

    def run_model(self, T_onset=1000, T_record=1000) -> None:
        """ Runs the model, given the parameters T_onset and T_record which represent the amount of
            'startup' steps (to get rid of startup anomalies) and recorded steps respectively.
        """
        self.running = True
        self.record = False
        for _ in range(T_onset):
            self.step()
        self.record = True
        for _ in range(T_record):
            self.step()
        self.running = False

    def market_size(self) -> float:
        """ Returns the percentage out of all agents which currently is in the global market.
        """
        return len([a for a in self.schedule.agents if a.in_market]) / self.num_agents

    def trust_rate(self) -> float:
        """ Returns the percentage out of all agents which decided to play
            (so, trust) the prisoners' dilemma with the agent they have been matched with.
        """
        return len([a for a in self.schedule.agents if a.play]) / self.num_agents

    def cooperating_agents(self) -> float:
        """ Returns the percentage out of all agents that played the prisoners' dilemma with the
            agent they have been matched with and decided to cooperate.
        """
        return len([a for a in self.schedule.agents if a.pdtchoice == PDTChoice.COOPERATE]) \
            / self.num_agents

    def trust_in_strangers(self) -> float:
        """ Returns the percentage out of all agents matched with a stranger that have decided
            to play (so, trust) the prisoners' dilemma with the agent they have been matched with.
        """
        a_with_stranger_partners = [
            a for a in self.schedule.agents if a.stranger_partner]
        return len([a for a in a_with_stranger_partners if a.play]) / len(a_with_stranger_partners)

    def trust_in_neighbors(self) -> float:
        """ Returns the percentage out of all agents matched with a neighbour that have decided
            to play (so, trust) the prisoners' dilemma with the agent they have been matched with.
        """
        a_with_neighbor_partners = [
            a for a in self.schedule.agents if not a.in_market]
        return len([a for a in a_with_neighbor_partners if a.play]) / len(a_with_neighbor_partners)

    def trust_in_newcomers(self) -> float:
        """ Returns the percentage out of all agents matched with a newcommer of the neighbourhood
            that have decided  to play (so, trust) the prisoners' dilemma with the agent they
            have been matched with.
        """
        a_with_newcommers = [
            a for a in self.schedule.agents if a.partern_is_newcommer]
        if len(a_with_newcommers) == 0:
            return 0
        return len([a for a in a_with_newcommers if a.play]) / len(a_with_newcommers)

    def signal_reading(self) -> float:
        """ Returns the mean value of the probability to trust another agent amongst all agents.
        """
        return np.mean([a.trust_prob for a in self.schedule.agents])
