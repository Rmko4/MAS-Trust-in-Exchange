from typing import Union
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from trust.activation import TwoStepActivation
from trust.agent import MSAgent, WHAgent
import trust.agent as agent
from trust.choice import PDTChoice
from trust.network import Network


class PDTModel(Model):
    _PDT_PAYOFF = {(PDTChoice.DEFECT, PDTChoice.COOPERATE): 1,  # Without opportunity cost
                   (PDTChoice.COOPERATE, PDTChoice.COOPERATE): 0.7,
                   (PDTChoice.DEFECT, PDTChoice.DEFECT): -0.2,
                   (PDTChoice.COOPERATE, PDTChoice.DEFECT): -0.5,
                   }
    _EXIT_PAYOFF = -0.2

    def get_opportunity_cost(self, neighbourhood_size: int) -> float:
        return 1 - (neighbourhood_size - 1) / (self.num_agents - 1)

    def get_pdt_payoff(self, choices: 'tuple[int, int]', opportunity_cost: int) -> float:
        payoff = PDTModel._PDT_PAYOFF[choices]
        if choices[1] == PDTChoice.COOPERATE:
            payoff -= 0.5 * opportunity_cost
        return payoff

    @property
    def exit_payoff(self):
        return PDTModel._EXIT_PAYOFF

    # Can pass any class of agent that implements BaseAgent to AgentClass. Passing a str of the class also suffices: e.g. 'RLAgent'.
    # kwargs are keyword arguments that are passed on to the __init__ of RLAgent. Check implementation for available args.
    def __init__(self, AgentClass: Union[str, type] = MSAgent, N=1000, neighbourhood_size=50, mobility_rate=0.2, **kwargs) -> None:
        self.num_agents = N
        self.num_neighbourhoods = int(self.num_agents / neighbourhood_size)

        self.network = Network(self, self.num_neighbourhoods)
        self.schedule = TwoStepActivation(self)

        self.mobility_rate = mobility_rate

        if type(AgentClass) == str:
            AgentClass = getattr(agent, AgentClass)

        for i in range(self.num_agents):
            neighbourhood = int(i % self.num_neighbourhoods)
            # TODO: Cluster agent location into neighborhoods of randomly varying size.
            a = AgentClass(i, self, neighbourhood, **kwargs)
            self.schedule.add(a)
            self.network.add_agent_to_neighbourhood(a, neighbourhood)

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
        self.schedule.step()
        self.network.pair_and_play()

        if self.record == True:
            self.datacollector.collect(self)
        self.schedule.finalize()

    def run_model(self, T_onset=1000, T_record=1000) -> None:
        self.running = True
        self.record = False
        for _ in range(T_onset):
            self.step()
        self.record = True
        for _ in range(T_record):
            self.step()
        self.running = False

    def market_size(self) -> float:
        return len([a for a in self.schedule.agents if a.in_market]) / self.num_agents

    def trust_rate(self) -> float:
        return len([a for a in self.schedule.agents if a.play]) / self.num_agents

    def cooperating_agents(self) -> float:
        return len([a for a in self.schedule.agents if a.pdtchoice == PDTChoice.COOPERATE]) / self.num_agents

    def trust_in_strangers(self) -> float:
        a_with_stranger_partners = [
            a for a in self.schedule.agents if a.stranger_partner]
        return len([a for a in a_with_stranger_partners if a.play]) / len(a_with_stranger_partners)

    def trust_in_strangers(self) -> float:
        a_with_stranger_partners = [
            a for a in self.schedule.agents if a.stranger_partner]
        return len([a for a in a_with_stranger_partners if a.play]) / len(a_with_stranger_partners)

    def trust_in_neighbors(self) -> float:
        a_with_neighbor_partners = [
            a for a in self.schedule.agents if not a.in_market]
        return len([a for a in a_with_neighbor_partners if a.play]) / len(a_with_neighbor_partners)

    def trust_in_newcomers(self) -> float:
        a_with_newcommers = [
            a for a in self.schedule.agents if a.partern_Is_Newcommer]
        if len(a_with_newcommers) == 0:
            return 0
        return len([a for a in a_with_newcommers if a.play]) / len(a_with_newcommers)

    def signal_reading(self) -> float:
        return np.mean([a.signal_reading_prob for a in self.schedule.agents])
