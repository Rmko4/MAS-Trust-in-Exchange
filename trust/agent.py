""" This file contains the classes defining different types of agents.
"""
from typing import TYPE_CHECKING
from mesa import Agent

from trust.choice import PDTChoice
if TYPE_CHECKING:
    from trust.model import PDTModel


class BaseAgent(Agent):
    """ Defines a base agent, which is an implementation of an Agent as defined by the MESA module.
    """

    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int) -> None:
        """ Initializes the baseAgent. Saves the neighbourhood that the agent is in, and does not
            mark it as a newcomer. Initializs the propensity to read signals (antagonist
            parochialism), propensity to cooperate and propensity to enter the open market (over
            staying in the neighbourhood) to a random value between 0.0 and 1.0. It also sets
            the variable keeping track of the cumulative payoff to zero. This variable will be
            updated after every interaction with another agent, based on the outcome of the
            prisoners' dilemma (or the exit cost if the agents decides not to play).
        """
        super().__init__(unique_id, model)
        self.neighbourhood = neighbourhood
        self.newcomer = False

        # Equivalent to the propensity to read signals
        self.trust_prob = self.random.random()
        # Propensity to cooperate (over defect)
        self.trustworthiness_prob = self.random.random()
        # Propensity to enter the open market
        self.location_prob = self.random.random()

        self.play = True
        self.pdtchoice = PDTChoice.COOPERATE
        self.in_market = False
        self.read_signal = False
        self.paired = False

        self.payoff = 0
        self.cumulative_payoff = 0

    def step(self) -> None:
        """ Every step, the agent moves to a new neigbhourhood with a certain probability
            as defined by the mobility rate. Also, the agent chooses (based on the location
            probability) wether to stay in the neighbourhood or move to the global market
            for its next interaction.
        """
        # Change neighbourhood involuntary
        if self.random.random() < self.model.mobility_rate:
            self.move()
        else:
            self.stay()

        # Chose to exchange in neighbourhood or market
        if self.random.random() < self.location_prob:
            self.enter_market()

    def finalize(self) -> None:
        """ If the agent is paired with another agent, it will update its behaviour (i.e. its
            decision to play or exit and to cooperate or defect in the prisoners' dilema).
            Afterwards, the variable paired is reset to False and the agent will leave the
            global market (if it was there).
        """
        if self.paired:
            self.update_behaviour()
        self.paired = False
        self.leave_market()

    def decide_cooperation(self) -> None:
        """ Decides whether the agent will cooperate or defect in the current prisoners' dilemma.
            This is decided based on the agents' value for the probability of being trustworthy.

            As this function is equal for all types of agent, this remains uniform across the
            conditions of being in the neighbourhood versus the global market.
        """
        if self.random.random() < self.trustworthiness_prob:
            self.pdtchoice = PDTChoice.COOPERATE
        else:
            self.pdtchoice = PDTChoice.DEFECT

    def decide_play(self, exchange_partner: 'BaseAgent') -> None:
        """ The dicison whether or not to play the prisoners' dilemma is dependent on the
            type of agent and thus not implemented for the base agent.

            Returns a NotImplementedError in case the type of agent is not specified (this
            should not happen).
            TODO: update doc, as this should now be called from sub
        """
        self.paired = True

        # TODO: We never use the value of self.partner_is_newcomer right...?
        if not self.in_market and (exchange_partner.newcomer or self.newcomer):
            self.partern_is_newcommer = True
        else:
            self.partern_is_newcommer = False

        if exchange_partner.newcomer or self.newcomer:
            self.stranger_partner = True
        else:
            self.stranger_partner = False

    def move(self) -> None:
        """ Moves an agent to a different neighbourhood than it is in now, also marks
            the agent as a newcomer and resets its cumulative payoff.
        """
        new_nbh = self.random.randint(0, self.model.num_neighbourhoods - 1)
        # Ensure that it won't stay in the same neighbourhood
        if new_nbh >= self.neighbourhood:
            new_nbh = (new_nbh + 1) % (self.model.num_neighbourhoods - 1)
        self.model.network.add_agent_to_neighbourhood(self, new_nbh)

        self.newcomer = True
        self.cumulative_payoff = 0

    def stay(self) -> None:
        """ Removes the newcomer mark from an agent.
        """
        self.newcomer = False

    def enter_market(self) -> None:
        """ Adds the agent to the networks global market.
        """
        self.model.network.add_agent_to_market(self)
        self.in_market = True

    def leave_market(self) -> None:
        """ Removes the agent from the networks global market.
        """
        self.model.network.remove_agent_from_market(self)
        self.in_market = False

    def receive_payoff(self, payoff):
        """ Saves the payoff of the current step and adds it to the agent's cumulative payoff.
        """
        self.payoff = payoff
        self.cumulative_payoff += payoff

    def stochastic_learning(self, prob: float, payoff: float) -> float:
        """ Calculates and returns a new value according to the stochastic learning rate
            given the probability and payoff.
        """
        if payoff >= 0:
            return prob + (1 - prob) * payoff
        else:
            return prob + prob * payoff

    def update_behaviour(self):
        """ Updates the behaviour of the agent.

            Updates the role model of the agent, i.e. the most successfull agent of the
            neighbourhood. Also, the values for the propensity to enter the global market,
            propensity to read signals (I.e. base its decision to either cooperate or
            defect on reading signals instead of Parochialism) and propensity to either
            cooperate or defect are updated.
        """
        role_model = self.model.network.get_role_model(self.neighbourhood)
        social_learning = role_model is not None and role_model is not self

        if social_learning and self.random.random() > 0.5:
            self.location_prob = role_model.location_prob
        elif not self.in_market:
            self.location_prob = 1 - \
                self.stochastic_learning(1 - self.location_prob, self.payoff)
        else:
            self.location_prob = self.stochastic_learning(
                self.location_prob, self.payoff)

        if social_learning and self.random.random() > 0.5:
            self.trust_prob = role_model.trust_prob
        elif not self.read_signal:
            self.trust_prob = 1 - \
                self.stochastic_learning(1 - self.trust_prob, self.payoff)
        else:
            self.trust_prob = self.stochastic_learning(
                self.trust_prob, self.payoff)

        if social_learning and self.random.random() > 0.5:
            self.trustworthiness_prob = role_model.trustworthiness_prob
        elif self.pdtchoice == PDTChoice.COOPERATE:
            self.trustworthiness_prob = 1 - \
                self.stochastic_learning(
                    1 - self.trustworthiness_prob, self.payoff)
        else:
            self.trustworthiness_prob = self.stochastic_learning(
                self.trustworthiness_prob, self.payoff)


class MSAgent(BaseAgent):
    """ Implementation of the Macy and Sato agent, extend a BaseAgent.

        For this agent, the dicision to either play or walk away from an opportunity to play
        the prisoners' dilemma with the agent he is matched with is based on the propensity
        to read signals.
    """

    def decide_play(self, exchange_partner: 'MSAgent') -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma based
            on its propensity to trust another agent.
        """
        super().decide_play(exchange_partner)

        if self.random.random() < self.trust_prob:
            self.play = True
        else:
            self.play = False


class WHAgent(BaseAgent):
    """ Implementation of the Will and Hegselmann agent, extends a BaseAgent.

        For this agent, the dicision to either play or walk away from an opportunity to play
        the prisoners' dilemma with the agent he is matched with is different for known agents
        and stangers.
    """

    def decide_play(self, exchange_partner: 'WHAgent') -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma.

            First, the agent decides to either base its decision on signal reading
            or parochialism based on the agents' propensity to trust another agent.

            If the agent decides to read the signals of the other agent, the outcome of
            the signal reading determines the agents' choice to either cooperate or defect.

            If the agent decides to act parochial, it will only trust the other agent if
            it knows its opponent (so it is not on the global market, and the opponent
            is not a newcomer to the neighbourhood, and the agent itself is not a newcomer
            to the neighbourhood).
        """
        super().decide_play(exchange_partner)

        if self.random.random() < self.trust_prob:
            # Signal reading
            self.read_signal = True
            signal = exchange_partner.get_signal()
            if signal == PDTChoice.COOPERATE:
                self.play = True
            else:
                self.play = False
        else:
            # Parochialism
            self.read_signal = False
            if self.stranger_partner:
                # Also assume self as newcomer to distrust strangers
                self.play = False
            else:
                self.play = True

    def get_signal(self) -> PDTChoice:
        """ Returns the choice the agent makes on whether to cooperate or defect in the
            prisoners' dilemma, based on the reading of the signals the other agent
            'shows'.

            At a trustworthiness of 0.5 the agents is ambivalent.
            At either 0 or 1 the signal is assumed to be perfect.
            The signal correctness is linearly interpolated between those values.
        """
        signal_correctness = 0.5 + abs(self.trustworthiness_prob - 0.5)
        if self.random.random() < signal_correctness:
            return self.pdtchoice

        # Returns opposite signal of the PDT choice
        return PDTChoice.COOPERATE if self.pdtchoice == PDTChoice.DEFECT else PDTChoice.DEFECT


class RLAgent(WHAgent):
    """ Implementation of the Reinforcement Learning agent, extends a WHAgent.

        Similar to the WHAgent, though the learning rate is no longer stochastic
        but based on reinforcement learning.
    """

    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int,
                 learning_rate: float, relative_reward: bool = False) -> None:
        super().__init__(unique_id, model, neighbourhood)

        self.total_payoff = 0
        self.n_payoffs = 0

        # The learning rate for the reinforcement learning mechanism
        self.learning_rate = learning_rate
        # Whether or not to use the relative reward for learning
        self.relative_reward = relative_reward

    def receive_payoff(self, payoff):
        """ Saves the payoff of the current step and adds it to the agent's cumulative payoff.
            Also updates the total payoff value, and the total amount of payoffs.
        """
        super().receive_payoff(payoff)
        self.total_payoff += payoff
        self.n_payoffs += 1

    # Overrides default stochastic learning behaviour
    def stochastic_learning(self, prob: float, payoff: float) -> float:
        """ Calculates and returns a new value according to the reinforcement learning rate
            given the probability, current payoff and average payoff.
        """
        if (self.relative_reward and self.n_payoffs > 0):
            # Use the relative reward which is the current reward minus the average reward
            payoff = payoff - self.total_payoff / self.n_payoffs

        if payoff >= 0:
            return prob + self.learning_rate * (1 - prob) * payoff
        else:
            return prob + self.learning_rate * prob * payoff


class GossipAgent(WHAgent):
    """ Implementation of the Gossip agent, extends a WHAgent.

        This agent asks the role model whether or not the agent he has
        been matched with can be considered trustworthy, i.e. it takes the
        advice from the role model to decide whether or not to cooperate in the
        prisonters' dilemma.
    """

    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int,
                 num_agents: int) -> None:
        """ Complementary to the init of its super, this agent also creates
            a list of previous experiences with other agents.
        """
        super().__init__(unique_id, model, neighbourhood)

        self.memories = [None] * num_agents

    def decide_play(self, exchange_partner) -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma.
            
            First, the role model is updated. If the agent has had a positive
            previous experiment with this role model, the agent will ask the
            role model for advice. 

            The role model can only give advice if he has had a previous encounter
            with the agent that he's asked advice about. 

            If the agent receives advice from the role model, it will trust this
            advice and decide to cooperate or defect accordingly. In case the
            role model can't give advice, the agent will fall back to its ability
            to read the signals of the other agent, similar to the behavior of
            the WHAgent.
        """
        super().decide_play(exchange_partner)

        role_model = self.model.network.get_role_model(self.neighbourhood)

        # See if you trust the rolemodel
        # if self.memories[role_model.unique_id] == 1:
        #     advice = role_model.memories[exchange_partner.unique_id]
        #     # Take over the advice from the role model, if there is any advice
        #     if advice is not None:
        #         self.play = 1 if advice == 1 else 0

        advice = role_model.memories[exchange_partner.unique_id]
        if advice is not None:
            self.play = 1 if advice == 1 else 0
        else:
            # If you don't trust him, use signal reading
            self.read_signal = True
            signal = exchange_partner.get_signal()
            if signal == PDTChoice.COOPERATE:
                self.play = True
            else:
                self.play = False

    def memorize(self, exchange_partner, payoff) -> None:
        """ TODO
        """
        if payoff > 0:
            self.memories[exchange_partner.unique_id] = 1
        else:
            self.memories[exchange_partner.unique_id] = 0
