""" This file contains the classes defining different types of agents.
"""
from typing import TYPE_CHECKING

from mesa import Agent
from utils.dictionary import LimitedDict

from trust.choice import PDTChoice

if TYPE_CHECKING:
    from trust.model import PDTModel


class BaseAgent(Agent):
    """ Defines a base agent, which is an implementation of an Agent as defined by the MESA module.
    """
    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int) -> None:
        """ Initializes the baseAgent. Saves the neighbourhood that the agent is in, and does not
            mark it as a newcomer. Initializes the propensity to read signals (antagonist
            parochialism), propensity to cooperate and propensity to enter the open market (over
            staying in the neighbourhood) to random values between 0.0 and 1.0. Also, the social
            learning rate is set to 0.5 and the variable keeping track of both the current and
            cumulative payoffs are set to zero. This variable will be updated after every
            interaction with another agent, based on the outcome of the prisoners' dilemma
            (or the exit cost if the agents decides not to play).
        """
        super().__init__(unique_id, model)
        self.model: 'PDTModel'

        self.neighbourhood = neighbourhood
        self.newcomer = False

        # Equivalent to the propensity to play or read signals
        self.trust_prob = self.random.random()
        # Propensity to cooperate (over defect)
        self.trustworthiness_prob = self.random.random()
        # Propensity to enter the open market
        self.location_prob = self.random.random()

        self.social_learning_rate = 0.5

        self.play = True
        self.pdtchoice = PDTChoice.COOPERATE
        self.in_market = False
        self.paired = False

        self.partner_is_stranger = False
        self.partner_is_newcomer = False

        self.payoff = 0
        self.cumulative_payoff = 0

    def step(self) -> None:
        """ Every step, the agent involuntarily moves to a new neigbhourhood with a certain probability
            as defined by the mobility rate. Also, the agent chooses (based on the location
            probability) wether to stay in the neighbourhood or move to the global market
            for its next interaction.
        """
        if self.random.random() < self.model.mobility_rate:
            self.move()
        else:
            self.stay()

        if self.random.random() < self.location_prob:
            self.enter_market()

    def finalize(self) -> None:
        """ If the agent is paired with another agent, it will update its behaviour (i.e. its
            decision to play/exit and to cooperate/defect in the prisoners' dilemma).
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
        """ Updates whether or not the agent the current agent has been matched with is
            a newcomer, and whether or not it is a stranger. This method is called from the
            subclass (so for example the GossipAgent) in their own decide_play method, where
            the agent still needs to decide to play for itself as this is different per agent
            type. 
        """
        self.paired = True
        self.exchange_partner = exchange_partner

        if not self.in_market and (exchange_partner.newcomer or self.newcomer):
            self.partner_is_newcomer = True
        else:
            self.partner_is_newcomer = False

        if exchange_partner.newcomer or self.newcomer or self.in_market:
            self.partner_is_stranger = True
        else:
            self.partner_is_stranger = False

    def move(self) -> None:
        """ Moves an agent to a different neighbourhood than it is in now, also marks
            the agent as a newcomer and resets its cumulative payoff.
        """
        new_nbh = self.random.randint(0, self.model.num_neighbourhoods - 1)
        if new_nbh >= self.neighbourhood:
            new_nbh = (new_nbh + 1) % (self.model.num_neighbourhoods - 1)
        self.model.network.add_agent_to_neighbourhood(self, new_nbh)

        self.newcomer = True

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

    def receive_payoff(self, payoff: float):
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

    def update_propensity(self, action_prob_attr: str, action_test: bool) -> None:
        """ Updates the passed propensity according to the learning rate. At this moment, this
            is used to update the propensity to play or read signals, the propensity to
            cooperate (over defect) and the propensity to enter the open market.
        """
        role_model = self.model.network.get_role_model(self.neighbourhood)
        social_learning = role_model is not None and role_model is not self

        if social_learning and self.random.random() < self.social_learning_rate:
            prob = getattr(role_model, action_prob_attr)
        elif not action_test:
            prob = 1 - \
                self.stochastic_learning(
                    1 - getattr(self, action_prob_attr), self.payoff)
        else:
            prob = self.stochastic_learning(
                getattr(self, action_prob_attr), self.payoff)
        setattr(self, action_prob_attr, prob)

    def update_behaviour(self):
        """ Updates the propensities of the agent.

            The values for the propensity to enter the global market,
            propensity to either cooperate or defect are updated according to
            the learning rate and the propensity to read signals (I.e. base
            its decision to either cooperate or defect on reading signals
            instead of Parochialism).
        """
        self.update_propensity('location_prob', self.in_market)
        self.update_propensity('trustworthiness_prob',
                               self.pdtchoice == PDTChoice.COOPERATE)
        self.update_propensity('trust_prob', self.play)

    @property
    def trust_in_stranger(self) -> bool:
        """ Returns true if the agent is matched with a stranger and decided
            to participate in the prisoners dilemma.
        """
        return self.play and self.partner_is_stranger and self.paired

    @property
    def paired_with_stranger(self) -> bool:
        """ Returns true if the agent is matched with a stranger.
        """
        return self.partner_is_stranger and self.paired


class MSAgent(BaseAgent):
    """ Implementation of the Macy and Sato agent, extends a BaseAgent.

        For this agent, the dicision to either play or walk away from an opportunity to play
        the prisoners' dilemma with the agent he is matched with is based on the propensity
        to read signals.
    """

    def decide_play(self, exchange_partner: 'MSAgent') -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma.
        
            First, the agents propensities are updated in the decide_play method of the
            super (BaseAgent). After this, the agent will play or exit based on the updated
            propensity to play or exit.
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

    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int) -> None:
        """ Initializes the WHAgent. In addition to the initialization done for the BaseAgent,
            the reading of signals is set to false.  
        """
        super().__init__(unique_id, model, neighbourhood)
        self.read_signal = False

    def decide_play(self, exchange_partner: 'WHAgent') -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma.

            First, the agents propensities are updated in the decide_play method of the
            super (BaseAgent). Based on the propensity to either read signals or act
            parochial, the agent will decide to either cooperate or defect. 

            If the agent decides to read the signals of the other agent, the outcome of
            the signal reading determines the agents' choice to either cooperate or defect.

            If the agent decides to act parochial, it will only trust the other agent if
            it knows its opponent (so it is not on the global market, and the opponent
            is not a newcomer to the neighbourhood, and the agent itself is not a newcomer
            to the neighbourhood).
        """
        super().decide_play(exchange_partner)

        if self.random.random() < self.trust_prob:
            self.signal_reading()
        else:
            self.parochialism()

    def signal_reading(self):
        """ Determines whether or not to trust (so play/exit) the opponent, based on
            the reading of signals. If the reading of signals shows that the opponent
            will cooperate, the agent will trust the opponent. Otherwise, the agent
            will not trust the opponent and exit the game. 
        """
        self.read_signal = True
        exchange_partner: 'WHAgent' = self.exchange_partner

        signal = exchange_partner.get_signal()
        if signal == PDTChoice.COOPERATE:
            self.play = True
        else:
            self.play = False

    def parochialism(self):
        """ Determines whether or not to trust (so play/exit) the opponent, based on
            parochialism. If a partner is a stranger (so either the agent itself or
            its partner is a newcomer, or you have entered the global market) the agent
            decides not to trust the opponent. Otherwise, it will trust the opponent and
            play the prisoners' game.
        """
        self.read_signal = False
        if self.partner_is_stranger:
            self.play = False
        else:
            self.play = True

    def get_signal(self) -> PDTChoice:
        """ Returns the interpretation of the opponents signalled decision on
            cooperating/defecting. Based on the signal of the opponent, and the
            agents ability to read this signal.

            At a trustworthiness of 0.5 the agents is ambivalent.
            At either 0 or 1 the signal is assumed to be perfect.
            The signal correctness is linearly interpolated between those values.
        """
        signal_correctness = 0.5 + abs(self.trustworthiness_prob - 0.5)
        if self.random.random() < signal_correctness:
            return self.pdtchoice

        # Returns opposite signal of the PDT choice
        return PDTChoice.COOPERATE if self.pdtchoice == PDTChoice.DEFECT else PDTChoice.DEFECT

    def update_behaviour(self):
        """ Updates the propensities of the agent.

            The values for the propensity to enter the global market,
            propensity to either cooperate or defect are updated according to
            the learning rate and the propensity to read signals (I.e. base
            its decision to either cooperate or defect on reading signals
            instead of Parochialism). Note that the updating of the trust probabiltiy
            is now updated on the action of reading the signal.
        """
        self.update_propensity('location_prob', self.in_market)
        self.update_propensity('trustworthiness_prob',
                               self.pdtchoice == PDTChoice.COOPERATE)
        self.update_propensity('trust_prob', self.read_signal)


class RLAgent(WHAgent):
    """ Implementation of the Reinforcement Learning agent, extends a WHAgent.

        Similar to the WHAgent, though the learning rate is no longer stochastic
        but based on a new implementation of reinforcement learning.
    """

    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int,
                 learning_rate: float, discount_factor: float = 0.9,
                 social_learning_rate: float = 0.5, relative_reward: bool = False) -> None:
        """ Initializes the RLAgent. In addition to the initialization of the WHAgent
            the RLAgent also initializes the total number of payoffs, the discount factor,
            the learning rate, the social learning rate and the relative reward.
        """
        super().__init__(unique_id, model, neighbourhood)
        self.n_payoffs = 0

        # The factor with which the previous cumulative reward is discounted
        self.discount_factor = discount_factor
        # The learning rate for the reinforcement learning mechanism
        self.learning_rate = learning_rate
        # The learning rate for social learning from the role model
        self.social_learning_rate = social_learning_rate
        # Whether or not to use the relative reward for learning
        self.relative_reward = relative_reward

    def receive_payoff(self, payoff: float):
        """ Saves the payoff of the current step and adds it to the agent's cumulative payoff.
            Also updates the total payoff value, and the total amount of payoffs.
        """
        self.payoff = payoff
        self.cumulative_payoff = payoff + self.discount_factor * self.cumulative_payoff
        self.n_payoffs = 1 + self.discount_factor * self.n_payoffs

    def stochastic_learning(self, prob: float, payoff: float) -> float:
        """ Overrides the default stochastic learning behaviour.
        
            Calculates and returns a new value according to the reinforcement learning rate
            given the probability, current payoff and average payoff.
        """
        if (self.relative_reward and self.n_payoffs > 0):
            # Use the relative reward which is the current reward minus the average reward
            payoff = payoff - self.cumulative_payoff / self.n_payoffs

        if payoff >= 0:
            return prob + self.learning_rate * (1 - prob) * payoff
        else:
            return prob + self.learning_rate * prob * payoff


class BaseGossipAgent(WHAgent):
    """ Implementation of the BaseGossip agent, extends a WHAgent.

        This agent asks the role model whether or not the agent he has
        been matched with can be considered trustworthy, i.e. it takes the
        advice from the role model to decide whether or not to cooperate in the
        prisoners' dilemma.
    """

    def __init__(self, memory_size: int) -> None:
        """ Initializes the BaseGossipAgent
        """
        self.memories: LimitedDict[int, bool] = LimitedDict(memory_size)

    def finalize(self) -> None:
        """ Finalize step of the BaseGossipAgent. Memorizes the outcome of
            the last exchange, if both agents decided to play on top of the
            finilize step of the BaseAgent (the super of the WHAgent).
        """
        if self.paired and self.play and self.exchange_partner.play:
            self.memorize_trust()
        return super().finalize()

    def decide_play(self, exchange_partner: 'BaseGossipAgent') -> None:
        """ Updates the agents decision to play or exit a prisoners' dilemma.

            First, the propensities of the agent are updated by the super method.
            
            If the agent memorizes a previous experience with its opponent, this will
            determine its decicion to play/exit. If not, it will ask the role model for
            advice which it will then base its decision on. If the role model also has
            no recollection of a previous encounter with the opponent, the agent will
            go back to the default and base its decision on signal reading.
        """
        super(WHAgent, self).decide_play(exchange_partner)

        role_model: 'BaseGossipAgent' = self.model.network.get_role_model(
            self.neighbourhood)

        partner_id = self.exchange_partner.unique_id

        if partner_id in self.memories:
            memory = self.memories[partner_id]
            self.play = memory
        elif partner_id in role_model.memories:
            advice = role_model.memories[partner_id]
            self.play = advice
        else:
            if self.random.random() < self.trust_prob:
                self.signal_reading()
            else:
                self.parochialism()

    def memorize_trust(self) -> None:
        """ Memorizes the encounter with the opponent, save wehter or not
            this was a positive encounter.
        """
        if self.payoff > 0:
            self.memories[self.exchange_partner.unique_id] = True
        else:
            self.memories[self.exchange_partner.unique_id] = False


class GossipAgent(BaseGossipAgent):
    """ Implementation of the GossipAgent, extends the BaseGossipAgent and WHAgent.
    """
    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int,
                 memory_size: int) -> None:
        """ Initializes the GossipAgent.
        """
        super().__init__(memory_size)
        super(WHAgent, self).__init__(unique_id, model, neighbourhood)


class RLGossipAgent(BaseGossipAgent, RLAgent):
    """ Implementation of the RLGossipAgent, extends the BaseGossipAgent and RLAgent.
    """
    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int, **kwargs) -> None:
        """ Initializes the RLGossipAgent.
        """
        memory_size = kwargs.pop('memory_size')
        super().__init__(memory_size)
        RLAgent.__init__(self, unique_id, model, neighbourhood, **kwargs)
