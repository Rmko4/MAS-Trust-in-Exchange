from typing import TYPE_CHECKING
from mesa import Agent

from trust.choice import PDTChoice
if TYPE_CHECKING:
    from trust.model import PDTModel


class PDTAgent(Agent):
    def __init__(self, unique_id: int, model: 'PDTModel', neighbourhood: int) -> None:
        super().__init__(unique_id, model)
        self.neighbourhood = neighbourhood
        self.newcomer = False

        # Equivalent to the propensity to read signals (antagonist parochialism)
        self.signal_reading_prob = self.random.random()
        # Propensity to cooperate (over defect)
        self.trustworthiness_prob = self.random.random()
        # Propensity to enter the open market (over staying in the neighbourhood)
        self.location_prob = self.random.random()

        self.play = True
        self.pdtchoice = PDTChoice.COOPERATE
        self.in_market = False
        self.read_signal = False

        self.payoff = 0
        self.cumulative_payoff = 0

        self.total_payoff = 0
        self.n_payoffs = 0

        #the learning rate for the reinforcement learning mechanism
        self.learning_rate = 0.05
        #wather or not to use the relative reward for learning 
        self.relative_reward = False


    def step(self) -> None:
        # Change neighbourhood involuntary
        if self.random.random() < self.model.mobility_rate:
            self.move()
        else:
            self.stay()

        # Chose to exchange in neighbourhood or market
        if self.random.random() < self.location_prob:
            self.enter_market()

    def finalize(self) -> None:
        self.update_behaviour()
        self.leave_market()

    def decide_cooperation(self) -> None:
        # Trustworthiness is not model conditionally here,
        # therefore it remains uniform across conditions (neighbourhood or open market)
        if self.random.random() < self.trustworthiness_prob:
            self.pdtchoice = PDTChoice.COOPERATE
        else:
            self.pdtchoice = PDTChoice.DEFECT

    def decide_play(self, exchange_partner : 'PDTAgent') -> None:

        #needed to plot results
        if not self.in_market and (exchange_partner.newcomer or self.newcomer) :
            self.partern_Is_Newcommer = True
        else:
            self.partern_Is_Newcommer = False

        if exchange_partner.newcomer or self.newcomer or self.in_market:
            self.stranger_partner = True
        else:
            self.stranger_partner = False

        if self.random.random() < self.signal_reading_prob:
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

    def move(self) -> None:
        new_nbh = self.random.randint(0, self.model.num_neighbourhoods - 1)
        # Ensure that it won't stay in the same neighbourhood
        if new_nbh >= self.neighbourhood:
            new_nbh = (new_nbh + 1) % (self.model.num_neighbourhoods - 1)
        self.model.network.add_agent_to_neighbourhood(self, new_nbh)

        self.newcomer = True
        self.cumulative_payoff = 0

    def stay(self) -> None:
        self.newcomer = False

    def enter_market(self) -> None:
        self.model.network.add_agent_to_market(self)
        self.in_market = True

    def leave_market(self) -> None:
        self.model.network.remove_agent_from_market(self)
        self.in_market = False

    def get_signal(self) -> PDTChoice:
        # At a trustworthiness of 0.5 the agents is ambivalent.
        # At either 0 or 1 the signal is assumed to be perfect.
        # The signal correctness is linearly interpolated between those values
        signal_correctness = 0.5 + abs(self.trustworthiness_prob - 0.5)
        if self.random.random() < signal_correctness:
            return self.pdtchoice
        else:
            # Returns opposite signal of the PDT choice
            return PDTChoice.COOPERATE if self.pdtchoice == PDTChoice.DEFECT else PDTChoice.DEFECT

    def receive_payoff(self, payoff):
        self.payoff = payoff
        self.cumulative_payoff += payoff
        self.total_payoff += payoff
        self.n_payoffs +=1

    def update_behaviour(self):

        def stochastic_learning(prob: float, payoff: float) -> float:
            if (self.relative_reward and self.n_payoffs > 0):
                #Use the relative reward which is the current reward minus the average reward
                payoff = payoff - self.total_payoff / self.n_payoffs

            if payoff >= 0:
                return prob + self.learning_rate * (1 - prob) * payoff
            else:
                return prob + self.learning_rate * prob * payoff

        role_model = self.model.network.get_role_model(self.neighbourhood)
        if role_model is not None and self.random.random() > 0.5:
            self.location_prob = role_model.location_prob
        elif self.in_market == False:
            self.location_prob = 1 - \
                stochastic_learning(1 - self.location_prob, self.payoff)
        else:
            self.location_prob = stochastic_learning(
                self.location_prob, self.payoff)

        if role_model is not None and self.random.random() > 0.5:
            self.signal_reading_prob = role_model.signal_reading_prob
        elif self.read_signal == False:
            self.signal_reading_prob = 1 - \
                stochastic_learning(1 - self.signal_reading_prob, self.payoff)
        else:
            self.signal_reading_prob = stochastic_learning(
                self.signal_reading_prob, self.payoff)

        #Comment by Lukas: Is it right that the trustworthiness is always updated, even if the game of PF was not played? 
        if role_model is not None and self.random.random() > 0.5:
            self.trustworthiness_prob = role_model.trustworthiness_prob
        elif self.pdtchoice == PDTChoice.COOPERATE:
            self.trustworthiness_prob = 1 - \
                stochastic_learning(
                    1 - self.trustworthiness_prob, self.payoff)
        else:
            self.trustworthiness_prob = stochastic_learning(
                self.trustworthiness_prob, self.payoff)
