from trust.model import PDTChoice
from trust.model import PDTModel
from mesa import Agent


class PDTAgent(Agent):
    def __init__(self, unique_id: int, model: PDTModel, neighbourhood: int) -> None:
        super().__init__(unique_id, model)
        self.neighbourhood = neighbourhood
        self.model.network.add_agent(self, neighbourhood)
        self.newcomer = False

        self.trust_prob = self.random.random()
        self.thrustworthiness_prob = self.random.random()
        self.location_prob = self.random.random()

        self.play = True
        self.pdtchoice = PDTChoice.COOPERATE
        self.in_market = True

        self.payoff = 0
        self.cumulative_payoff = 0

    def step(self) -> None:
        # Change neighbourhood involuntary
        if self.random.random() < self.model.mobility_rate:
            self.move()
            self.cumulative_payoff = 0
            self.enter_market()  # TODO: Should newcomers enter the market?
        else:
            self.stay()

        # Chose to exchange in neighbourhood or market
        if self.random.random() < self.location_prob:
            self.enter_market()

    def move(self) -> None:
        new_nbh = self.random.randint(0, self.model.num_neighbourhoods)
        # Ensure that it won't stay in the same neighbourhood
        if new_nbh >= self.neighbourhood:
            new_nbh = new_nbh + 1 % self.model.num_neighbourhoods
        self.model.network.add_agent(self, new_nbh)
        self.newcomer = True

    def stay(self) -> None:
        self.newcomer = False

    def enter_market(self) -> None:
        self.model.network.add_agent_to_market(self)
        self.in_market = True

    def decide_exchange(self) -> None:
        if self.random.random() < self.model.trust:
            self.play = True
        else:
            self.play = False

        if self.random.random() < self.model.trustworthiness:
            self.pdtchoice = PDTChoice.COOPERATE
        else:
            self.pdtchoice = PDTChoice.DEFECT

    def update_payoff(self, payoff):
        self.payoff = payoff
        self.cumulative_payoff += payoff

    def stochastic_learning(self, prob: float, payoff: float) -> float:
        if payoff >= 0:
            return prob + (1 - prob) * payoff
        else:
            return prob + prob * payoff

    def update_behaviour(self):
        role_model = self.model.network.get_role_model(self.neighbourhood)
        if self.random > 0.5:
            self.location_prob = role_model.location_prob
        elif self.in_market == False:
            self.location_prob = 1 - \
                self.stochastic_learning(1 - self.location_prob, self.payoff)
        else:
            self.location_prob = self.stochastic_learning(
                self.location_prob, self.payoff)


        if self.random > 0.5:
            self.trust_prob = role_model.trust_prob
        elif self.play == False:
            self.trust_prob = 1 - \
                self.stochastic_learning(1 - self.trust_prob, self.payoff)
        else:
            self.trust_prob = self.stochastic_learning(
                self.trust_prob, self.payoff)

        if self.random > 0.5:
            self.thrustworthiness_prob = role_model.thrustworthiness_probb
        elif self.pdtchoice == PDTChoice.COOPERATE:
            self.thrustworthiness_prob = 1 - \
                self.stochastic_learning(
                    1 - self.thrustworthiness_prob, self.payoff)
        else:
            self.thrustworthiness_prob = self.stochastic_learning(
                self.thrustworthiness_prob, self.payoff)

