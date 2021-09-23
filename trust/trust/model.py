from trust.agent import PDTAgent
from trust.network import Network
from trust.choice import PDTChoice
from mesa import Model
from mesa.time import RandomActivation


class PDTModel(Model):
    _PDT_PAYOFF = {(PDTChoice.DEFECT, PDTChoice.COOPERATE): 1,  # Without opportunity cost
                   # Without opportunity cost
                   (PDTChoice.COOPERATE, PDTChoice.COOPERATE): 0.7,
                   (PDTChoice.DEFECT, PDTChoice.DEFECT): -0.2,
                   (PDTChoice.COOPERATE, PDTChoice.DEFECT): -0.5,
                   }
    EXIT_PAYOFF = -0.2

    def opportunity_cost(self, neighbourhood_size: int) -> float:
        return 1 - (neighbourhood_size - 1) / (self.num_agents - 1)

    def pdt_payoff(self, choices: 'tuple[int, int]', opportunity_cost: int) -> float:
        payoff = self._PDT_PAYOFF[choices]
        if choices[1] == PDTChoice.COOPERATE:
            payoff -= 0.5 * opportunity_cost
        return payoff

    def __init__(self, N=1000, neighbourhood_size=50, mobility_rate=0.2) -> None:
        self.num_agents = N
        self.num_neighbourhoods = int(self.num_agents / neighbourhood_size)

        self.network = Network(self, self.num_neighbourhoods)
        self.schedule = RandomActivation(self)

        self.mobility_rate = mobility_rate

        for i in range(self.num_agents):
            neighbourhood = int(i % self.num_neighbourhoods)
            # TODO: Cluster agent location into neighborhoods of randomly varying size.
            a = PDTAgent(i, self, neighbourhood)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
        self.network.pair_and_play()
        self.network.clean_up()

    def run_model(self, T=2000) -> None:
        for _ in range(T):
            self.step()
        self.running = False
