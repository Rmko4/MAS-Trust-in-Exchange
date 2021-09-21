from mesa import Model
from mesa import Agent


class PDTAgent(Agent):
    def __init__(self, unique_id: int, model: Model, neighbourhood: int) -> None:
        super().__init__(unique_id, model)
        self.neighbourhood = neighbourhood
        self.model.network.add_agent(self, neighbourhood)
        self.newcomer = False

        self.location_prob = self.random.random()

    def step(self) -> None:
        # Change neighbourhood involuntary
        if self.random.random() < self.model.mobility_rate:
            self.move()
        else:
            self.stay()
            
        # Chose to exchange in neighbourhood or market
        if self.random.random() < self.location_prob:
            self.enter_market()

    def advance(self) -> None:
        pass

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