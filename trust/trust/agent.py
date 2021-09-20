from mesa import Agent
from mesa import Model

class PDTAgent(Agent):
    def __init__(self, unique_id: int, model: Model, neighbourhood: int) -> None:
        super().__init__(unique_id, model)
        self.neighbourhood = neighbourhood


