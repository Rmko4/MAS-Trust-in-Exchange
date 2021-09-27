from mesa.time import BaseScheduler


class TwoStepActivation(BaseScheduler):

    def step(self) -> None:
        """Executes the step of all agents, one at a time, in
        random order.

        """
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        self.time += .5

    def finalize(self) -> None:
        """Executes the finalize of all agents, one at a time, in
        random order.

        """
        for agent in self.agent_buffer(shuffled=True):
            agent.finalize()
        self.time += .5
        self.steps += 1
