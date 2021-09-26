from mesa.time import BaseScheduler


class TwoStepActivation(BaseScheduler):

    def step(self) -> None:
        """Executes the step of all agents, one at a time, in
        random order.

        """
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        self.steps += .5

    def finalize(self) -> None:
        """Executes the finalize of all agents, one at a time, in
        random order.

        """
        for agent in self.agent_buffer(shuffled=True):
            agent.finalize()
        self.steps += .5
        self.time += 1
