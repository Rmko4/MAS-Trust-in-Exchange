from mesa.time import BaseScheduler


class TwoStepActivation(BaseScheduler):

    def step(self) -> None:
        """Executes the step of all agents, one at a time.

        """
        for agent in self.agent_buffer(shuffled=False):
            agent.step()
        self.time += .5

    def finalize(self) -> None:
        """Executes the finalize of all agents, one at a time.

        """
        for agent in self.agent_buffer(shuffled=False):
            agent.finalize()
        self.time += .5
        self.steps += 1
