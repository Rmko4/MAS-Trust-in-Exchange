""" This file defines the scheduler used for the model.
"""
from typing import TYPE_CHECKING, Iterator, List

from mesa.time import BaseScheduler

if TYPE_CHECKING:
    from trust.agent import BaseAgent


class TwoStepActivation(BaseScheduler):
    """ This class represents the scheduler created for the model.
    """

    def step(self) -> None:
        """ Executes the step method of all agents, one at a time.
        """
        for agent in self.agent_buffer(shuffled=False):
            agent.step()
        self.time += .5

    def finalize(self) -> None:
        """ Executes the finalize method of all agents, one at a time.
            After completing all finalize methods, it moves to the next step.
        """
        for agent in self.agent_buffer(shuffled=False):
            agent.finalize()
        self.time += .5
        self.steps += 1

    '''Used for type checking
    '''
    @property
    def agents(self) -> List['BaseAgent']:
        return super().agents

    def agent_buffer(self, shuffled: bool = False) -> Iterator['BaseAgent']:
        return super().agent_buffer(shuffled=shuffled)
