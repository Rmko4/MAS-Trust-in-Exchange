""" This file contains the enumeration for the choices in the prisoners' dilemma.
"""
from enum import Enum


class PDTChoice(Enum):
    """ Contains the two possible choices when participaiting in the prisoners' dilemma:
        defect or cooperate.
    """
    DEFECT = 0
    COOPERATE = 1
