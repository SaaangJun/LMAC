from .q_learner import QLearner

from .lmac_learner import LMAC_learner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["lmac_learner"] = LMAC_learner
