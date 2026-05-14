from .rnn_agent import RNNAgent
from .lmac_agent import LMACAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["lmac"] = LMACAgent
