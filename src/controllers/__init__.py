REGISTRY = {}

from .basic_controller import BasicMAC
from .lmac_controller import LMAC_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["lmac_mac"] = LMAC_MAC
