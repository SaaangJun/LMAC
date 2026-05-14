from .Discriminator_net import Discriminator_net
from .meta import meta




REGISTRY = {"Discriminator_net": Discriminator_net,
            "meta": meta
            }
