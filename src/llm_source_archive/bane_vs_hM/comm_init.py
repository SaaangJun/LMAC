import torch as th

def message_design_instruction():
    """
    Communication Protocol Description:

    To achieve perfect unison explosion of the three Banelings against the Hydralisk,
    agents must share critical spatial and health information that is not locally observable
    due to limited field of view and partial observability.

    Key communicated information per agent includes:
    1. Own absolute position (X, Y) - since observations only provide relative positions,
       sharing absolute coordinates enables global spatial awareness and precise synchronization.
    2. Own health ratio - critical to ensure all agents are alive and capable of striking simultaneously.
    3. Sender identity (one-hot vector of length 3) - to distinguish message origin.

    This message is broadcasted to all agents, enabling each to reconstruct the global state
    of all teammates and coordinate their timing and positioning for the synchronized explosion.

    The message is compact (6 dims = 2 position + 1 health + 3 sender id bits),
    unique (contains self info not locally observable by others),
    sufficient (enables inference of teammate readiness and position),
    and explicit (no abstract encoding).

    This protocol balances communication overhead with the need for precise coordination in a partially observable,
    multi-agent environment with critical timing constraints.
    """
    return message_design_instruction.__doc__


def communication(o):
    """
    Input:
        o: torch.Tensor of shape (batch_size=32, n_agents=3, obs_dim=42)
           Observation tensor per agent per batch.

    Output:
        messages_o: torch.Tensor of shape (32, 3, 42 + message_dim)
           Enhanced observation tensor with integrated messages appended.

    Message design:
        For each agent, construct a message containing:
        - Own relative position to enemy (indices 6,7) as proxy for absolute position.
        - Own health ratio (index 29).
        - Sender identity (indices 39-41).

    Communication type: broadcast - all agents receive messages from others.
    """

    batch_size, n_agents, obs_dim = o.shape
    device = o.device

    # Extract required fields per agent:
    rel_enemy_x = o[:, :, 6]  # shape (batch, agents)
    rel_enemy_y = o[:, :, 7]

    # Own health ratio: idx 29
    own_health = o[:, :, 29]

    # Agent ID one-hot: indices 39-41 (3 dims)
    agent_id = o[:, :, 39:42]  # shape (batch, agents, 3)

    # Construct message tensor per agent: (rel_enemy_x, rel_enemy_y, own_health, agent_id(3))
    # message_dim = 1 + 1 + 1 + 3 = 6 dims
    own_health = own_health.unsqueeze(-1)  # (batch, agents, 1)

    message = th.cat([rel_enemy_x.unsqueeze(-1), rel_enemy_y.unsqueeze(-1), own_health, agent_id], dim=-1)
    # shape: (batch, agents, 6)

    message_dim = message.shape[-1]

    # Expand dims for broadcasting
    message_exp = message.unsqueeze(2)  # (batch, sender_agent, 1, message_dim)
    message_exp = message_exp.repeat(1, 1, n_agents, 1)  # (batch, sender_agent, receiver_agent, message_dim)

    # Create mask to zero out self messages
    sender_ids = th.arange(n_agents, device=device).view(1, n_agents, 1)
    receiver_ids = th.arange(n_agents, device=device).view(1, 1, n_agents)

    self_mask = (sender_ids == receiver_ids).expand(batch_size, -1, -1)
    message_exp = message_exp.masked_fill(self_mask.unsqueeze(-1), 0.0)

    # Permute to (batch, receiver_agent, sender_agent, message_dim)
    message_exp = message_exp.permute(0, 2, 1, 3)

    # Indices of other agents
    sender_indices = th.tensor([[1, 2],
                               [0, 2],
                               [0, 1]], device=device)  # (3, 2)
    sender_indices = sender_indices.unsqueeze(0).expand(batch_size, -1, -1)

    # Gather messages
    messages_from_others = th.gather(
        message_exp, 2,
        sender_indices.unsqueeze(-1).expand(-1, -1, -1, message_dim)
    )  # (batch, receiver, 2, message_dim)

    # Concatenate messages from the two other agents
    messages_concat = messages_from_others.reshape(batch_size, n_agents, -1)  # (batch, 3, 12)

    # Append to original observations
    messages_o = th.cat([o, messages_concat], dim=-1)  # (batch, 3, 42 + 12)

    return messages_o
