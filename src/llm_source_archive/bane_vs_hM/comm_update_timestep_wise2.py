import torch as th

def message_design_instruction():
    """
    Enhanced Temporal-Spatial Coordination Communication Protocol:

    To address the uneven prediction accuracy and inconsistent inference in absolute coordinates and synchronization,
    this protocol extension enables agents to share temporally aggregated and behaviorally contextualized information
    that cannot be inferred locally due to limited field of view and partial observability.

    Key communicated information per agent is constructed from the last T=10 timesteps of observations:

    1. Temporal Aggregation of Own Movement & Action History (6 dims):
       - Summed normalized counts of movement attempts in each direction (N, S, E, W) over the last 10 timesteps
         (o[...,0:4] summed over T dimension, normalized by T, continuous [0,1])
       - Summed normalized counts of attack and no-op actions over last 10 timesteps
         (o[...,37] Attack + o[...,31] No-op, normalized by T, continuous [0,1])
       This provides a compact behavioral embedding reflecting the agent's recent intentions and mobility,
       aiding others in inferring movement trends and predicting future positions.

    2. Most Recent Reliable Absolute Position Anchor (2 dims):
       - Estimated absolute position coordinates of the agent at the last timestep where the agent was near an intersection
         (where it can "see" allies or is at a known map feature, approximated here as the relative positions to allies when visible)
       - Since absolute positions are not directly observable, we approximate by sharing the relative positions to allies if visible,
         or zeros if not visible in the last timestep.
       This acts as a positional anchor that helps receivers align their internal state estimates spatially.

    3. Freshness Indicator (1 dim):
       - Normalized timestep index (current timestep / max timestep T-1) indicating message staleness and enabling temporal alignment.
       - This allows agents to weigh information based on recency.


    Communication type: broadcast.

    Message dimension: 6 (agg movements & actions) + 2 (positional anchor) + 1 (freshness) = 9 dims.

    This message complements the previous protocol by:
      - Providing temporal context via aggregated recent behaviors.
      - Sharing approximate positional anchors to reduce spatial ambiguity.
      - Including freshness info to synchronize temporal understanding.

    Together with previous messages, this supports precise, temporally aligned coordination under partial observability,
    improving consistent inference of absolute positions and synchronized attack timing.

    Implementation notes:
      - Vectorized, no explicit loops over batch, time, or agents.
      - Aggregates temporal info reducing (batch, T, agents, obs_dim) -> (batch, agents, message_dim).
      - Excludes all info covered in previous protocol to avoid redundancy.
    """
    return message_design_instruction.__doc__


def communication(o):
    """
    Input:
        o: torch.Tensor of shape (batch_size=32, T=10, n_agents=3, obs_dim=42)
           Observation tensor for each agent over last T timesteps.

    Output:
        messages_o: torch.Tensor of shape (batch_size=32, n_agents=3, 42 + 9)
           Enhanced observation tensor with integrated new task-specific message appended.

    Message design (new fields only):
        For each agent, compute from last T timesteps:
        - Aggregated normalized movement attempts in N, S, E, W directions (4 dims)
        - Aggregated normalized counts of Attack and No-op actions (2 dims)
        - Most recent relative position to ally (x, y) at last timestep (2 dims)
        - Freshness indicator: normalized current timestep index (1 dim)

        (Removed: Sender ID)

    Communication type: broadcast.

    Implementation details:
        - Use vectorized operations, avoid explicit loops.
        - Append concatenated messages from other agents (excluding self) to each agent's current observation at last timestep.
    """
    batch_size, T, n_agents, obs_dim = o.shape
    device = o.device
    dtype = o.dtype

    # 1) Aggregate movement attempts over last T timesteps:
    # Movement flags indices: [0 (N),1 (S),2 (E),3 (W)] categorical 0/1
    move_flags = o[:, :, :, 0:4].float()  # (batch, T, agents, 4)
    move_counts = move_flags.sum(dim=1) / T  # normalized counts over T, shape (batch, agents, 4)

    # 2) Aggregate Attack and No-op action counts over last T timesteps:
    # Last action indices: Attack=37, No-op=31
    attack_flags = o[:, :, :, 37].float()
    noop_flags = o[:, :, :, 31].float()
    attack_count = attack_flags.sum(dim=1) / T  # (batch, agents)
    noop_count = noop_flags.sum(dim=1) / T      # (batch, agents)
    actions_agg = th.stack([attack_count, noop_count], dim=-1)  # (batch, agents, 2)

    # 3) Most recent relative position to ally at last timestep:
    # Use relative X and Y position to ally1 at last timestep (t = T-1)
    # Indices: relative X ally1 = 18, relative Y ally1 = 19
    # If ally1 visible (index 16), use these relative positions; else zeros.
    last_obs = o[:, -1, :, :]  # (batch, agents, 42)
    ally1_visible = last_obs[:, :, 16:17]  # (batch, agents, 1)
    rel_x_ally1 = last_obs[:, :, 18:19]   # (batch, agents, 1)
    rel_y_ally1 = last_obs[:, :, 19:20]   # (batch, agents, 1)
    rel_pos_ally1 = th.cat([rel_x_ally1, rel_y_ally1], dim=-1) * ally1_visible  # zero if not visible

    # 4) Freshness indicator: normalized timestep index = 1.0 (last timestep)
    # Since last timestep is T-1, normalized by (T-1)
    freshness = th.full((batch_size, n_agents, 1), fill_value=(T - 1) / max(T - 1, 1), device=device, dtype=dtype)

    # Concatenate all message parts: (batch, agents, 4+2+2+1=9)
    message = th.cat([move_counts, actions_agg, rel_pos_ally1, freshness], dim=-1)

    # Broadcast messages: each agent receives messages from other two agents (excluding self)

    # Expand dims for broadcasting
    message_exp = message.unsqueeze(2)  # (batch, sender_agent, 1, 12)
    message_exp = message_exp.repeat(1, 1, n_agents, 1)  # (batch, sender_agent, receiver_agent, 12)

    # Create mask to zero out self messages (sender == receiver)
    sender_ids = th.arange(n_agents, device=device).view(1, n_agents, 1)  # (1, sender_agent, 1)
    receiver_ids = th.arange(n_agents, device=device).view(1, 1, n_agents)  # (1, 1, receiver_agent)
    self_mask = (sender_ids == receiver_ids)  # (1, sender_agent, receiver_agent)
    self_mask = self_mask.expand(batch_size, -1, -1)  # (batch, sender_agent, receiver_agent)
    message_exp = message_exp.masked_fill(self_mask.unsqueeze(-1), 0.0)

    # Permute to (batch, receiver_agent, sender_agent, message_dim)
    message_exp = message_exp.permute(0, 2, 1, 3)  # (batch, receiver_agent, sender_agent, 12)

    # Prepare indices of senders per receiver excluding self (hardcoded for 3 agents)
    sender_indices = th.tensor([[1, 2],
                               [0, 2],
                               [0, 1]], device=device)  # (3, 2)
    sender_indices = sender_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, receiver_agent=3, 2)

    # Gather messages from other agents
    message_dim = message.shape[-1]
    messages_from_others = th.gather(message_exp, 2, sender_indices.unsqueeze(-1).expand(-1, -1, -1, message_dim))
    # (batch, receiver_agent=3, 2, 12)

    # Concatenate messages from two other agents along last dimension
    messages_concat = messages_from_others.reshape(batch_size, n_agents, -1)  # (batch, 3, 24)

    # Append new messages to last timestep observations only (shape: batch, agents, 42)
    # As input is (batch, T, agents, 42), return enhanced obs for last timestep with appended messages
    last_obs_enhanced = th.cat([last_obs, messages_concat], dim=-1)  # (batch, agents, 42 + 24)

    return last_obs_enhanced
