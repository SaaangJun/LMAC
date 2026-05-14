import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Task Context:
    Three attackers (agents) confront a defender and goalkeeper near the box.
    The central attacker holds the ball initially and must decide to dribble or pass.
    Coordination between attackers is crucial to create scoring opportunities under defensive pressure.

    Communication Protocol:
    - Broadcast communication: each agent sends a message to all other agents.
    - Messages include a sender identity (one-hot vector of length 3).
    - Each agent receives messages from other agents (excluding own message).
    - Messages are concatenated to the original observations along the last dimension,
      resulting in shape (32, 3, 48 + message_dim).

    Message Content Rationale:
    1. Uniqueness & Sufficiency:
       - Each agent communicates information about its own state and perception that others cannot directly observe.
       - Avoid sending info already known or easily inferred by others (e.g., ego agent's own absolute position is known to self).
       - Share self-position, direction, recent key actions, and relative positions of key entities from own perspective.
    
    2. Essential Information to Share:
       a) Ego Agent absolute position (2 dims: X,Y) - crucial for spatial coordination, since others only get relative positions.
       b) Ego Agent direction vector (2 dims: X,Y) - shows movement intent, aiding prediction of future positions.
       c) Ball possession & ball-relative info:
          - Ball relative position (2 dims: X,Y) from ego's perspective.
          - Ball direction (2 dims: X,Y).
       d) Last action summary (categorical one-hot compressed):
          - Instead of all 19 last action one-hot dims, select only actions relevant for coordination:
            * Dribble (o[43])
            * Pass (sum of LongPass[35], HighPass[36], ShortPass[37])
            * Shot (38)
            * Sprint (39)
          - This 4-dim last action summary captures key tactical decisions.
       e) Relative positions of visible allies (4 dims: Ally1 and Ally2 relative X,Y) - helps others understand agent's perception of teammates.
       f) Relative positions of visible enemies (4 dims: goalkeeper & center-back relative X,Y) - critical defensive info from ego's perspective.
    
    3. Sender Identity (3 dims one-hot):
       - Each agent appends a one-hot vector indicating its own id (0,1,2).
       - Enables receivers to identify message origin explicitly.

    Message Dimension:
    - Ego absolute pos (2)
    - Ego direction (2)
    - Ball relative pos (2)
    - Ball direction (2)
    - Allies relative pos (4)
    - Enemies relative pos (4)
    - Last action summary (4)
    - Sender one-hot (3)
    = 23 dims total per message.

    Summary:
    Each agent broadcasts a 23-dim message containing uniquely valuable personal observations and recent behavior cues.
    Receiving agents integrate these messages, enabling better inference of teammates' positions, intentions, and perceptions,
    improving coordinated passing, dribbling, and positioning under partial observability.

    """
    return message_design_instruction.__doc__


def communication(o: th.Tensor) -> th.Tensor:
    """
    Args:
        o: Observation tensor with shape (batch=32, agents=3, obs_dim=48)

    Returns:
        Tensor of shape (32, 3, 48 + 23) with appended messages from other agents.
    """
    device = o.device
    batch_size, num_agents, obs_dim = o.shape
    assert num_agents == 3 and obs_dim == 48, "Unexpected input shape"

    # Extract relevant dims for message construction per agent (batch-wise, agent-wise).

    # 1) Ego absolute position (2)
    ego_abs_pos = o[:, :, 0:2]  # (32,3,2)

    # 2) Ego direction (2)
    ego_dir = o[:, :, 6:8]  # (32,3,2)

    # 3) Ball relative position (2)
    ball_rel_pos = o[:, :, 20:22]  # (32,3,2)

    # 4) Ball direction (2)
    ball_dir = o[:, :, 23:25]  # (32,3,2)

    # 5) Allies relative positions: Ally1 (2 dims), Ally2 (2 dims) => total 4 dims
    allies_rel_pos = o[:, :, 2:6]  # (32,3,4)

    # 6) Enemies relative positions: Goalkeeper (2 dims), Center-back (2 dims) => total 4 dims
    enemies_rel_pos = o[:, :, 12:16]  # (32,3,4)

    # 7) Last action summary (4 dims):
    # Dribble (43)
    dribble = o[:, :, 43:44]  # (32,3,1)
    # Pass = sum of LongPass(35), HighPass(36), ShortPass(37)
    pass_actions = o[:, :, 35:38].sum(dim=2, keepdim=True).clamp(0,1)  # (32,3,1), clamp to binary 0/1
    # Shot (38)
    shot = o[:, :, 38:39]  # (32,3,1)
    # Sprint (39)
    sprint = o[:, :, 39:40]  # (32,3,1)

    last_action_summary = th.cat([dribble, pass_actions, shot, sprint], dim=2)  # (32,3,4)

    # 8) Sender identity one-hot (3 dims)
    # Create a tensor of shape (3,3) as identity matrix (one-hot)
    identity_mat = th.eye(num_agents, device=device)  # (3,3)
    # Expand to (1,3,3) then repeat batch times
    sender_id = identity_mat.unsqueeze(0).repeat(batch_size, 1, 1)  # (32,3,3)

    # Concatenate all parts to form messages per agent
    # message shape: (32,3,23)
    messages = th.cat([
        ego_abs_pos,        # 2
        ego_dir,            # 2
        ball_rel_pos,       # 2
        ball_dir,           # 2
        allies_rel_pos,      # 4
        enemies_rel_pos,     # 4
        last_action_summary, # 4
        sender_id           # 3
    ], dim=2)

    # Prepare the message to be received by each agent:
    # Each agent should receive messages from the other two agents only.

    # For each agent, gather messages from other agents:
    # messages shape is (batch, 3, 23)
    # We want to create a received_message tensor (batch, 3, 2*23) by concatenating messages from other two agents.

    # Create mask to exclude self messages:
    # mask shape (3,3) with False on diagonal
    mask = (1 - th.eye(num_agents, device=device)).bool()  # (3,3)

    # Expand mask to (1, 3, 3) for batch broadcasting
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, agents, agents)

    # For each agent (dim=1), select messages from other agents (dim=2)
    # messages: (batch, sender_agent, 23)
    # We want for each receiver agent (dim=1) to pick messages from senders (dim=2) where mask is True.
    # This requires swapping dims or advanced indexing.

    # We can do this by:
    # 1) Expand messages to (batch, 1, sender_agent, 23)
    # 2) Expand mask to (batch, receiver_agent, sender_agent)
    # 3) Use mask to select messages for each receiver agent.

    messages_exp = messages.unsqueeze(1)  # (batch, 1, sender_agent, 23)
    mask_exp = mask.unsqueeze(-1)          # (batch, receiver_agent, sender_agent, 1)

    # Broadcast messages_exp along receiver_agent dim
    messages_exp = messages_exp.expand(-1, num_agents, -1, -1)  # (batch, receiver_agent, sender_agent, 23)

    # Masked messages: zero out messages from self (diagonal)
    masked_messages = messages_exp * mask_exp.float()  # (batch, receiver_agent, sender_agent, 23)

    # Now, for each receiver agent, the messages from other two agents are along dim=2.
    # We want to concatenate these two messages along feature dim (dim=3).
    # But some messages are zeroed out, so sum won't work.
    # Instead, gather messages from the two other agents.

    # Since each agent has exactly two True in mask row, we can:
    # 1) Get indices of other agents per agent (fixed for 3 agents):
    # agent 0 receives from 1 and 2
    # agent 1 receives from 0 and 2
    # agent 2 receives from 0 and 1

    # Hardcode indices:
    other_agents = th.tensor([[1,2],[0,2],[0,1]], device=device)  # (3,2)

    # Expand batch dimension for indexing
    batch_idx = th.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, num_agents, 2)  # (batch,3,2)
    agent_idx = th.arange(num_agents, device=device).unsqueeze(-1).expand(num_agents, 2)  # (3,2)
    agent_idx = agent_idx.unsqueeze(0).expand(batch_size, -1, -1)  # (batch,3,2)

    # Index into masked_messages: shape (batch, receiver_agent, sender_agent, 23)
    # Gather along sender_agent dim=2
    gathered_messages = masked_messages[batch_idx, agent_idx, other_agents.unsqueeze(0).expand(batch_size, -1, -1), :]  # (batch,3,2,23)

    # Concatenate messages from the two other agents along feature dim
    received_messages = gathered_messages.reshape(batch_size, num_agents, -1)  # (batch,3,46)

    # Final output: concatenate original obs with received messages
    messages_o = th.cat([o, received_messages], dim=2)  # (batch,3,48+46=94)

    return messages_o
