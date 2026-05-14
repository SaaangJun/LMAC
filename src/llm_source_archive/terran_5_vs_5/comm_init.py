import torch as th

def message_design_instruction():
    """
    Task-specific communication protocol for SMACv2 terran_5_vs_5 scenario:

    **Objective**: 
    Enable each agent to share only the essential, non-redundant, and non-locally observable information about itself 
    (health, cooldown, absolute position, and unit type) with all teammates. This allows all agents to reconstruct 
    the full allied team state and reason about strategies (focus fire, support, retreat, etc.) under partial observability.

    **Message Content Construction**:
    - Each agent sends a message containing:
        - **Sender identity**: 5-dim one-hot vector indicating which agent sent the message.
        - **Own health**: Scalar (from o[..., 76])
        - **Own cooldown**: Not directly observable in the provided obs, so omitted unless present. (If present, would be included.)
        - **Own absolute position**: (o[..., 80], o[..., 81])
        - **Own unit type**: 3-dim one-hot (o[..., 77:80])
    - **Why?**: 
        - Each agent’s own health and position are always locally known, but not always visible to others due to partial observability.
        - By broadcasting these, all agents have up-to-date, full-team status for tactical coordination.
        - Sender identity allows recipients to map the info to the correct teammate.
        - No redundant info is sent: Each agent only sends its own info, and all content is necessary for team-level coordination.
    - **Message Distribution**:
        - **Broadcast**: Each message is sent to all other agents (not self).
        - For each agent, the incoming message block consists of the concatenated messages from the 4 other agents.
        - The enhanced observation for agent i is: [original obs | message from agent 0 | ... | message from agent 4], excluding message from itself.
    - **Message Format**:
        - Each message: [sender_id (5), health (1), abs_x (1), abs_y (1), unit_type (3)] = 11 dimensions.
        - Each agent receives 4 such messages: 44 dims.
        - Enhanced obs shape: (32, 5, 98+44)
    - **Efficiency**:
        - All message construction and distribution is vectorized for batch efficiency.

    **Summary**:
    This protocol ensures each agent has non-local, up-to-date, and sufficient information about all teammates' critical state dimensions for optimal coordination, while minimizing redundancy and message size.
    """
    return (
        "Each agent constructs a message containing: "
        "1) sender identity (one-hot, 5-dim), "
        "2) own health (scalar), "
        "3) own absolute x position (scalar), "
        "4) own absolute y position (scalar), "
        "5) own unit type (one-hot, 3-dim). "
        "Each message is broadcast to all teammates (not self). "
        "Each agent's enhanced observation is the concatenation of its original 98-dim obs and the 4 messages received from teammates "
        "(total 98+44=142 dims). "
        "This protocol enables all agents to reconstruct the full allied team state under partial observability, "
        "improving coordination while minimizing redundancy."
    )

def communication(o):
    """
    Input:
        o: Tensor of shape (batch, num_agents, obs_dim=98)
    Output:
        messages_o: Tensor of shape (batch, num_agents, 98 + 44)
    """
    # Shapes
    B, N, D = o.shape
    device = o.device
    # Message components
    # Sender identity: one-hot (N,)
    sender_ids = th.eye(N, device=device).unsqueeze(0).expand(B, N, N)  # (B, N, N), [batch, sender, id]
    # Own health: o[..., 76]
    own_health = o[..., 76].unsqueeze(-1)  # (B, N, 1)
    # Own absolute position: o[..., 80], o[..., 81]
    own_abs_x = o[..., 80].unsqueeze(-1)   # (B, N, 1)
    own_abs_y = o[..., 81].unsqueeze(-1)   # (B, N, 1)
    # Own unit type: o[..., 77:80]
    own_unit_type = o[..., 77:80]          # (B, N, 3)
    # Assemble message: [sender_id (N), health (1), abs_x (1), abs_y (1), unit_type (3)]
    # For each agent, we want to send out a message vector of dim (N + 1 + 1 + 1 + 3) = N+6 = 11 (for N=5)
    # But for compactness, sender_id is always 5-dim, so message = (5, 1, 1, 1, 3) = 11 dims
    message = th.cat([sender_ids, own_health, own_abs_x, own_abs_y, own_unit_type], dim=-1)  # (B, N, 11)
    # For each agent, collect messages from all other agents (not self)
    # We want for agent i: [message_0, ..., message_{i-1}, message_{i+1}, ..., message_{N-1}] (total N-1 messages)
    # So, for batch efficiency:
    # Expand message to shape (B, N, N, 11): message from sender j to receiver i is message[:, j, :]
    message_exp = message.unsqueeze(1).expand(B, N, N, 11)  # (B, receiver, sender, 11)
    # Mask out self-message (don't send to self)
    mask = ~th.eye(N, dtype=th.bool, device=device).unsqueeze(0).expand(B, N, N)  # (B, N, N)
    # For each agent (receiver), select messages from all other agents (sender != receiver)
    # message_exp[mask]: (B*N*(N-1), 11), reshape to (B, N, N-1, 11)
    msg_others = message_exp[mask].view(B, N, N-1, 11)  # (B, N, N-1, 11)
    # Flatten incoming messages for each agent
    msg_others_flat = msg_others.reshape(B, N, (N-1)*11)  # (B, N, 44)
    # Concatenate to original observation
    messages_o = th.cat([o, msg_others_flat], dim=-1)  # (B, N, 98+44)
    return messages_o
