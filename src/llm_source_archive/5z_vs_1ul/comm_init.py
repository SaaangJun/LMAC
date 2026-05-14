import torch as th

def message_design_instruction():
    """
    Message Design Instruction for 5z_vs_1ul (SMAC):

    1. **Purpose**: 
       The message is designed to coordinate Zealots for optimal kiting, focus fire, and loss minimization against the Ultralisk under partial observability.

    2. **Content Selection & Justification**:
       Each agent broadcasts a compact message containing:
         - [A] **Own Position**: Relative X, Y position of the Ultralisk (dims 5,6 if visible; else zeros). 
           - *Rationale*: If an agent sees the Ultralisk, sharing its own measurement of the Ultralisk's position helps teammates infer or correct the global enemy location, especially when they cannot see the enemy themselves.
         - [B] **Own Distance to Ultralisk**: Normalized distance (dim 7 if visible; else zero).
           - *Rationale*: Reveals whether the agent is at risk, supports coordinated retreat/attack decisions.
         - [C] **Own Health Ratio**: (dim 34).
           - *Rationale*: Critical for allies to know which units are low and need to kite or be protected.
         - [D] **Own Last Action**: One-hot vector (dims 36-42).
           - *Rationale*: Allows allies to infer intentions and synchronize kiting or attack patterns.
         - [E] **Sender Identity**: One-hot (dims 43-47).
           - *Rationale*: Receiver can distinguish which teammate sent the message.

       All fields are directly copied from the sender's own observation, except that enemy-relative fields are set to zero if the enemy is not visible to the sender.

    3. **Communication Protocol**:
       - **Broadcast**: Each agent sends its message to all others (no peer-to-peer customization).
       - Each agent receives the messages from the other 4 agents (not itself), for a total incoming message size of 4 × message_dim.
       - Each received message is a concatenation of [A,B,C,D,E] from its corresponding teammate.
       - On input, the final tensor is (batch, 5, 48+4*message_dim).

    4. **Message Structure**:
       - [A] Relative X to Ultralisk (1 float)
       - [A] Relative Y to Ultralisk (1 float)
       - [B] Normalized distance to Ultralisk (1 float)
       - [C] Own health ratio (1 float)
       - [D] Last action one-hot (7 floats)
       - [E] Sender one-hot (5 floats)
       - **Total message_dim = 1+1+1+1+7+5 = 16**

    5. **Why these specific fields?**
       - Enemy position/distance is the most crucial, but only rarely visible; thus, sharing sightings is essential.
       - Health and last action allow for dynamic, context-aware team strategies (e.g., synchronizing focus fire, kiting, or protecting low-health agents).
       - Sender identity is needed for explicitness and for agents to distinguish which teammate's state is being referenced.

    6. **Efficiency**:
       - All operations are batch- and vectorized; no explicit for-loops over batch or agent dimension.
       - No trainable parameters.
    """
    return (
        "Message: Each agent broadcasts a vector consisting of: "
        "[Relative X to Ultralisk, Relative Y to Ultralisk, Normalized distance to Ultralisk] "
        "(all zero if Ultralisk not visible), own health ratio, last action (one-hot, 7 dims), "
        "and sender identity (one-hot, 5 dims). Each agent receives the 4 messages from teammates "
        "(not including itself), concatenating them to its own observation. "
        "Total added message_dim = 16*4 = 64. "
        "This protocol maximizes coordination under partial observability by sharing unique, actionable, "
        "and non-redundant information directly related to task-critical coordination."
    )

def communication(o):
    """
    Input:
        o: torch.Tensor, shape (batch, 5, 48), original agent observations.

    Output:
        messages_o: torch.Tensor, shape (batch, 5, 48+64), concatenated observation and received messages.
    """
    # Device and dtype preservation
    device = o.device
    dtype = o.dtype
    batch_size = o.shape[0]
    n_agents = o.shape[1]
    obs_dim = o.shape[2]
    assert n_agents == 5 and obs_dim == 48

    # Build each agent's message: (batch, 5, 16)
    # [RelX_ultra, RelY_ultra, NormDist_ultra] (dims 5,6,7 if visible, else 0)
    # [Own health] (dim 34)
    # [Last action one-hot] (dims 36-42)
    # [Sender one-hot] (dims 43-47)

    # (A) Ultralisk visibility: (batch, 5, 1)
    ultra_visible = o[..., 4:5]  # (batch, 5, 1), 1 if visible, 0 else

    # (A) Relative X, Y, NormDist to Ultralisk: (batch, 5, 3)
    rel_x = o[..., 5:6] * ultra_visible  # zero if not visible
    rel_y = o[..., 6:7] * ultra_visible
    norm_dist = o[..., 7:8] * ultra_visible

    # (C) Own health ratio: (batch, 5, 1)
    own_health = o[..., 34:35]

    # (D) Last action one-hot: (batch, 5, 7)
    last_action = o[..., 36:43]

    # (E) Sender identity: (batch, 5, 5)
    sender_id = o[..., 43:48]

    # Stack all message fields: (batch, 5, 16)
    msg_fields = [rel_x, rel_y, norm_dist, own_health, last_action, sender_id]
    msg = th.cat(msg_fields, dim=-1)  # (batch, 5, 16)

    # Now, for each agent, gather messages from the other 4 agents (not itself)
    # We'll do this by masking out self-message and stacking the others

    # For each agent, generate a mask to exclude self
    # (5, 5) mask: mask[i, j] = 1 if i != j else 0
    mask = (1 - th.eye(n_agents, device=device, dtype=dtype)).unsqueeze(0).unsqueeze(-1)  # (1, 5, 5, 1)

    # Expand msg for pairwise selection: (batch, 5, 5, 16)
    msg_exp = msg.unsqueeze(1).expand(batch_size, n_agents, n_agents, 16)

    # Mask out self-messages
    msg_masked = msg_exp * mask  # self-message is zeroed

    # For each receiver (agent axis=1), extract the messages from other agents (axis=2)
    # Now, for each agent, we want to collect the 4 messages from others (not itself).
    # To do this, we can, for each agent, select the indices [0,1,2,3,4] except the agent's own index.

    # Prepare indices: for agent i, get [0,...,i-1, i+1,...,4]
    indices = []
    for i in range(n_agents):
        indices.append([j for j in range(n_agents) if j != i])
    indices = th.tensor(indices, device=device)  # (5, 4)

    # Gather messages: for each agent, get messages from other 4 agents
    # msg_masked: (batch, receiver_agent, sender_agent, 16)
    # We want for each batch, each agent, the 4 messages from indices[agent]
    # Use gather along sender_agent axis (dim=2)
    indices_expand = indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 5, 4)
    indices_expand = indices_expand.unsqueeze(-1).expand(-1, -1, -1, 16)  # (batch, 5, 4, 16)
    msgs_from_others = th.gather(msg_masked, 2, indices_expand)  # (batch, 5, 4, 16)

    # Reshape to (batch, 5, 64)
    msgs_from_others = msgs_from_others.reshape(batch_size, n_agents, 4 * 16)

    # Concatenate with original observation
    messages_o = th.cat([o, msgs_from_others], dim=-1)  # (batch, 5, 48+64)

    return messages_o
