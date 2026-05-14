import torch as th

def message_design_instruction():
    """
    Improved Message Design Instruction for 5z_vs_1ul (SMAC):
    
    1. **Purpose**:
       Address the previous protocol's weaknesses by enabling agents to infer team configuration and intent history.

    2. **Content Selection & Justification**:
       Each agent broadcasts a message constructed from the **past 9 steps** (t-9 to t-1):

       [A] **Own Absolute Position History (past 9 steps):**
           - Normalized absolute X and Y positions (2 × 9 = 18 dims).

       [B] **Own Health & Shield Ratio History (past 9 steps):**
           - Health ratio (1 × 9 = 9 dims)
           - Shield ratio (1 × 9 = 9 dims)

       [C] **Own Weapon Cooldown History (past 9 steps):**
           - Weapon cooldown status (1 × 9 = 9 dims)
           
       **Total message_dim = 18 + 9 + 9 + 9 = 45**
       (Current step info, Ultralisk visibility, and Sender ID are omitted as they are in `comm_update` or `comm_init`)

    3. **Communication Protocol:**
       - **Broadcast**: Each agent sends its message to all others.
       - Each agent receives the 4 messages from teammates (not itself), for a total incoming message size of 4 × 45 = 180.
       - Each received message is a concatenation of history fields.
       - On input, the final tensor is (batch, 5, 48+180).

    4. **Efficiency:**
       - All operations are batch- and vectorized.
    """
    return (
        "Each agent's message consists of its **past 9 steps** history of: "
        "1) Normalized absolute X and Y position (18 dims), "
        "2) Health ratio (9 dims), "
        "3) Shield ratio (9 dims), "
        "4) Weapon cooldown status (9 dims). "
        "Total added message_dim = 45*4 = 180. "
        "Current step info and Sender ID are omitted to avoid redundancy with other comm modules."
    )

def communication(o):
    """
    Input:
        o: torch.Tensor, shape (batch, T, 5, 48), agent observations for last 10 steps (T >= 10).
    Output:
        messages_o: torch.Tensor, shape (batch, 5, 48+180), concatenated current observation and received messages.
    """
    # Device and dtype preservation
    device = o.device
    dtype = o.dtype
    batch_size, T, n_agents, obs_dim = o.shape
    assert n_agents == 5 and obs_dim == 48
    assert T >= 10, "Need at least 10 steps to extract temporal context"

    # Use last 10 steps (including current)
    o_last10 = o[:, -10:, :, :]  # (batch, 10, 5, 48)

    # 1. Absolute X/Y position: 
    # If your system stores absolute positions elsewhere, replace the following lines accordingly.
    # For demonstration, let's assume dims 0 and 1 of each agent's own observation encode normalized absolute X/Y.
    # If not, these fields should be provided in the environment preprocessing.
    abs_x = o_last10[..., 0]  # (batch, 10, 5)
    abs_y = o_last10[..., 1]  # (batch, 10, 5)

    # 2. Health ratio (dim 34), 3. Shield ratio (dim 35), 4. Weapon cooldown (dim 36)
    health = o_last10[..., 34]  # (batch, 10, 5)
    shield = o_last10[..., 35]  # (batch, 10, 5)
    # Weapon cooldown: if not available, set to zeros
    if obs_dim > 36:
        cooldown = o_last10[..., 36]  # (batch, 10, 5)
    else:
        cooldown = th.zeros_like(health)

    # Reshape histories: (batch, 5, 10)
    abs_x = abs_x.permute(0, 2, 1)  # (batch, 5, 10)
    abs_y = abs_y.permute(0, 2, 1)
    health = health.permute(0, 2, 1)
    shield = shield.permute(0, 2, 1)
    cooldown = cooldown.permute(0, 2, 1)

    # Use only PAST 9 steps (exclude current step at index 9)
    # (batch, 5, 9)
    abs_x = abs_x[..., :-1]
    abs_y = abs_y[..., :-1]
    health = health[..., :-1]
    shield = shield[..., :-1]
    cooldown = cooldown[..., :-1]

    # Stack history vectors per agent: (batch, 5, 18+9+9+9) = (batch, 5, 45)
    hist_vec = th.cat([
        abs_x,        # 9
        abs_y,        # 9
        health,       # 9
        shield,       # 9
        cooldown      # 9
    ], dim=-1)  # (batch, 5, 45)

    msg = hist_vec  # (batch, 5, 45)

    # For each agent, collect messages from other 4 agents (not itself)
    # Create a mask to exclude self-messages
    mask = (1 - th.eye(n_agents, device=device, dtype=dtype)).unsqueeze(0).unsqueeze(-1)  # (1, 5, 5, 1)

    msg_exp = msg.unsqueeze(1).expand(batch_size, n_agents, n_agents, 45)  # (batch, 5, 5, 45)
    msg_masked = msg_exp * mask  # self-message is zeroed

    # For each agent, get indices of other agents
    indices = []
    for i in range(n_agents):
        indices.append([j for j in range(n_agents) if j != i])
    indices = th.tensor(indices, device=device)  # (5, 4)

    indices_expand = indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 5, 4)
    indices_expand = indices_expand.unsqueeze(-1).expand(-1, -1, -1, 45)  # (batch, 5, 4, 45)
    msgs_from_others = th.gather(msg_masked, 2, indices_expand)  # (batch, 5, 4, 45)

    # Reshape to (batch, 5, 180)
    msgs_from_others = msgs_from_others.reshape(batch_size, n_agents, 4 * 45)

    # Get the current observation for each agent: o[:, -1, :, :] -> (batch, 5, 48)
    o_cur = o[:, -1, :, :]

    # Concatenate with current observation
    messages_o = th.cat([o_cur, msgs_from_others], dim=-1)  # (batch, 5, 48+180)

    return messages_o
