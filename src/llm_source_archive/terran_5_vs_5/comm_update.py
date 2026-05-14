import torch as th

def message_design_instruction():
    """
    Task-specific communication protocol for SMACv2 terran_5_vs_5 scenario (Improved):

    Objective:
    Enable each agent to share only essential, non-redundant, and non-locally observable information
    about its own state, its local observations about allies and enemies, and its current visibility/certainty sets.
    This empowers agents to resolve ambiguities under partial observability, synchronize global state understanding,
    and coordinate tactics according to the randomized team composition and layout.

    Message Content Construction:
    Each agent constructs a message containing:
    1. Own state (Last Action only):
        - Last action (11-dim one-hot; o[...,82:93])
    2. Visibility/certainty vectors:
        - Ally visibility mask: 5-dim binary vector (which allies are visible to the sender, including self)
        - Enemy visibility mask: 5-dim binary vector (which enemies are visible/shootable to the sender)
    3. Recent local observations about each ally (excluding self), with validity bit:
        For each of the 4 other allies:
            - health, rel_x, rel_y, unit type (3-dim), visible (1-dim validity)
        (7 dims per ally × 4 = 28 dims)
    4. Recent local observations about each enemy, with validity bit:
        For each of the 5 enemies:
            - health, rel_x, rel_y, unit type (3-dim), shootable (1-dim validity)
        (7 dims per enemy × 5 = 35 dims)

    Each message: 11 (own last action) + 5 (ally vis) + 5 (enemy vis) + 28 (allies) + 35 (enemies) = 84 dims

    Message Distribution:
    - Broadcast: Each message is sent to all teammates (not self).
    - Each agent receives the concatenated messages from all other agents (4 × 84 = 336 dims).
    - The enhanced observation for each agent: [original 98-dim obs | messages from other 4 agents (336 dims)]
      = 434 dims.

    Protocol ensures:
    - Each agent has explicit meta-information about what each teammate knows and sees at the current timestep.
    - All agents can resolve which data is direct, inferred, or missing, and can synchronize global state beliefs.
    - The protocol is vectorized for batch efficiency and avoids redundant or easily inferable information.

    """
    return (
        "Each agent constructs a message containing: "
        "1) own last action (11); "
        "2) visibility masks for all allies (5) and all enemies (5); "
        "3) most recent local observations about each other ally (health, rel_x, rel_y, unit type (3), visible/validity), for 4 allies (28); "
        "4) most recent local observations about each enemy (health, rel_x, rel_y, unit type (3), shootable/validity), for 5 enemies (35). "
        "Each message is broadcast to all teammates (not self). "
        "Each agent's enhanced observation is the concatenation of its original 98-dim obs and the 4 messages received from teammates "
        "(total 98+336=434 dims). "
        "Redundant fields (Sender ID, Own Health/Pos/Type) removed."
    )

def communication(o):
    """
    Input:
        o: Tensor of shape (batch, num_agents=5, obs_dim=98)
    Output:
        messages_o: Tensor of shape (batch, num_agents=5, 98 + 336)
    """
    B, N, D = o.shape  # (batch, 5, 98)
    device = o.device


    # --- Own state (Retained: Last Action only) ---
    own_last_action = o[..., 82:93]         # (B, N, 11)

    # --- Ally Visibility Mask (include self=1) ---
    ally_visible_mask = th.zeros(B, N, N, device=device)
    # Self always visible
    idx = th.arange(N)
    ally_visible_mask[:, idx, idx] = 1
    # Indices for ally_j_visible in obs: [44,52,60,68]
    ally_visible_indices = [44, 52, 60, 68]
    for i in range(N):
        others = [j for j in range(N) if j != i]
        for k, j in enumerate(others):
            ally_visible_mask[:, i, j] = o[:, i, ally_visible_indices[k]]
    # (B, N, 5)

    # --- Enemy Visibility Mask (5-dim, shootable as proxy) ---
    enemy_shootable_idx = [4, 12, 20, 28, 36]
    enemy_visible_mask = o[:, :, enemy_shootable_idx]  # (B, N, 5)

    # --- Recent observations about each ally (excluding self), with validity ---
    ally_health_idx = [48, 56, 64, 72]
    ally_relx_idx = [46, 54, 62, 70]
    ally_rely_idx = [47, 55, 63, 71]
    ally_utype_idx = [
        [49, 50, 51], [57, 58, 59], [65, 66, 67], [73, 74, 75]
    ]
    # For each of the 4 allies (not self), for each agent
    ally_fields = []
    for k in range(4):
        f = th.cat([
            o[:, :, ally_health_idx[k]].unsqueeze(-1),  # (B, N, 1)
            o[:, :, ally_relx_idx[k]].unsqueeze(-1),    # (B, N, 1)
            o[:, :, ally_rely_idx[k]].unsqueeze(-1),    # (B, N, 1)
            o[:, :, ally_utype_idx[k]],                 # (B, N, 3)
            o[:, :, ally_visible_indices[k]].unsqueeze(-1)  # (B, N, 1) validity
        ], dim=-1)  # (B, N, 7)
        ally_fields.append(f)
    ally_fields = th.stack(ally_fields, dim=2)  # (B, N, 4, 7)
    ally_fields_flat = ally_fields.reshape(B, N, 28)  # (B, N, 28)

    # --- Recent observations about each enemy, with validity ---
    enemy_health_idx = [8, 16, 24, 32, 40]
    enemy_relx_idx = [6, 14, 22, 30, 38]
    enemy_rely_idx = [7, 15, 23, 31, 39]
    enemy_utype_idx = [
        [9, 10, 11], [17, 18, 19], [25, 26, 27], [33, 34, 35], [41, 42, 43]
    ]
    enemy_fields = []
    for k in range(5):
        f = th.cat([
            o[:, :, enemy_health_idx[k]].unsqueeze(-1),  # (B, N, 1)
            o[:, :, enemy_relx_idx[k]].unsqueeze(-1),    # (B, N, 1)
            o[:, :, enemy_rely_idx[k]].unsqueeze(-1),    # (B, N, 1)
            o[:, :, enemy_utype_idx[k]],                 # (B, N, 3)
            o[:, :, enemy_shootable_idx[k]].unsqueeze(-1)  # (B, N, 1) validity
        ], dim=-1)  # (B, N, 7)
        enemy_fields.append(f)
    enemy_fields = th.stack(enemy_fields, dim=2)  # (B, N, 5, 7)
    enemy_fields_flat = enemy_fields.reshape(B, N, 35)  # (B, N, 35)

    # --- Compose message: [own_last_action(11), ally_vis(5), enemy_vis(5), allies(28), enemies(35)] = 84 ---
    msg_parts = [
        own_last_action,          # (B, N, 11)
        ally_visible_mask,        # (B, N, 5)
        enemy_visible_mask,       # (B, N, 5)
        ally_fields_flat,         # (B, N, 28)
        enemy_fields_flat         # (B, N, 35)
    ]
    message = th.cat(msg_parts, dim=-1)  # (B, N, 84)

    # --- Distribute messages: for each agent, collect messages from all other agents (not self) ---
    # message: (B, N, 84)
    message_exp = message.unsqueeze(1).expand(B, N, N, 84)  # (B, receiver, sender, 84)
    mask = ~th.eye(N, dtype=th.bool, device=device).unsqueeze(0).expand(B, N, N)  # (B, N, N)
    msg_others = message_exp[mask].view(B, N, N-1, 84)      # (B, N, 4, 84)
    msg_others_flat = msg_others.reshape(B, N, 4*84)        # (B, N, 336)

    # --- Concatenate to original observation ---
    messages_o = th.cat([o, msg_others_flat], dim=-1)    # (B, N, 98+336=434)
    return messages_o
