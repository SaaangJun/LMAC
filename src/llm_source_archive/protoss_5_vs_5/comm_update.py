import torch as th

def message_design_instruction():
    """
    Enhanced Message Design Instruction for protoss_5_vs_5 (SMACv2):

    1. **Purpose**:
        - Addresses feedback by adding explicit **Visibility Flags**, **Timestamps**, and **Explicit Missing Values**.

    2. **What to Communicate (per message, from sender to receivers)**:
        For each sender agent, the message contains:

        **A. ENEMY ENTITY BLOCKS (5 enemies):**
            For each enemy (indexed 0..4, fixed order):
                - [visible_flag (1), health (1), shield (1), abs_x (1), abs_y (1), timestamp (1)]
                - If visible: flag=1, timestamp=1, attributes=values.
                - If NOT visible: flag=0, timestamp=0, attributes=-1.

        **B. ALLY ENTITY BLOCKS (4 allies, excluding self):**
            For each ally (indexed 0..3, fixed order, excluding the sender itself):
                - [visible_flag (1), health (1), shield (1), abs_x (1), abs_y (1), timestamp (1)]
                - If visible: flag=1, timestamp=1, attributes=values.
                - If NOT visible: flag=0, timestamp=0, attributes=-1.

        **C. SENDER IDENTITY:**
             - Include One-hot vector (length 5) for robustness and standalone context.

    3. **Message Size Calculation**:
        - Enemy block: 5 × 6 = 30
        - Ally block: 4 × 6 = 24
        - Sender one-hot: 5
        - Message size per agent: **59**
        - Each agent receives 4 such messages: 4 × 59 = **236**
        - Enhanced observation shape: (batch, 5, 108 + 236) = (batch, 5, 344)
    """
    return (
        "Each agent broadcasts an enhanced update message containing:\n"
        "- Enemy Blocks (5): [visible, health, shield, abs_x, abs_y, timestamp] (missing=-1)\n"
        "- Ally Blocks (4): [visible, health, shield, abs_x, abs_y, timestamp] (missing=-1)\n"
        "- Sender ID (5)\n"
        "Total dim: 59. Explicit visibility/missingness handling ensures robust inference."
    )

def communication(o):
    """
    Enhanced communication function for protoss_5_vs_5:
    o: Tensor of shape (batch, 5, 108)
    Returns: Tensor of shape (batch, 5, 344) = (batch, 5, 108 + 236)
    """
    device = o.device
    batch_size, n_agents, n_obs = o.shape
    n_enemies = 5
    n_allies = 5
    msg_enemy_fields = 6  # [visible_flag, health, shield, abs_x, abs_y, timestamp]
    msg_ally_fields = 6
    msg_sender_fields = n_agents
    msg_enemy_block = n_enemies * msg_enemy_fields
    msg_ally_block = (n_allies - 1) * msg_ally_fields

    # === ENEMY ENTITY BLOCKS ===
    enemy_offsets = th.arange(n_enemies, device=device) * 9
    enemy_health_idx = 8 + enemy_offsets
    enemy_shield_idx = 9 + enemy_offsets
    enemy_relx_idx = 6 + enemy_offsets
    enemy_rely_idx = 7 + enemy_offsets

    # Own absolute positions
    own_x = o[..., 87].unsqueeze(2)  # (batch, 5, 1)
    own_y = o[..., 88].unsqueeze(2)  # (batch, 5, 1)

    # Enemy relative info
    enemy_health = o[..., enemy_health_idx]  # (batch, 5, 5)
    enemy_shield = o[..., enemy_shield_idx]
    enemy_relx = o[..., enemy_relx_idx]
    enemy_rely = o[..., enemy_rely_idx]

    enemy_absx = own_x + enemy_relx  # (batch, 5, 5)
    enemy_absy = own_y + enemy_rely

    # Visibility: 'shootable' flag as main visibility cue
    shootable_idx = 4 + enemy_offsets
    enemy_visible = (o[..., shootable_idx] > 0.0).float()  # (batch, 5, 5)

    # Timestamp: 1 for current
    enemy_timestamp = enemy_visible.clone()

    # Explicit Missing Values (-1)
    missing_value = -th.ones_like(enemy_health)
    
    # Fill based on visibility
    enemy_health_filled = enemy_health * enemy_visible + missing_value * (1 - enemy_visible)
    enemy_shield_filled = enemy_shield * enemy_visible + missing_value * (1 - enemy_visible)
    enemy_absx_filled = enemy_absx * enemy_visible + missing_value * (1 - enemy_visible)
    enemy_absy_filled = enemy_absy * enemy_visible + missing_value * (1 - enemy_visible)

    # Stack fields
    enemy_block = th.stack([
        enemy_visible,                 # visible_flag
        enemy_health_filled,           # health
        enemy_shield_filled,           # shield
        enemy_absx_filled,             # abs_x
        enemy_absy_filled,             # abs_y
        enemy_timestamp                # timestamp
    ], dim=-1)

    # Reshape: (batch, 5, 30)
    enemy_block = enemy_block.reshape(batch_size, n_agents, n_enemies * msg_enemy_fields)

    # === ALLY ENTITY BLOCKS (excluding self) ===
    ally_offsets = th.arange(n_allies - 1, device=device) * 9
    ally_health_idx = 53 + ally_offsets
    ally_shield_idx = 54 + ally_offsets
    ally_relx_idx = 51 + ally_offsets
    ally_rely_idx = 52 + ally_offsets
    ally_visible_idx = 49 + ally_offsets

    ally_health = o[..., ally_health_idx]  # (batch, 5, 4)
    ally_shield = o[..., ally_shield_idx]
    ally_relx = o[..., ally_relx_idx]
    ally_rely = o[..., ally_rely_idx]
    ally_visible = o[..., ally_visible_idx]  # 1 if visible, 0 if not

    # Compute absolute positions for allies
    own_x4 = own_x[..., :4]  # (batch, 5, 4)
    own_y4 = own_y[..., :4]
    ally_absx = own_x4 + ally_relx
    ally_absy = own_y4 + ally_rely

    ally_timestamp = ally_visible.clone()

    missing_value_ally = -th.ones_like(ally_health)
    ally_health_filled = ally_health * ally_visible + missing_value_ally * (1 - ally_visible)
    ally_shield_filled = ally_shield * ally_visible + missing_value_ally * (1 - ally_visible)
    ally_absx_filled = ally_absx * ally_visible + missing_value_ally * (1 - ally_visible)
    ally_absy_filled = ally_absy * ally_visible + missing_value_ally * (1 - ally_visible)

    ally_block = th.stack([
        ally_visible,                 # visible_flag
        ally_health_filled,           # health
        ally_shield_filled,           # shield
        ally_absx_filled,             # abs_x
        ally_absy_filled,             # abs_y
        ally_timestamp                # timestamp
    ], dim=-1)

    # Reshape: (batch, 5, 24)
    ally_block = ally_block.reshape(batch_size, n_agents, (n_allies - 1) * msg_ally_fields)

    # === SENDER IDENTITY ===
    sender_id = th.eye(n_agents, device=device)[None, :, :].expand(batch_size, n_agents, n_agents)  # (batch, 5, 5)

    # === MESSAGE CONCAT ===
    message = th.cat([enemy_block, ally_block, sender_id], dim=-1) # (batch, 5, 59)

    # === BROADCAST ===
    agent_indices = th.arange(n_agents, device=device)
    other_agent_indices = th.stack([
        th.cat([agent_indices[:i], agent_indices[i+1:]])
        for i in range(n_agents)
    ])  # (5, 4)

    received = th.stack(
        [message[:, other_agent_indices[i], :] for i in range(n_agents)],
        dim=1
    )  # (batch, 5, 4, 59)
    received = received.reshape(batch_size, n_agents, -1)  # (batch, 5, 236)

    # === CONCATENATE TO OBSERVATION ===
    messages_o = th.cat([o, received], dim=-1)  # (batch, 5, 344)
    return messages_o
