import torch as th

def message_design_instruction():
    """
    Message Design Instruction for SMACv2 zerg_5_vs_5 Communication Protocol

    Protocol Overview:
    ------------------
    Each agent constructs a task-specific message containing only the most critical state dimensions 
    that are likely to be partially observable to other agents and essential for coordination:
      - Its own absolute position (x, y), health, and unit type (one-hot).
      - Its *locally observed* information about enemies: for each enemy it can currently observe, 
        its estimate of enemy absolute position (x, y), health, and unit type (one-hot).
    
    The protocol is **broadcast**: each agent shares its message with all teammates. 
    Each message includes a sender identity (one-hot vector) for proper source attribution.

    Message Structure:
    ------------------
    - Sender ID: One-hot vector (length 5)
    - Own info: [own_health, own_pos_x, own_pos_y, own_unit_type_bit_0, own_unit_type_bit_1, own_unit_type_bit_2]
    - For each observed enemy (max 5):
        - [enemy_i_health, enemy_i_absolute_x, enemy_i_absolute_y, enemy_i_unit_type_bit_0, enemy_i_unit_type_bit_1, enemy_i_unit_type_bit_2]
        - If unobservable, all zeros

    This structure ensures:
      - **Uniqueness**: Each agent shares only what it uniquely observes (e.g., unseen enemies by others).
      - **Sufficiency**: All critical combat/tactical dimensions are included.
      - **Compactness**: No redundant ally info (since each agent knows its own state and receives others' self-reports).
      - **Explicitness**: All fields are interpretable and directly related to task objectives.
      - **Computational Efficiency**: The protocol is vectorized across batch and agents.

    The enhanced observation tensor is: (batch, n_agents, obs_dim + n_agents * message_dim)
    where message_dim = 5 (sender) + 6 (own) + 5*6 (enemies) = 41

    Usage:
    ------
    The protocol enables agents to infer full friendly team composition/roles, and collectively build 
    a more complete and up-to-date estimate of enemy positions, types, and health, even under partial observability.
    This improves focus fire, healing/cover, and role-adaptive tactics.
    """
    return (
        "Each agent constructs a broadcast message containing: "
        "(1) a one-hot sender ID (length 5); "
        "(2) its own absolute health, position (x, y), and unit type (3 bits); "
        "(3) for each enemy (5 total), if observable: [enemy health, absolute x, absolute y, unit type (3 bits)], else zeros. "
        "All agents share their messages with every other agent. "
        "Messages are concatenated (in fixed agent order) to each agent's local observation. "
        "This ensures efficient, explicit, and non-redundant sharing of all critical state dimensions for coordination."
    )

def communication(o):
    """
    o: Tensor of shape (batch, n_agents, obs_dim=98)
    Returns: Tensor of shape (batch, n_agents, obs_dim + n_agents*message_dim)
    """
    # Constants
    batch_size, n_agents, obs_dim = o.shape
    device = o.device

    # Indices for own info
    idx_own_health = 76
    idx_own_unit_type = [77, 78, 79]
    idx_own_pos_x = 80
    idx_own_pos_y = 81

    # Indices for enemy info (5 enemies)
    enemy_offsets = [4 + i*9 for i in range(5)]  # enemy_i_shootable is at 4,13,20,28,36
    idx_enemy_health = [offset+4 for offset in enemy_offsets]          # 8,16,24,32,40
    idx_enemy_rel_x = [offset+2 for offset in enemy_offsets]           # 6,14,22,30,38
    idx_enemy_rel_y = [offset+3 for offset in enemy_offsets]           # 7,15,23,31,39
    idx_enemy_distance = [offset+1 for offset in enemy_offsets]        # 5,13,21,29,37
    idx_enemy_unit_type = [[offset+5, offset+6, offset+7] for offset in enemy_offsets]  # 9-11,17-19,...

    # Indices for own absolute position
    idx_own_pos_x = 80
    idx_own_pos_y = 81

    # Sender identity (one-hot, length 5)
    sender_ids = th.eye(n_agents, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, n_agents, 5)

    # Own info: (health, pos_x, pos_y, unit_type[3])
    own_health = o[:, :, idx_own_health:idx_own_health+1]           # (batch, n_agents, 1)
    own_pos_x = o[:, :, idx_own_pos_x:idx_own_pos_x+1]              # (batch, n_agents, 1)
    own_pos_y = o[:, :, idx_own_pos_y:idx_own_pos_y+1]              # (batch, n_agents, 1)
    own_unit_type = o[:, :, idx_own_unit_type]                      # (batch, n_agents, 3)
    own_info = th.cat([own_health, own_pos_x, own_pos_y, own_unit_type], dim=-1)  # (batch, n_agents, 6)

    # For each enemy, compute absolute position if observable, else zero
    # Get own absolute position for each agent (for relative->absolute conversion)
    own_abs_x = own_pos_x  # (batch, n_agents, 1)
    own_abs_y = own_pos_y  # (batch, n_agents, 1)

    # Collect enemy info for all agents in batch
    enemy_health = th.stack([o[:, :, idx] for idx in idx_enemy_health], dim=2)  # (batch, n_agents, 5)
    enemy_rel_x = th.stack([o[:, :, idx] for idx in idx_enemy_rel_x], dim=2)    # (batch, n_agents, 5)
    enemy_rel_y = th.stack([o[:, :, idx] for idx in idx_enemy_rel_y], dim=2)    # (batch, n_agents, 5)
    enemy_unit_type = th.stack(
        [o[:, :, idxs] for idxs in idx_enemy_unit_type], dim=2
    )  # (batch, n_agents, 5, 3)

    # Convert relative positions to absolute
    # own_abs_x, own_abs_y: (batch, n_agents, 1)
    # enemy_rel_x, enemy_rel_y: (batch, n_agents, 5)
    enemy_abs_x = own_abs_x + enemy_rel_x    # (batch, n_agents, 5)
    enemy_abs_y = own_abs_y + enemy_rel_y    # (batch, n_agents, 5)

    # If enemy is unobservable (all zeros), zero out all corresponding values
    # enemy_health, enemy_abs_x, enemy_abs_y, enemy_unit_type
    # If health==0 & all unit_type==0, treat as unobservable
    enemy_observable = (enemy_health > 0) | (enemy_unit_type.sum(-1) > 0)  # (batch, n_agents, 5)
    # Expand mask for broadcasting
    enemy_mask = enemy_observable.float().unsqueeze(-1)  # (batch, n_agents, 5, 1)

    # Compose enemy info: [health, abs_x, abs_y, unit_type(3)]
    enemy_info = th.cat([
        enemy_health.unsqueeze(-1),         # (batch, n_agents, 5, 1)
        enemy_abs_x.unsqueeze(-1),          # (batch, n_agents, 5, 1)
        enemy_abs_y.unsqueeze(-1),          # (batch, n_agents, 5, 1)
        enemy_unit_type                     # (batch, n_agents, 5, 3)
    ], dim=-1)  # (batch, n_agents, 5, 6)
    enemy_info = enemy_info * enemy_mask    # Zero out if unobservable

    # Flatten enemy info per agent: (batch, n_agents, 5*6)
    enemy_info_flat = enemy_info.reshape(batch_size, n_agents, 5*6)

    # Final message per agent: [sender_id (5), own_info (6), enemy_info (30)] = 41
    message = th.cat([sender_ids, own_info, enemy_info_flat], dim=-1)  # (batch, n_agents, 41)
    message_dim = message.shape[-1]

    # Each agent receives all n_agents messages (excluding its own if desired, but here we include for simplicity)
    # For each agent, stack messages from all agents in fixed order (agent0, agent1,...)
    # For each agent, messages from all agents (batch, n_agents, n_agents*message_dim)
    # To avoid self-message, mask out if needed (not required per protocol, but possible)
    messages_all = message  # (batch, n_agents, message_dim)
    # Expand for cross-agent: for each agent, collect all n_agents messages
    # We want, for each agent, all messages from other agents
    # messages_to_concat: (batch, n_agents, n_agents, message_dim)
    messages_to_concat = messages_all.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (batch, n_agents, n_agents, message_dim)
    # For each agent, flatten all n_agents messages into one vector
    messages_concat = messages_to_concat.reshape(batch_size, n_agents, n_agents*message_dim)  # (batch, n_agents, n_agents*message_dim)

    # Final enhanced observation: (batch, n_agents, obs_dim + n_agents*message_dim)
    enhanced_o = th.cat([o, messages_concat], dim=-1)
    return enhanced_o
