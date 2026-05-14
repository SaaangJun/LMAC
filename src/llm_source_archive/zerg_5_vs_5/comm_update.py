import torch as th

def message_design_instruction():
    """
    Message Design Instruction for SMACv2 zerg_5_vs_5 Communication Protocol

    Protocol Overview:
    ------------------
    Each agent constructs a broadcast message containing only the most essential and unique information to maximize coordination under partial observability:
      - For each ally (excluding self, 4 total):
          - Visibility flag (1 if currently observed, 0 otherwise),
          - Health, absolute position (x, y), unit type (3 bits);
          - If unobservable, these fields are set to -1.
      - Local visibility masks for both enemies (5 bits) and allies (4 bits) are also included, so recipients can disambiguate "unseen" from "destroyed/never seen".
    
    All agents' messages are concatenated (in fixed order) to every agent's local observation, forming an enhanced observation tensor of shape (batch, n_agents, 98 + n_agents*37).

    This protocol:
      - Explicitly distinguishes between "not observed" and "zero value" via visibility flags and -1 padding.
      - Shares only unique, non-redundant, and non-inferable information.
      - Enables agents to resolve partial observability, coordinate targeting, support, and adapt to composition/position.

    """
    return (
        "Each agent broadcasts a message containing unique, non-redundant information: "
        "1) for each ally (excluding self, 4 total): [visibility flag (1/0), health, abs_x, abs_y, unit type (3 bits)], with -1 for unobservable (7 dims * 4 = 28 dims); "
        "2) 4-bit local ally visibility mask and 5-bit enemy visibility mask (9 dims). "
        "Sender ID, Own Info, and Enemy Info are excluded as they are redundant with the initialization message. "
        "Total message dimension per agent is 37. "
        "All agents' messages are concatenated to each agent's local observation, providing critical cross-agent visibility context."
    )

def communication(o):
    """
    o: Tensor of shape (batch, n_agents, obs_dim=98)
    Returns: Tensor of shape (batch, n_agents, obs_dim + n_agents*message_dim)
    """
    batch_size, n_agents, obs_dim = o.shape
    device = o.device
    # === Own info indices ===
    idx_own_pos_x = 80
    idx_own_pos_y = 81

    # === Own Absolute Position (needed for relative->absolute conversion) ===
    own_abs_x = o[:, :, idx_own_pos_x:idx_own_pos_x+1]
    own_abs_y = o[:, :, idx_own_pos_y:idx_own_pos_y+1]
    # === Ally info indices (excluding self) ===
    ally_offsets = [44 + i*9 for i in range(4)]
    idx_ally_visible = [offset for offset in ally_offsets]
    idx_ally_rel_x = [offset+2 for offset in ally_offsets]
    idx_ally_rel_y = [offset+3 for offset in ally_offsets]
    idx_ally_health = [offset+4 for offset in ally_offsets]
    idx_ally_unit_type = [[offset+5, offset+6, offset+7] for offset in ally_offsets]

    # === Ally info (excluding self) ===
    ally_visible = th.stack([o[:, :, idx] for idx in idx_ally_visible], dim=2)  # (batch, n_agents, 4)
    ally_health = th.stack([o[:, :, idx] for idx in idx_ally_health], dim=2)
    ally_rel_x = th.stack([o[:, :, idx] for idx in idx_ally_rel_x], dim=2)
    ally_rel_y = th.stack([o[:, :, idx] for idx in idx_ally_rel_y], dim=2)
    ally_unit_type = th.stack([o[:, :, idxs] for idxs in idx_ally_unit_type], dim=2)  # (batch, n_agents, 4, 3)
    # Absolute pos
    ally_abs_x = own_abs_x + ally_rel_x  # (batch, n_agents, 4)
    ally_abs_y = own_abs_y + ally_rel_y
    # Visibility: visible > 0
    ally_visible_mask = (ally_visible > 0).float()
    unobs_ally = -th.ones_like(ally_health)
    ally_health_filled = ally_health * ally_visible_mask + unobs_ally * (1 - ally_visible_mask)
    ally_abs_x_filled = ally_abs_x * ally_visible_mask + unobs_ally * (1 - ally_visible_mask)
    ally_abs_y_filled = ally_abs_y * ally_visible_mask + unobs_ally * (1 - ally_visible_mask)
    ally_unit_type_filled = ally_unit_type * ally_visible_mask.unsqueeze(-1) + unobs_ally.unsqueeze(-1) * (1 - ally_visible_mask.unsqueeze(-1))
    # Compose per-ally: [vis_flag, health, abs_x, abs_y, unit_type(3)]
    ally_info = th.cat([
        ally_visible_mask.unsqueeze(-1),         # (batch, n_agents, 4, 1)
        ally_health_filled.unsqueeze(-1),        # (batch, n_agents, 4, 1)
        ally_abs_x_filled.unsqueeze(-1),         # (batch, n_agents, 4, 1)
        ally_abs_y_filled.unsqueeze(-1),         # (batch, n_agents, 4, 1)
        ally_unit_type_filled                    # (batch, n_agents, 4, 3)
    ], dim=-1)  # (batch, n_agents, 4, 6)
    ally_info_flat = ally_info.reshape(batch_size, n_agents, -1)  # (batch, n_agents, 24)

    # === Ally and enemy local visibility masks ===
    # For each agent, 4 bits for ally vis (excluding self), 5 bits for enemy vis
    ally_vis_mask = ally_visible_mask  # (batch, n_agents, 4)
    enemy_vis_mask = enemy_visible     # (batch, n_agents, 5)

    # === Compose message per agent ===
    # Only Ally Info (28) + Ally Vis Mask (4) + Enemy Vis Mask (5) = 37 dims
    message = th.cat([
        ally_info_flat,        # (batch, n_agents, 28)
        ally_vis_mask,         # (batch, n_agents, 4)
        enemy_vis_mask         # (batch, n_agents, 5)
    ], dim=-1)  # (batch, n_agents, 37)

    message_dim = message.shape[-1]  # Should be 37

    # === Broadcast: Each agent receives all n_agents messages (including its own), concatenated ===
    # (batch, n_agents, n_agents, message_dim)
    messages_to_concat = message.unsqueeze(1).expand(-1, n_agents, -1, -1)
    # (batch, n_agents, n_agents*message_dim)
    messages_concat = messages_to_concat.reshape(batch_size, n_agents, n_agents*message_dim)

    # === Final enhanced observation ===
    enhanced_o = th.cat([o, messages_concat], dim=-1)
    return enhanced_o
