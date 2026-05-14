import torch as th

def message_design_instruction():
    """
    Temporal-Intent Communication Protocol for SMACv2 zerg_5_vs_5

    Overview:
    ----------
    Each agent broadcasts a temporally-structured message containing:
      1. Own 9-step trajectory & intent (excluding current step):
         - For the past 9 timesteps: [absolute x, absolute y, health, unit type (3 bits), last action (7-hot), inferred intent (3-hot)], for a total of 16 fields per step.
      2. For each enemy (5 total):
         - Last seen absolute x, y, health, unit type (3 bits), and time since last seen.
         - If never seen, all fields are set to -1, and time since last seen is 10.
      
    All agents' messages are concatenated (in fixed order) to every agent's local observation, forming an enhanced observation tensor of shape (batch_size, n_agents, obs_dim + n_agents*message_dim), where message_dim = 9*16 (trajectory) + 5*7 (enemy last-seen) = 179.

    This protocol:
      - Explicitly encodes temporal context and intent for robust trajectory/intent inference.
      - Uses a window of 9 steps (t-9 to t-1) to avoid overlap with comm_init (current step t).
      - Is computationally efficient (fully vectorized).
    """
    return (
        "Each agent broadcasts a message containing: "
        "1) Own 9-step temporal trajectory (excluding current step): for each of the past 9 timesteps, [absolute x, absolute y, health, unit type (3 bits), last action (7-hot), inferred intent (3-hot)], for a total of 16 fields per step (9*16=144 dims); "
        "2) For each enemy (5): last seen [absolute x, absolute y, health, unit type (3 bits), time since last seen], all set to -1 if never seen in window (35 dims); "
        "Sender ID is excluded (redundant). "
        "Total message dimension per agent is 179. "
        "All agents' messages are concatenated to each agent's local observation, yielding (batch, n_agents, obs_dim + n_agents*179)."
    )


def communication(o):
    """
    Temporal-intent-aware protocol for SMACv2 zerg_5_vs_5.
    Args:
        o: (batch, T, n_agents, obs_dim=98)  # T >= 10, last 10 steps are relevant
    Returns:
        enhanced_o: (batch, n_agents, obs_dim + n_agents*message_dim), for current time step (last in window)
    """
    batch, T, n_agents, obs_dim = o.shape
    device = o.device
    
    window = 9
    assert T >= window + 1, "Observation buffer must contain at least 10 steps (last one excluded)"



    # --- 1. Own Temporal Trajectory & Intent (for window of 9 steps) ---
    idx_own_health = 76
    idx_own_unit_type = [77, 78, 79]
    idx_own_pos_x = 80
    idx_own_pos_y = 81
    idx_last_action = list(range(82, 93))  # 11 actions

    # Exclude current step (T-1 index 0..T-1). So we want T-window-1 to T-1.
    # e.g. T=10, window=9. T-1=9. range(0, 9) = 0..8.
    step_idxs = th.arange(T - window - 1, T - 1, device=device)  # (window,) (BATCH, WINDOW, ...)
    o_win = o[:, step_idxs, :, :]  # (batch, window, n_agents, obs_dim)

    # Gather own features
    own_health_win = o_win[..., idx_own_health]  # (batch, window, n_agents)
    own_pos_x_win = o_win[..., idx_own_pos_x]
    own_pos_y_win = o_win[..., idx_own_pos_y]
    own_unit_type_win = o_win[..., idx_own_unit_type]  # (batch, window, n_agents, 3)
    last_action_all = o_win[..., idx_last_action]      # (batch, window, n_agents, 11)

    # Reduce last action to 7-hot (no-op, stop, move_north, move_south, move_east, move_west, any_attack)
    attack_any = last_action_all[..., 6:11].max(dim=-1, keepdim=True)[0]  # (batch, window, n_agents, 1)
    last_action_7 = th.cat([
        last_action_all[..., 0:6],  # no-op, stop, move_north, move_south, move_east, move_west
        attack_any                  # any_attack
    ], dim=-1)  # (batch, window, n_agents, 7)

    # Inferred intent (attack, support, move/retreat)
    # Medivac: unit_type third bit is 1, others are 0
    is_medivac = (own_unit_type_win[..., 2] > 0.5) & (own_unit_type_win[..., 0:2].sum(-1) < 0.5)  # (batch, window, n_agents)
    attack_flag = last_action_7[..., 6] > 0.5  # attack_any
    move_flag = last_action_7[..., 2:6].sum(-1) > 0.5
    stop_flag = last_action_7[..., 1] > 0.5
    intent = th.zeros(o_win.shape[0], window, n_agents, 3, device=device)
    # intent[..., 0]: attack
    intent[..., 0] = attack_flag.float()
    # intent[..., 1]: support (medivac & (move or stop))
    intent[..., 1] = (is_medivac & (move_flag | stop_flag)).float()
    # intent[..., 2]: move/retreat (non-medivac & move)
    intent[..., 2] = ((~is_medivac) & move_flag).float()

    # Shape to (batch, n_agents, window, feat)
    own_pos_x_win_t = own_pos_x_win.permute(0, 2, 1)  # (batch, n_agents, window)
    own_pos_y_win_t = own_pos_y_win.permute(0, 2, 1)
    own_health_win_t = own_health_win.permute(0, 2, 1)
    own_unit_type_win_t = own_unit_type_win.permute(0, 2, 1, 3)
    last_action_7_t = last_action_7.permute(0, 2, 1, 3)
    intent_t = intent.permute(0, 2, 1, 3)

    # Concatenate all features per timestep: [x, y, health, unit_type(3), last_action(7), intent(3)] -> 16 per step
    own_traj = th.cat([
        own_pos_x_win_t.unsqueeze(-1),   # (batch, n_agents, window, 1)
        own_pos_y_win_t.unsqueeze(-1),
        own_health_win_t.unsqueeze(-1),
        own_unit_type_win_t,             # (batch, n_agents, window, 3)
        last_action_7_t,                 # (batch, n_agents, window, 7)
        intent_t                         # (batch, n_agents, window, 3)
    ], dim=-1)  # (batch, n_agents, window, 16)
    own_traj_flat = own_traj.reshape(batch, n_agents, window * 16)  # (batch, n_agents, 144)

    # --- 2. Last-Seen Enemy Info (per enemy, 5 enemies) ---
    enemy_offsets = [4 + i*9 for i in range(5)]
    idx_enemy_shootable = [offset for offset in enemy_offsets]
    idx_enemy_rel_x = [offset+2 for offset in enemy_offsets]
    idx_enemy_rel_y = [offset+3 for offset in enemy_offsets]
    idx_enemy_health = [offset+4 for offset in enemy_offsets]
    idx_enemy_unit_type = [[offset+5, offset+6, offset+7] for offset in enemy_offsets]

    # For each step in window, each agent, each enemy: get [shootable, rel_x, rel_y, health, unit_type]
    shootable_win = th.stack([o_win[..., idx] for idx in idx_enemy_shootable], dim=-1)  # (batch, window, n_agents, 5)
    rel_x_win = th.stack([o_win[..., idx] for idx in idx_enemy_rel_x], dim=-1)
    rel_y_win = th.stack([o_win[..., idx] for idx in idx_enemy_rel_y], dim=-1)
    health_win = th.stack([o_win[..., idx] for idx in idx_enemy_health], dim=-1)
    unit_type_win = th.stack([o_win[..., idxs] for idxs in idx_enemy_unit_type], dim=-2)  # (batch, window, n_agents, 5, 3)
    own_pos_x_win_e = o_win[..., idx_own_pos_x]
    own_pos_y_win_e = o_win[..., idx_own_pos_y]
    abs_x_win = own_pos_x_win_e.unsqueeze(-1) + rel_x_win  # (batch, window, n_agents, 5)
    abs_y_win = own_pos_y_win_e.unsqueeze(-1) + rel_y_win

    # For each agent, enemy: get last time step seen, last known (x,y,health,type)
    is_visible = (shootable_win > 0.5)  # (batch, window, n_agents, 5)
    is_visible_rev = is_visible.flip(1)  # reverse window dim (so argmax gives last seen index from end)
    last_seen_idx = is_visible_rev.float().argmax(dim=1)  # (batch, n_agents, 5)
    never_seen = (is_visible.sum(1) == 0)  # (batch, n_agents, 5)
    sel_idx = (window - 1 - last_seen_idx).clamp(0, window-1)  # (batch, n_agents, 5)

    # Prepare for advanced indexing
    batch_idx = th.arange(batch, device=device)[:, None, None].expand(batch, n_agents, 5)
    agent_idx = th.arange(n_agents, device=device)[None, :, None].expand(batch, n_agents, 5)
    enemy_idx = th.arange(5, device=device)[None, None, :].expand(batch, n_agents, 5)

    # Helper to gather [batch, window, n_agents, 5] with sel_idx [batch, n_agents, 5]
    def gather_last_seen(feat_win):
        # feat_win: (batch, window, n_agents, 5[, D])
        # Output: (batch, n_agents, 5[, D])
        shape = feat_win.shape
        if len(shape) == 5:
            # (batch, window, n_agents, 5, D)
            D = shape[-1]
            gather_idx = sel_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (batch, n_agents, 5, D)
            feat_win_perm = feat_win.permute(0,2,3,1,4)  # (batch, n_agents, 5, window, D)
            out = feat_win_perm.gather(3, gather_idx.unsqueeze(-2)).squeeze(3)  # (batch, n_agents, 5, D)
            # Mask for never seen
            mask = never_seen.unsqueeze(-1).expand(-1, -1, -1, D)
            out = out.masked_fill(mask, -1)
            return out
        else:
            # (batch, window, n_agents, 5)
            gather_idx = sel_idx  # (batch, n_agents, 5)
            feat_win_perm = feat_win.permute(0,2,3,1)  # (batch, n_agents, 5, window)
            out = feat_win_perm.gather(3, gather_idx.unsqueeze(-1)).squeeze(3)  # (batch, n_agents, 5)
            # Mask for never seen
            out = out.masked_fill(never_seen, -1)
            return out

    last_seen_abs_x = gather_last_seen(abs_x_win)
    last_seen_abs_y = gather_last_seen(abs_y_win)
    last_seen_health = gather_last_seen(health_win)
    last_seen_unit_type = gather_last_seen(unit_type_win)

    # Time since last seen: (window-1 - sel_idx), or window if never seen
    time_since_last_seen = (window - 1 - sel_idx).float()
    time_since_last_seen = time_since_last_seen.masked_fill(never_seen, window)

    # Compose enemy info: [abs_x, abs_y, health, unit_type(3), time_since_last_seen] per enemy
    enemy_last_seen = th.cat([
        last_seen_abs_x.unsqueeze(-1),    # (batch, n_agents, 5, 1)
        last_seen_abs_y.unsqueeze(-1),
        last_seen_health.unsqueeze(-1),
        last_seen_unit_type,              # (batch, n_agents, 5, 3)
        time_since_last_seen.unsqueeze(-1)
    ], dim=-1)  # (batch, n_agents, 5, 7)
    enemy_last_seen_flat = enemy_last_seen.reshape(batch, n_agents, 5*7)  # (batch, n_agents, 35)

    # --- 3. Compose message per agent ---
    # own_traj_flat (144), enemy_last_seen_flat (35)
    message = th.cat([
        own_traj_flat,        # (batch, n_agents, 144)
        enemy_last_seen_flat  # (batch, n_agents, 35)
    ], dim=-1)  # (batch, n_agents, message_dim=179)

    message_dim = message.shape[-1]  # Should be 200

    # --- 4. Broadcast: Each agent receives all n_agents messages (including its own), concatenated ---
    # (batch, n_agents, n_agents, message_dim)
    messages_to_concat = message.unsqueeze(1).expand(-1, n_agents, -1, -1)
    # (batch, n_agents, n_agents*message_dim)
    messages_concat = messages_to_concat.reshape(batch, n_agents, n_agents*message_dim)

    # --- 5. Final enhanced observation: use only current (last) step obs for each agent ---
    o_cur = o[:, -1, :, :]  # (batch, n_agents, obs_dim)
    enhanced_o = th.cat([o_cur, messages_concat], dim=-1)
    return enhanced_o
