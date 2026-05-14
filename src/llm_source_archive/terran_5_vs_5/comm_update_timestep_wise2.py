import torch as th

def message_design_instruction():
    """
    Task-specific communication protocol for SMACv2 terran_5_vs_5 scenario (Enhanced Temporal & Intent Sharing):

    Objective:
    Enable each agent to share only essential, non-redundant, and non-locally observable information,
    leveraging both temporal and intent cues, to maximize consistent and accurate global state inference
    and coordination—especially for weakly observable enemy positions and team tactics.

    Message Content Construction:
    Each agent constructs a message containing:
    1. Own recent trajectory (last 2 steps, excluding current):
        - Absolute position x (2), y (2)
        - Health (2)
        - Last actions (2x11 one-hot = 22)
    2. Own current movement intent: (5-dim one-hot, N/S/E/W/Stay, derived from move action values)
    3. Attack focus: (which enemy, 5-dim one-hot, from last action attack_enemy_x)
    4. Support focus: (Medivac only, 5-dim one-hot, index of closest ally)
    5. Enemy movement/position summary for each enemy (5 enemies): 
        - If seen in last 3 steps: 
            - (dx, dy) relative movement (1,1)
            - last health (1)
            - unit type (3)
            - "last seen time" (normalized, 1)
        - If not seen: 
            - mean and std of rel_x/rel_y over last 3 steps (4)
            - zeros for other fields (4)
        (8 dims per enemy x 5 = 40)
    6. Observed ally movement summary for each other ally (4 allies): 
        - If seen in last 3 steps: 
            - (dx, dy) relative movement (1,1)
            - last health (1)
            - unit type (3)
            - "last seen time" (normalized, 1)
        - If not seen: 
            - zeros (7)
        (7 dims per ally x 4 = 28)
    
    Each message: 
        2 + 2 + 2 + 22 (own traj) + 5 (intent) + 5 (attack) + 5 (support) + 40 (enemies) + 28 (allies) = 111 dims

    Message Distribution:
    - Broadcast: Each message is sent to all teammates (not self).
    - Each agent receives the concatenated messages from all other agents (4 × 111 = 444 dims).
    - The enhanced observation for each agent: [original 98-dim obs at last step | messages from other 4 agents (444 dims)]
      = 542 dims.

    Protocol ensures:
    - All agents share temporal and behavioral cues, supporting inference of enemy and ally positions and intent.
    - Communication enables agents to align beliefs about global state, even under severe partial observability.
    - The protocol is vectorized for batch efficiency and avoids redundant or easily inferable information.

    """
    return (
        "Each agent constructs a message containing: "
        "1) own recent trajectory over last 2 timesteps (excluding current): abs x (2), abs y (2), health (2), last actions (22); "
        "2) current movement intent (one-hot, 5-dim: north, south, east, west, stay); "
        "3) current attack focus (which enemy, 5-dim one-hot); "
        "4) support focus (Medivac only, 5-dim one-hot for closest ally); "
        "5) for each enemy (5): if seen in last 3 steps, (dx, dy movement, last health, unit type (3), last seen time); "
        "   if not seen, share mean/std of rel_x/rel_y (4), zeros for other fields (4); (8 dims × 5 = 40); "
        "6) for each other ally (4): if seen in last 3 steps, (dx, dy, last health, unit type (3), last seen time); "
        "   if not seen, zeros; (7 dims × 4 = 28). "
        "Each message is broadcast to all teammates (not self). "
        "Each agent's enhanced observation is the concatenation of its original 98-dim obs at last step and the 4 messages received from teammates "
        "(total 98+444=542 dims). "
    )

def communication(o):
    """
    Input: 
        o: Tensor of shape (batch, T, n_agents=5, obs_dim=98)
            Contains information from the previous up to 10 steps (T>=1) per agent
    Output: 
        messages_o: Tensor of shape (batch, n_agents=5, 98 + 444)
    """
    B, T, N, D = o.shape  # (batch, T, 5, 98)
    device = o.device
    dtype = o.dtype

    last_steps = 3
    # Pad time if needed
    if T < last_steps:
        pad = th.zeros(B, last_steps-T, N, D, device=device, dtype=dtype)
        o_padded = th.cat([pad, o], dim=1)
    else:
        o_padded = o
    o_last3 = o_padded[:, -last_steps:, :, :]  # (B, 3, N, 98)


    # --- 2. Own Recent Trajectory: last 2 abs positions, health, last-actions (Excluding current) ---
    # We want t-2 and t-1. o_last3 has indices 0 (t-2), 1 (t-1), 2 (t).
    # Take indices 0 and 1.
    trajectory_steps = 2
    o_traj = o_last3[:, :trajectory_steps, :, :] # (B, 2, N, 98)

    own_abs_x = o_traj[..., 80].permute(0,2,1)   # (B, N, 2)
    own_abs_y = o_traj[..., 81].permute(0,2,1)   # (B, N, 2)
    own_health = o_traj[..., 76].permute(0,2,1)  # (B, N, 2)
    own_last_action = o_traj[..., 82:93].permute(0,2,1,3).reshape(B,N,trajectory_steps*11) # (B,N,22)

    # --- 3. Own Current Movement Intent ---
    move_idx = [0,1,2,3]  # north, south, east, west
    move_vals = o_padded[:, -1, :, move_idx]  # (B, N, 4)
    stay_mask = (move_vals.abs().sum(-1, keepdim=True) < 1e-5).float()  # (B, N, 1)
    move_dir = th.cat([move_vals, stay_mask], dim=-1)  # (B,N,5)
    move_dir_max = move_dir.max(-1, keepdim=True)[0]
    move_dir_oh = (move_dir == move_dir_max).float()  # (B,N,5)

    # --- 4. Attack Focus (which enemy being attacked in last action) ---
    last_action_att = o_padded[:, -1, :, 88:93]  # (B,N,5), attack_enemy_0~4
    attack_focus_max = last_action_att.max(-1, keepdim=True)[0]
    attack_focus_oh = (last_action_att == attack_focus_max).float()  # (B,N,5)

    # --- 5. Support Focus (Medivac: closest ally, others: zeros) ---
    is_medivac = (o_padded[:, -1, :, 77:80] == th.tensor([0,0,1], device=device, dtype=dtype)).all(-1)  # (B,N)
    # For each agent, find closest other agent (using ally_*_distance)
    ally_distance_idx = [45,53,61,69]
    ally_distances = th.stack([o_padded[:, -1, :, idx] for idx in ally_distance_idx], dim=-1)  # (B,N,4)
    inf = th.full((B,N,1), 1e6, device=device, dtype=dtype)
    ally_distances_full = th.cat([inf, ally_distances], dim=-1)  # (B,N,5)
    # For each agent i, set ally_distances_full[:,i,i] = inf
    eye_mask = th.eye(N, device=device, dtype=th.bool).unsqueeze(0).expand(B,N,N) # (B,N,N)
    ally_distances_full = ally_distances_full.clone()
    ally_distances_full[eye_mask] = 1e6
    min_idx = ally_distances_full.min(-1)[1]  # (B,N)
    support_focus_oh = th.zeros(B,N,5,device=device,dtype=dtype)
    support_focus_oh.scatter_(-1, min_idx.unsqueeze(-1), 1.0)
    support_focus_oh = support_focus_oh * is_medivac.unsqueeze(-1).float()  # Only Medivac have nonzero

    # --- 6. Observed Enemy Movement & Belief Summary ---
    enemy_relx_idx = [6,14,22,30,38]
    enemy_rely_idx = [7,15,23,31,39]
    enemy_health_idx = [8,16,24,32,40]
    enemy_utype_idx = [
        [9,10,11], [17,18,19], [25,26,27], [33,34,35], [41,42,43]
    ]
    enemy_shootable_idx = [4,12,20,28,36]

    # Build enemy feature tensors (B, 3, N, 5)
    enemy_relx = th.stack([o_last3[..., idx] for idx in enemy_relx_idx],dim=-1)   # (B,3,N,5)
    enemy_rely = th.stack([o_last3[..., idx] for idx in enemy_rely_idx],dim=-1)
    enemy_health = th.stack([o_last3[..., idx] for idx in enemy_health_idx],dim=-1)
    enemy_shootable = th.stack([o_last3[..., idx] for idx in enemy_shootable_idx],dim=-1)  # (B,3,N,5)
    enemy_utype = th.stack([o_last3[..., idxs] for idxs in enemy_utype_idx], dim=-2) #(B,3,N,5,3)

    # Boolean mask: (B,3,N,5)
    enemy_shootable_bin = (enemy_shootable > 0.5)
    flip_shootable = th.flip(enemy_shootable_bin, dims=[1])  # (B,3,N,5)
    any_seen = flip_shootable.any(1)  # (B,N,5)
    idx_first_seen = flip_shootable.float().argmax(dim=1)  # (B,N,5)
    idx_first_seen = th.where(any_seen, idx_first_seen, th.full_like(idx_first_seen, last_steps))
    last_seen_t = last_steps-1 - idx_first_seen.clamp(0,last_steps-1)  # (B,N,5)

    # Batch, agent, enemy indices for gather
    batch_idx = th.arange(B, device=device)[:,None,None].expand(B,N,5)
    agent_idx = th.arange(N, device=device)[None,:,None].expand(B,N,5)
    enemy_idx_t = th.arange(5, device=device)[None,None,:].expand(B,N,5)
    seen_t = last_seen_t.clamp(0,last_steps-1)

    relx_last = enemy_relx[batch_idx, seen_t, agent_idx, enemy_idx_t]  # (B,N,5)
    rely_last = enemy_rely[batch_idx, seen_t, agent_idx, enemy_idx_t]
    health_last = enemy_health[batch_idx, seen_t, agent_idx, enemy_idx_t]
    utype_last = enemy_utype[batch_idx, seen_t, agent_idx, enemy_idx_t]  # (B,N,5,3)
    relx_prev = enemy_relx[batch_idx, (seen_t-1).clamp(0,last_steps-1), agent_idx, enemy_idx_t]
    rely_prev = enemy_rely[batch_idx, (seen_t-1).clamp(0,last_steps-1), agent_idx, enemy_idx_t]
    dx = relx_last - relx_prev
    dy = rely_last - rely_prev
    last_seen_time = (seen_t.float()+1)/last_steps  # (B,N,5)

    # For unobserved enemies (never observed in last 3), share inferred pos mean/std
    ever_obs = any_seen  # (B,N,5)
    ever_obs_float = ever_obs.float()
    obs_count = enemy_shootable_bin.float().sum(1) + 1e-6  # avoid div0, (B,N,5)
    relx_mean = (enemy_relx * enemy_shootable_bin.float()).sum(1) / obs_count
    rely_mean = (enemy_rely * enemy_shootable_bin.float()).sum(1) / obs_count
    relx_sq = (enemy_relx**2 * enemy_shootable_bin.float()).sum(1) / obs_count
    rely_sq = (enemy_rely**2 * enemy_shootable_bin.float()).sum(1) / obs_count
    relx_std = (relx_sq - relx_mean**2).clamp(min=0).sqrt()
    rely_std = (rely_sq - rely_mean**2).clamp(min=0).sqrt()
    # Only for unobserved (masking)
    relx_mean_pad = relx_mean * (~ever_obs).float()
    rely_mean_pad = rely_mean * (~ever_obs).float()
    relx_std_pad = relx_std * (~ever_obs).float()
    rely_std_pad = rely_std * (~ever_obs).float()
    # 4 zeros for the fields (dx, dy, health, last_seen_time) that are not present for unobserved
    zeros4 = th.zeros(B,N,5,4,device=device,dtype=dtype)
    fallback_fields = th.cat([relx_mean_pad.unsqueeze(-1), rely_mean_pad.unsqueeze(-1),
                              relx_std_pad.unsqueeze(-1), rely_std_pad.unsqueeze(-1), zeros4],dim=-1)  # (B,N,5,8)

    obs_fields = th.cat([
        dx.unsqueeze(-1), dy.unsqueeze(-1), health_last.unsqueeze(-1), utype_last, last_seen_time.unsqueeze(-1)
    ], dim=-1)  # (B,N,5,7)
    # For seen: [dx,dy,health,unit_type(3),last_seen_time], for unseen: [relx_mean, rely_mean, relx_std, rely_std, zeros(4)]
    # Pad obs_fields with 1 zero at the end to make it (B,N,5,8)
    obs_fields_padded = th.cat([obs_fields, th.zeros(B,N,5,1,device=device,dtype=dtype)], dim=-1)  # (B,N,5,8)

    obs_mask = ever_obs.unsqueeze(-1).float()  # (B,N,5,1)
    # obs_fields_padded and fallback_fields both have 8 dims at the end
    enemy_info = obs_fields_padded * obs_mask + fallback_fields * (1.0-obs_mask)  # (B,N,5,8)
    enemy_info_flat = enemy_info.reshape(B,N,5*8)

    # --- 7. Observed Ally Movement Summary (for each other ally) ---
    ally_relx_idx = [46,54,62,70]
    ally_rely_idx = [47,55,63,71]
    ally_health_idx = [48,56,64,72]
    ally_utype_idx = [
        [49,50,51], [57,58,59], [65,66,67], [73,74,75]
    ]
    ally_visible_idx = [44,52,60,68]
    ally_relx = th.stack([o_last3[..., idx] for idx in ally_relx_idx],dim=-1) #(B,3,N,4)
    ally_rely = th.stack([o_last3[..., idx] for idx in ally_rely_idx],dim=-1)
    ally_health = th.stack([o_last3[..., idx] for idx in ally_health_idx],dim=-1)
    ally_utype = th.stack([o_last3[..., idxs] for idxs in ally_utype_idx], dim=-2) #(B,3,N,4,3)
    ally_visible = th.stack([o_last3[..., idx] for idx in ally_visible_idx],dim=-1) #(B,3,N,4)
    ally_visible_bin = (ally_visible > 0.5).float()
    flip_ally_visible = th.flip(ally_visible_bin, dims=[1])
    any_seen_ally = flip_ally_visible.any(1)  # (B,N,4)
    idx_first_seen_ally = flip_ally_visible.float().argmax(dim=1)  # (B,N,4)
    idx_first_seen_ally = th.where(any_seen_ally, idx_first_seen_ally, th.full_like(idx_first_seen_ally, last_steps))
    last_seen_t_ally = last_steps-1 - idx_first_seen_ally.clamp(0,last_steps-1)  # (B,N,4)
    # Indices for gather
    batch_idx4 = th.arange(B, device=device)[:,None,None].expand(B,N,4)
    agent_idx4 = th.arange(N, device=device)[None,:,None].expand(B,N,4)
    ally_idx4 = th.arange(4, device=device)[None,None,:].expand(B,N,4)
    seen_t_ally = last_seen_t_ally.clamp(0,last_steps-1)
    relx_last_ally = ally_relx[batch_idx4, seen_t_ally, agent_idx4, ally_idx4]
    rely_last_ally = ally_rely[batch_idx4, seen_t_ally, agent_idx4, ally_idx4]
    health_last_ally = ally_health[batch_idx4, seen_t_ally, agent_idx4, ally_idx4]
    utype_last_ally = ally_utype[batch_idx4, seen_t_ally, agent_idx4, ally_idx4]  # (B,N,4,3)
    relx_prev_ally = ally_relx[batch_idx4, (seen_t_ally-1).clamp(0,last_steps-1), agent_idx4, ally_idx4]
    rely_prev_ally = ally_rely[batch_idx4, (seen_t_ally-1).clamp(0,last_steps-1), agent_idx4, ally_idx4]
    dx_ally = relx_last_ally - relx_prev_ally
    dy_ally = rely_last_ally - rely_prev_ally
    last_seen_time_ally = (seen_t_ally.float()+1)/last_steps  # (B,N,4)
    ally_info = th.cat([
        dx_ally.unsqueeze(-1), dy_ally.unsqueeze(-1), health_last_ally.unsqueeze(-1), utype_last_ally, last_seen_time_ally.unsqueeze(-1)
    ], dim=-1) # (B,N,4,7)
    # For allies never seen, fill with zeros
    ally_obs_mask = any_seen_ally.unsqueeze(-1).float()
    ally_info = ally_info * ally_obs_mask  # (B,N,4,7)
    ally_info_flat = ally_info.reshape(B,N,4*7)

    # --- Compose message: 111 dims ---
    msg_parts = [
        own_abs_x,                    # (B,N,2)
        own_abs_y,                    # (B,N,2)
        own_health,                   # (B,N,2)
        own_last_action,              # (B,N,22)
        move_dir_oh,                  # (B,N,5)
        attack_focus_oh,              # (B,N,5)
        support_focus_oh,             # (B,N,5)
        enemy_info_flat,              # (B,N,40)
        ally_info_flat                # (B,N,28)
    ]
    message = th.cat(msg_parts, dim=-1)  # (B,N,111)

    # --- Distribute messages: for each agent, collect messages from all other agents (not self) ---
    message_exp = message.unsqueeze(1).expand(B, N, N, message.shape[-1])  # (B, receiver, sender, 111)
    mask = ~th.eye(N, dtype=th.bool, device=device).unsqueeze(0).expand(B, N, N)  # (B, N, N)
    msg_others = message_exp[mask].view(B, N, N-1, message.shape[-1])      # (B, N, 4, 111)
    msg_others_flat = msg_others.reshape(B, N, 4*message.shape[-1])        # (B, N, 444)

    # --- Concatenate to current observation (last step only) ---
    o_last = o_padded[:, -1, :, :]   # (B, N, 98)
    messages_o = th.cat([o_last, msg_others_flat], dim=-1)    # (B, N, 98+444=542)

    return messages_o
