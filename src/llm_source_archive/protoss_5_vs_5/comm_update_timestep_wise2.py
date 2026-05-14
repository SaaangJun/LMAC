import torch as th

def message_design_instruction():
    """
    Enhanced Message Design Instruction for protoss_5_vs_5 (SMACv2):

    1. Purpose:
        - Facilitate robust coordination and state inference under partial observability by sharing not only static entity attributes but also temporal, behavioral, and intent cues.

    2. Message Content (per agent, per message):
        A. Temporal Action History:
            - Last 5 actions taken by the agent (each as an 11-dimensional one-hot, for all action types: no-op, stop, move directions, attack_enemy_0..4).

        B. Intended Action (Current Step):
            - One-hot vector (length 11) of the agent's current action (most recent in history).

        C. Recent Movement Vectors:
            - Displacement vectors (dx, dy) for the last 5 timesteps, calculated from absolute positions.

        D. Attack Target Focus:
            - One-hot vector (length 6): [no target, attacking enemy_0, ..., attacking enemy_4] indicating which enemy (if any) the agent is currently attacking.

        E. Help Request Flag:
            - 1 if current shield < 0.2 (normalized), else 0. Indicates low shield/need for support.

        F. Freshest Enemy Sightings (per enemy, 5 enemies):
            - For each enemy, the most recent observation within the last 10 timesteps **EXCLUDING the current step**:
                [seen_flag (1), health (1), abs_x (1), abs_y (1)].
            - If never observed in history, seen_flag=0, other fields=-1.


    3. Protocol Mechanics:
        - Each agent broadcasts its message to all other agents (peer-to-peer, not to self).
        - Each agent receives 4 such messages (from the other agents), concatenates them in sender-index order to its own current observation.
        - Output shape: (batch, 5, 108 + 412) = (batch, 5, 520), where 412 = 4 * 103 (message_dim=103 per message).

    4. Efficiency:
        - All computations are vectorized, no per-batch or per-agent for-loops.

    5. Explicitness:
        - All fields are explicit; missingness is unambiguous (seen_flag, -1 fill).
    """
    return (
        "Each agent constructs a message containing:\n"
        "- Last 5 actions (each as 11-d one-hot, flattened to 55)\n"
        "- Intended action for the current step (11-d one-hot)\n"
        "- Recent movement vectors (dx, dy for last 5 steps, 10 floats)\n"
        "- Target focus (one-hot, length 6: [no target, enemy_0..enemy_4])\n"
        "- Help request flag (1 if shield < 0.2 else 0)\n"
        "- Freshest enemy sightings over last 10 steps (EXCLUDING current) for each enemy (5 enemies × [seen_flag, health, abs_x, abs_y])\n"
        "Total message dim: 103 (Sender ID removed).\n"
        "Each agent receives 4 such messages (from other agents), concatenates them to its own observation.\n"
        "Output: (batch, 5, 520)."
    )

def communication(o):
    """
    Communication function for protoss_5_vs_5 with temporal, intent, and coordination cues.
    Args:
        o: Tensor of shape (batch, T, 5, 108), where T >= 10 (last 10 steps incl. current)
    Returns:
        Tensor of shape (batch, 5, 108 + 412) = (batch, 5, 520)
    """
    device = o.device
    batch, T, n_agents, n_obs = o.shape
    assert n_agents == 5
    assert n_obs == 108
    assert T >= 10, "Input must have at least 10 timesteps (last 10 incl. current)"

    # --- 1. Sender identity (Removed: duplicated) ---
    # sender_id = ...

    # --- 2. Temporal Action History (last 5 actions) ---
    # Action one-hot indices: 92-102 (11 actions)
    action_idx = th.arange(92, 103, device=device)  # (11,)
    # For last 5 timesteps (most recent at -1)
    last5 = o[:, -5:, :, :]  # (batch, 5, 5, 108)
    # Rearrange to (batch, agents, 5, 11)
    actions_last5 = last5[..., action_idx]  # (batch, 5, 5, 11)
    actions_last5 = actions_last5.permute(0, 2, 1, 3).reshape(batch, n_agents, -1)  # (batch, 5, 55)

    # --- 3. Intended Action (use most recent action) ---
    last_action = o[:, -1, :, action_idx]  # (batch, 5, 11)

    # --- 4. Recent Movement Vector (last 5, dx, dy) ---
    pos_x = o[:, -6:, :, 87]  # (batch, 6, 5)
    pos_y = o[:, -6:, :, 88]  # (batch, 6, 5)
    # Compute displacements for last 5 steps: dx = x[t] - x[t-1]
    dx = pos_x[:, 1:, :] - pos_x[:, :-1, :]  # (batch, 5, 5)
    dy = pos_y[:, 1:, :] - pos_y[:, :-1, :]  # (batch, 5, 5)
    # Rearrange to (batch, agents, 5) for dx and dy, then flatten last 5
    dx = dx.permute(0, 2, 1)  # (batch, 5, 5)
    dy = dy.permute(0, 2, 1)  # (batch, 5, 5)
    movement = th.cat([dx, dy], dim=-1)  # (batch, 5, 10)

    # --- 5. Target Focus (coordination cue) ---
    # For most recent step: check which attack_enemy_* action is 1
    attack_action_idx = th.arange(98, 103, device=device)  # attack_enemy_0..4
    attack_action = o[:, -1, :, attack_action_idx]  # (batch, 5, 5)
    # For each agent, argmax if any attack, else 'no target'
    attack_mask = (attack_action > 0.5)
    any_attack = attack_mask.any(dim=-1, keepdim=True)  # (batch, 5, 1)
    # One-hot: [no target, enemy_0..4], length 6
    target_focus = th.cat([
        (~any_attack).float(),  # (batch, 5, 1)
        attack_mask.float()     # (batch, 5, 5)
    ], dim=-1)  # (batch, 5, 6)

    # --- 6. Low shield/help request ---
    own_shield = o[:, -1, :, 86]  # (batch, 5)
    help_flag = (own_shield < 0.2).float().unsqueeze(-1)  # (batch, 5, 1)


    enemy_offsets = th.arange(5, device=device) * 9
    health_idx = 8 + enemy_offsets
    relx_idx = 6 + enemy_offsets
    rely_idx = 7 + enemy_offsets
    shootable_idx = 4 + enemy_offsets

    # Slice EXCLUDING current step
    o_hist = o[:, :-1, :, :] # (batch, T-1, 5, 108)
    
    # We need absolute positions for history steps.
    own_x = o_hist[..., 87].unsqueeze(-1)  # (batch, T-1, 5, 1)
    own_y = o_hist[..., 88].unsqueeze(-1)  # (batch, T-1, 5, 1)
    
    # For each enemy, get fields over history
    enemy_health = o_hist[..., health_idx]  # (batch, T-1, 5, 5)
    enemy_relx = o_hist[..., relx_idx]      # (batch, T-1, 5, 5)
    enemy_rely = o_hist[..., rely_idx]      # (batch, T-1, 5, 5)
    enemy_shootable = o_hist[..., shootable_idx]  # (batch, T-1, 5, 5)
    
    # abs positions
    enemy_absx = own_x + enemy_relx  # (batch, T-1, 5, 5)
    enemy_absy = own_y + enemy_rely  # (batch, T-1, 5, 5)
    
    # seen_flag: 1 if shootable > 0
    seen_flag = (enemy_shootable > 0.5).float()  # (batch, T-1, 5, 5)

    # For each enemy, for each agent, find freshest step (largest t where seen_flag==1)
    # We'll use torch.argmax on reversed time axis to find most recent
    seen_flag_rev = th.flip(seen_flag, [1])  # (batch, T-1, 5, 5)
    has_seen = seen_flag_rev.any(dim=1)  # (batch, 5, 5)
    idx_rev = th.argmax(seen_flag_rev, dim=1)  # (batch, 5, 5)
    # idx in slice
    # slice len is L = o_hist.shape[1]
    L = o_hist.shape[1]
    idx = (L - 1) - idx_rev  # Map back to original time index within slice

    # Gather freshest sighting for each enemy-agent pair
    batch_idx = th.arange(batch, device=device)[:, None, None]
    agent_idx = th.arange(n_agents, device=device)[None, :, None]
    enemy_idx = th.arange(5, device=device)[None, None, :]
    # Shape: (batch, 5, 5)
    enemy_health_fresh = enemy_health[batch_idx, idx, agent_idx, enemy_idx]
    enemy_absx_fresh = enemy_absx[batch_idx, idx, agent_idx, enemy_idx]
    enemy_absy_fresh = enemy_absy[batch_idx, idx, agent_idx, enemy_idx]
    seen_flag_fresh = has_seen.float()

    # If never seen, fill -1 for fields, 0 for flag
    missing = (1 - has_seen.float())
    enemy_health_fresh = enemy_health_fresh * has_seen + (-1.0) * missing
    enemy_absx_fresh = enemy_absx_fresh * has_seen + (-1.0) * missing
    enemy_absy_fresh = enemy_absy_fresh * has_seen + (-1.0) * missing

    # Stack per-enemy: [seen_flag, health, abs_x, abs_y] × 5 = 20
    enemy_freshest = th.stack([
        seen_flag_fresh, enemy_health_fresh, enemy_absx_fresh, enemy_absy_fresh
    ], dim=-1)  # (batch, 5, 5, 4)
    enemy_freshest = enemy_freshest.reshape(batch, n_agents, -1)  # (batch, 5, 20)

    # --- 8. Assemble message (order: action_hist, intent, movement, target, help, enemy_freshest) ---
    msg_blocks = [
        # sender_id,       # Removed
        actions_last5,     # (batch, 5, 55)
        last_action,       # (batch, 5, 11)
        movement,          # (batch, 5, 10)
        target_focus,      # (batch, 5, 6)
        help_flag,         # (batch, 5, 1)
        enemy_freshest     # (batch, 5, 20)
    ]
    message = th.cat(msg_blocks, dim=-1)  # (batch, 5, 103)

    # --- 9. Peer-to-peer exchange (broadcast, exclude self) ---
    # For each agent, receive messages from others (not self), in sender index order
    agent_indices = th.arange(n_agents, device=device)
    other_agent_indices = th.stack([
        th.cat([agent_indices[:i], agent_indices[i+1:]])
        for i in range(n_agents)
    ])  # (5, 4)
    # message: (batch, 5, 103)
    received = th.stack(
        [message[:, other_agent_indices[i], :] for i in range(n_agents)],
        dim=1
    )  # (batch, 5, 4, 103)
    received = received.reshape(batch, n_agents, -1)  # (batch, 5, 412)

    # --- 10. Take most recent agent obs (last step) ---
    last_obs = o[:, -1, :, :]  # (batch, 5, 108)

    # --- 11. Concatenate ---
    messages_o = th.cat([last_obs, received], dim=-1)  # (batch, 5, 520)
    return messages_o
