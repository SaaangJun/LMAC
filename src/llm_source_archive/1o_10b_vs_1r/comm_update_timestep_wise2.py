import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Sender: Only the Overseer (agent 10) broadcasts a message.
    Receivers: All Banelings (agents 0–9) receive the message. Overseer receives all-zeros.

    Message Content (per batch):
    - [0] Overseer canonical X position (from spawn or reference, here assumed to be (0, 0) as anchor, or from own observation if absolute is encoded)
    - [1] Overseer canonical Y position (from spawn or reference)
    - [2] Roach canonical X position (Overseer’s observation: relative X to Roach plus Overseer's own canonical X)
    - [3] Roach canonical Y position (Overseer’s observation: relative Y to Roach plus Overseer's own canonical Y)
    - [4] Roach is_visible (Overseer’s o[:,4])
    - [5] Roach last observed at step idx (normalized 0–1, 0=current, 1=oldest)
    - [6:16] Baneling visibility flags (for each Baneling 0–9: 1 if Overseer sees it now, else 0)
    - [16:23] Overseer last action one-hot (fields 85–91, length 7): [No-op, Stop, Move N, Move S, Move E, Move W, Attack]
    - [23:34] Sender one-hot (length 11, index 10=1)

    Why:
    - Banelings can always reconstruct the Overseer’s "canonical" (anchored) position.
    - Each Baneling can compute Roach's "canonical" position using Overseer's anchor.
    - Roach last observed timestamp enables Banelings to weigh information freshness.
    - Visibility flags help Banelings know when Overseer can see them.
    - Overseer last action helps infer intent.
    - Sender identity provides explicit grounding.

    Protocol:
    - Overseer broadcasts this message to all Banelings each step.
    - Overseer receives all-zeros.
    - Message is 34-dimensional.
    - Final enhanced observation tensor is (batch, 11, 137) = (batch, 11, 103+34).
    - No redundant or easily inferable info is transmitted.
    """
    return (
        "Message Structure: [Overseer_canonical_X (1), Overseer_canonical_Y (1), "
        "Roach_canonical_X (1), Roach_canonical_Y (1), "
        "Roach_last_observed_step_idx (1), Baneling_visibility_flags (10)] = 15 dimensions. "
        "Only Overseer (agent 10) sends a message, broadcasting to all Banelings (agents 0–9). "
        "The message contains the Overseer's canonical position, Roach's canonical position, Roach last seen step, and baneling visibility flags. "
    )

def communication(o):
    # o: (batch, T, 11, 103)
    # Use most recent step (last in T) for communication
    device = o.device
    batch_size, T, n_agents, obs_dim = o.shape
    message_dim = 15

    # Use most recent timestep (t = T-1) for messaging
    t = T - 1
    obs_now = o[:, t, :, :]  # (batch, 11, 103)
    overseer_obs = obs_now[:, 10, :]  # (batch, 103)

    # 1. Overseer canonical position (we assume spawn at (0,0) as canonical anchor; if absolute available, replace here)
    overseer_canonical_x = th.zeros(batch_size, 1, device=device)  # (batch, 1)
    overseer_canonical_y = th.zeros(batch_size, 1, device=device)  # (batch, 1)

    # 2. Roach canonical position (Overseer’s relative-to-Roach + Overseer’s canonical)
    roach_rel_x = overseer_obs[:, 6:7]  # (batch,1)
    roach_rel_y = overseer_obs[:, 7:8]  # (batch,1)
    roach_canonical_x = overseer_canonical_x + roach_rel_x  # (batch,1)
    roach_canonical_y = overseer_canonical_y + roach_rel_y  # (batch,1)

    # 3. Roach last observed at step idx (normalized 0–1, 0=current, 1=oldest)
    # Find latest timestep in last 10 where Overseer saw Roach (o[:, t-9:t+1, 10, 4])
    last10_roach_vis = o[:, max(0, t-9):t+1, 10, 4]  # (batch, <=10)
    # For each batch, find last idx where visible==1 (0=oldest, ..., 9=newest)
    # Pad to length 10 if needed
    pad_len = 10 - last10_roach_vis.shape[1]
    if pad_len > 0:
        last10_roach_vis = th.cat([th.zeros(batch_size, pad_len, device=device), last10_roach_vis], dim=1)
    # Indices: 0=oldest, 9=newest
    vis_mask = last10_roach_vis > 0.5
    # For each batch: want the *last* index where True (most recent)
    # If never seen, set to 9 (oldest=0, newest=9). We'll normalize to [0,1]
    vis_idx = vis_mask.float() * th.arange(10, device=device).view(1, -1)  # (batch, 10)
    last_seen_idx = vis_idx.max(dim=1)[0]  # (batch,)
    last_seen_idx = last_seen_idx.unsqueeze(1)  # (batch,1)
    # Normalize to [0,1]: 0 = just now, 1 = oldest
    last_seen_idx_norm = 1.0 - last_seen_idx / 9.0  # (batch,1), 1=just now, 0=oldest

    # 4. Baneling visibility flags (for each Baneling 0–9: field 12+7*i)
    baneling_flags = []
    for i in range(10):
        baneling_flags.append(overseer_obs[:, 12 + 7 * i].unsqueeze(1))  # (batch,1)
    baneling_visibility = th.cat(baneling_flags, dim=1)  # (batch, 10)

    # 5. Assemble message: (batch, 15)
    overseer_message = th.cat([
        overseer_canonical_x, overseer_canonical_y,
        roach_canonical_x, roach_canonical_y,
        last_seen_idx_norm,
        baneling_visibility
    ], dim=1)  # (batch, 15)

    # 6. Broadcast: Banelings 0–9 get message, Overseer gets zeros
    messages = th.zeros(batch_size, n_agents, message_dim, device=device)
    messages[:, 0:10, :] = overseer_message.unsqueeze(1).expand(-1, 10, -1)
    # Overseer (10) gets zeros

    # 7. Concatenate with obs
    enhanced_obs = th.cat([obs_now, messages], dim=2)  # (batch, 11, 118)
    return enhanced_obs
