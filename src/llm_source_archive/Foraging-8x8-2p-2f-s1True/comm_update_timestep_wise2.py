import torch as th

def message_design_instruction():
    """
    Enhanced Message Design Instruction for LBF (grid obs) Task
    -----------------------------------------------------------
    **Objective**:  
    Address the prior protocol's shortcoming—persistent unpredictability of global/remote state—by enabling agents to relay, propagate, and summarize temporally-extended, non-local, and behaviorally relevant information to their peers, minimizing redundancy and maximizing informativeness for coordination.

    **Key Improvements Over Previous Protocol**:
    - **Temporal Relaying**: Agents now communicate not only their current local state, but also a compressed history of food and agent presence observed over the last 10 timesteps, propagating discoveries beyond the immediate local patch.
    - **Remote Event Reporting**: If an agent has observed food or another agent outside its current grid in the last 10 steps (but not in the present), it explicitly reports the event and its relative position, timestamped as "steps since seen".
    - **Movement Intent Summary**: Agents summarize their movement pattern over the last 10 steps as a 6-bin histogram of actions (from o[...,27:33]), allowing peers to infer likely intent and path without full action history, reducing bandwidth.
    - **Sender Identity**: Explicit one-hot vector grounding the message to a sender.

    **Message Content & Structure**:
    For each agent, **at the current timestep**, construct a message to be sent to the other agent containing:
    1. **Recent Action Histogram (6D)**: For the last 10 timesteps (including current), sum of each action (No-op, N, S, W, E, Pick-up) taken by the sender (from o[...,27:33] across last 10 steps). This is not a full trajectory, but a compact intent summary.
    2. **Food/Agent Event Buffer (max 3 events × 4D)**: For up to 3 most recent unique food or agent sightings outside sender's current grid (in last 10 steps), each event is encoded as:
        - (a) Type (1D): 0=food, 1=agent
        - (b) Relative position (2D): (dx, dy) relative to sender's current center (row,col offset in [-2,2])
        - (c) Steps since seen (1D): Integer in [0,9], 0 if seen this step, higher for earlier
      If fewer than 3 such events, pad events with zeros.
    3. **Timestamp (1D)**: Current timestep mod 10, for temporal alignment.

    **Total Message Dimension**:  
    - Action histogram: 6  
    - Event buffer: 3 × 4 = 12  
    - Timestamp: 1  
    = **19**

    **Communication Pattern**: Peer-to-peer, each agent receives the other's message in the same scenario.

    **Why this information?**
    - **Event Buffer**: Propagates discoveries of food/agents beyond local view, allowing indirect mapping and memory over time.
    - **Action Histogram**: Encodes behavior and intent, aiding in collision avoidance and teamwork.
    - **Timestamp**: For temporal reasoning.

    **Exclusions**:  
    - No redundant transmission of last action (now captured in histogram).
    - No re-sending of food/agent data from peer's current grid.
    - No raw peer message relaying—only summarized, novel, or propagated info.

    **Message Format**: [action_hist (6), event_buffer (12), timestamp (1)] = 19D

    **Summary**: This protocol enables agents to build a richer, temporally-aligned map of the environment and each other's intent, helping to fill persistent knowledge gaps and improve global predictability.
    """
    return (
        "Enhanced Message Design for LBF Task:\n"
        "Each agent sends a peer-to-peer message with the following fields:\n"
        "1. 6-bin histogram of its own actions over the last 10 timesteps (No-op, N, S, W, E, Pick-up)\n"
        "2. Up to 3 most recent unique food/agent sightings outside its current grid, each as:\n"
        "   [type (0=food, 1=agent), dx, dy (relative pos), steps_since_seen], padded to 3 events\n"
        "3. Current timestamp mod 10 (1D)\n"
        "Total: 19D/message, peer-to-peer\n"
        "This message relays temporally-propagated, non-redundant, and intent-relevant information to enhance global state predictability and coordination."
    )

def communication(o):
    """
    Enhanced peer-to-peer message exchange for LBF (grid obs) MARL scenario.
    Each agent receives a message from its peer, with the following structure (see message_design_instruction).

    Args:
        o: torch.Tensor, shape (batch, T, 2, 39) -- last 10 steps up to current

    Returns:
        messages_o: torch.Tensor, shape (batch, 2, 39+19)
    """
    # o: (batch, T, 2, 39), T >= 10 (last 10 steps up to current)
    device = o.device
    dtype = o.dtype
    batch, T, n_agents, obs_dim = o.shape
    assert n_agents == 2

    # Only use the last 10 steps for temporal features
    steps = min(T, 10)
    o_recent = o[:, -steps:, :, :]  # (batch, steps, 2, 39)

    # 1. Action histogram over last 10 steps (6D)
    # last_action: (batch, steps, 2, 6)
    last_action_hist = o_recent[..., 27:33].sum(dim=1)  # (batch, 2, 6)

    # 2. Event buffer: up to 3 most recent unique food/agent sightings outside current grid (per agent)
    # For each agent:
    #   - For each step in last 10, get food_layer[9:18] and agent_layer[0:9]
    #   - For each cell, if observed (==1) and cell is OUTSIDE center (i.e., not [1,1]), and is not currently in current grid
    #   - For each such event, record: type (0=food,1=agent), dx, dy (relative to center), steps_since_seen
    #   - Keep up to 3 most recent unique events

    # Precompute relative positions for 3x3 grid
    rel_pos = th.tensor([(i-1, j-1) for i in range(3) for j in range(3)], device=device, dtype=dtype)  # (9, 2)
    # Exclude center cell (1,1) -> index 4
    non_center_idx = [i for i in range(9) if i != 4]
    rel_pos_nc = rel_pos[non_center_idx]  # (8,2)

    # Helper: For each agent, batch, for each step in last 10:
    #   - food_obs: (batch, steps, 2, 9)
    #   - agent_obs: (batch, steps, 2, 9)
    food_obs = o_recent[..., 9:18]  # (batch, steps, 2, 9)
    agent_obs = o_recent[..., 0:9]  # (batch, steps, 2, 9)

    # Current step food/agent obs (for masking)
    food_now = o_recent[:, -1, :, 9:18]  # (batch, 2, 9)
    agent_now = o_recent[:, -1, :, 0:9]  # (batch, 2, 9)

    # For each agent, we want to find unique events (food/agent) outside current grid
    event_buffer = []
    for agent in range(n_agents):
        # For food and agent, collect events:
        events = []
        for typ, obs, obs_now in [(0, food_obs, food_now), (1, agent_obs, agent_now)]:
            # obs: (batch, steps, 9)
            # obs_now: (batch, 9)
            obs_nc = obs[..., agent, :][:, :, non_center_idx]  # (batch, steps, 8)
            obs_now_nc = obs_now[:, agent, non_center_idx]     # (batch, 8)
            # For each cell, for each batch:
            # Find first (most recent) timestep (from latest to earliest) where obs==1 and NOT in obs_now==1
            # steps_since_seen = steps - 1 - t
            # We'll scan from most recent to oldest
            # To vectorize: (batch, 8, steps)
            obs_nc_flip = obs_nc.flip(1)  # Now (batch, steps, 8), most recent first
            obs_nc_flip = obs_nc_flip.transpose(1,2)  # (batch, 8, steps)
            # Mask out if currently observed
            mask_not_now = (obs_now_nc == 0).unsqueeze(-1)  # (batch, 8, 1)
            obs_nc_flip = obs_nc_flip * mask_not_now  # (batch, 8, steps)
            # For each cell, find first step where obs==1
            seen = obs_nc_flip > 0.5
            # Find steps_since_seen (0 if seen at current step, 1 for last step, ...)
            first_seen = seen.float().argmax(dim=-1)  # (batch, 8)
            # But if never seen, all zeros, so need to mask out those where seen.sum==0
            never_seen = (seen.sum(dim=-1) == 0)  # (batch, 8)
            # Only keep those where at least one seen
            for cell in range(8):
                # Only keep if seen at all (and not currently observed)
                mask = ~never_seen[:, cell]
                if mask.any():
                    # For those batches, gather info
                    b_idx = mask.nonzero(as_tuple=False).squeeze(-1)
                    steps_since = first_seen[b_idx, cell].cpu()
                    # rel pos
                    dx, dy = int(rel_pos_nc[cell][0].item()), int(rel_pos_nc[cell][1].item())
                    # For each batch where event occurred
                    for bi, ss in zip(b_idx.tolist(), steps_since.tolist()):
                        events.append((bi, typ, dx, dy, int(ss)))
        # For each batch, keep up to 3 most recent events (lowest steps_since_seen)
        # For each batch, collect events and sort by steps_since
        batch_events = [[] for _ in range(batch)]
        for ev in events:
            bi, typ, dx, dy, ss = ev
            batch_events[bi].append((ss, typ, dx, dy))
        # For each batch, sort and keep up to 3
        eb = th.zeros((batch, 3, 4), device=device, dtype=dtype)
        for bi in range(batch):
            evs = sorted(batch_events[bi], key=lambda x: x[0])[:3]
            for ei, (ss, typ, dx, dy) in enumerate(evs):
                eb[bi, ei, 0] = typ
                eb[bi, ei, 1] = dx
                eb[bi, ei, 2] = dy
                eb[bi, ei, 3] = ss
        event_buffer.append(eb)  # (batch, 3, 4)
    # Stack to (batch, 2, 3, 4) then flatten last two dims
    event_buffer = th.stack(event_buffer, dim=1).reshape(batch, n_agents, 12)

    # 3. Timestamp mod 10 (1D)
    timestamp = ((T-1) % 10) * th.ones((batch, n_agents, 1), device=device, dtype=dtype)  # (batch, 2, 1)

    # Concatenate all fields: [action_hist (6), event_buffer (12), timestamp (1)] = 19
    message = th.cat([
        last_action_hist,                     # (batch, 2, 6)
        event_buffer,                         # (batch, 2, 12)
        timestamp                            # (batch, 2, 1)
    ], dim=-1)                                # (batch, 2, 19)

    # Transpose messages so each agent receives from its peer
    peer_message = message.flip(1)  # (batch, 2, 19)

    # Each agent's own latest observation (batch, 2, 39)
    o_now = o[:, -1, :, :]  # (batch, 2, 39)

    # Concatenate received message to each agent's own observation
    messages_o = th.cat([o_now, peer_message], dim=-1)  # (batch, 2, 39+19)

    return messages_o
