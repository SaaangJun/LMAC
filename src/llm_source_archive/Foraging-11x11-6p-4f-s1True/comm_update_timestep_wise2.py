import torch as th

def message_design_instruction():
    """
    Enhanced Message Design for LBF (grid obs) with Temporal Context:

    Purpose:
    - Enable agents to infer weakly observable global states and coordinate by sharing temporal patterns of their own movement and local discoveries.

    What is Communicated (per message, per agent):
    1. **Recent Trajectory (last 2 steps):**
        - Last 2 absolute positions (x, y), normalized to [0,1] (4 dims)
        - Excludes current step which is covered by comm_update's absolute position.
        - Rationale: Sharing trajectory enables others to predict intent and avoid collisions.
    2. **Recent Food Sightings (last 2 steps):**
        - For each of the last 2 steps, a 3x3 binary mask indicating food presence (2x9=18 dims)
        - Excludes current step which is covered by comm_init's food layer.
        - Rationale: Provides timely cues about food locations.
    3. **Recent Local Agent Encounters (last 3 steps):**
        - For each of the last 3 steps, a 8-dim binary vector indicating agent presence in local grid (excluding self) (3x8=24 dims)
        - Rationale: Helps to infer possible blockages or cooperation needs.
        - (Note: comm_init does not explicitly share local agent layer, so we keep current step here).
    
    (Removed: Sender ID - duplicated)

    - **Total message per agent:** 4+18+24 = 46 dims

    Communication Protocol:
    - **Broadcast**: Each agent broadcasts its message to all others.
    - Each agent receives 5 messages (from all others, ordered by sender ID), concatenated and flattened (5x46=230 dims).
    - The 230-dim received message vector is concatenated to the agent's own 39-dim observation, yielding (batch, 6, 269).

    Explicitness & Efficiency:
    - All fields are interpretable and actionable. No information is repeated from previous communication methods.
    - All operations are vectorized for batch and agent axes.

    """
    return (
        "Each agent broadcasts a message containing:\n"
        "- Its recent trajectory: last 2 absolute (x, y) positions (t-2, t-1), normalized to [0,1] (4 dims)\n"
        "- Recent food sightings: for last 2 steps (t-2, t-1), flattened 3x3 food grid (2x9=18 dims)\n"
        "- Recent local agent encounters: for last 3 steps (t-2, t-1, t), 8-dim binary vector (3x8=24 dims)\n"
        "Total: 46 dims (Sender ID removed).\n"
        "Each agent receives 5 such messages (5x46=230 dims),\n"
        "yielding a final tensor of shape (batch, 6, 269)."
    )

def communication(o):
    """
    Args:
        o: torch.Tensor of shape (batch, T, 6, 39)
           Contains last 10 steps (T>=10), current obs at o[:, -1, :, :]
    Returns:
        messages_o: torch.Tensor of shape (batch, 6, 269)
    """
    device = o.device
    batch_size, T, n_agents, obs_dim = o.shape
    assert n_agents == 6 and obs_dim == 39

    # Indices for x, y must be specified or assumed. Let's assume:
    # Absolute position is not in the provided obs, so we reconstruct relative trajectory from local movement and/or last_action.
    # Here, we assume (for code completeness) that you can provide pos_x_idx and pos_y_idx if available.
    # Otherwise, we approximate trajectory using last_action history.
    # For LBF, typically, global position is NOT available, so we approximate trajectory as movement vector history.

    # We'll use last_action one-hot (27:32) to accumulate relative positions.
    last_actions = o[:, -4:, :, 27:32]  # (batch, 4, 6, 5) -- last 3 steps + current
    # Action mapping: 0=No-op, 1=N, 2=S, 3=W, 4=E
    # Movement deltas: [0,0], [0,-1], [0,+1], [-1,0], [+1,0]
    action_to_delta = th.tensor(
        [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], device=device, dtype=o.dtype
    )  # (5,2)
    # Convert last_actions to deltas: (batch, 4, 6, 2)
    deltas = last_actions @ action_to_delta  # (batch, 4, 6, 2)

    # Assume starting position is (0.5, 0.5) (center), accumulate deltas over last 3 steps
    # To get last 3 positions, we roll cumulative sum
    # pos_t = pos_0 + sum_{i=1}^{t} delta_i
    # We'll start from (0.5, 0.5) and accumulate deltas for t=-3, -2, -1 (relative trajectory)
    init_pos = th.tensor([0.5, 0.5], device=device, dtype=o.dtype)  # (2,)
    traj = [init_pos.expand(batch_size, n_agents, 2)]  # (batch, 6, 2)
    for step in range(1, 4):
        prev_pos = traj[-1]
        delta = deltas[:, step, :, :]  # (batch, 6, 2)
        traj.append(prev_pos + delta)
    last3_pos = th.stack(traj[1:], dim=1)  # (batch, 3, 6, 2)
    
    # NEW: Take last 2 steps (t-2, t-1), excluding current t
    # last3_pos indices: 0 (t-3), 1 (t-2), 2 (t-1). We want index 1 and 2.
    last2_pos = last3_pos[:, 1:, :, :] # (batch, 2, 6, 2)
    last2_pos = last2_pos.transpose(1,2).reshape(batch_size, n_agents, 4) # (batch, 6, 4)

    # Food layer indices: 9-17 (3x3)
    food_layers = o[:, -4:, :, 9:18]  # (batch, 4, 6, 9)
    # Last 3 steps: 1 (t-2), 2 (t-1), 3 (t).
    # NEW: Take t-2 and t-1. (Indices 1 and 2). Exclude current t (Index 3).
    last2_food = food_layers[:, 1:3, :, :]  # (batch, 2, 6, 9)
    last2_food = last2_food.transpose(1,2).reshape(batch_size, n_agents, 18)  # (batch, 6, 18)

    # Agent layer indices: 0-8 (3x3)
    agent_layers = o[:, -4:, :, 0:9]  # (batch, 4, 6, 9)
    # For each of last 3 steps, construct a 5-dim binary vector: which other agents were seen
    # Agent IDs: one-hot in 33-38
    # agent_layers indices for t-2, t-1, t are 1, 2, 3.
    
    # Remove center cell (self) for each agent_layer
    non_center_idx = [i for i in range(9) if i != 4]
    last3_agent = agent_layers[:, 1:, :, non_center_idx]  # (batch, 3, 6, 8)
    last3_agent = last3_agent.transpose(1,2).reshape(batch_size, n_agents, 24)  # (batch, 6, 24)

    # Compose message: last2_pos (4), last2_food (18), last3_agent (24) -> 46 dims
    msg = th.cat([last2_pos, last2_food, last3_agent], dim=-1)  # (batch, 6, 46)
    msg_dim = msg.shape[-1]

    # Build all-to-all message matrix (excluding self)
    agent_indices = th.arange(n_agents, device=device)
    sender_indices = th.stack([
        th.cat([agent_indices[:i], agent_indices[i+1:]], dim=0)
        for i in range(n_agents)
    ], dim=0)  # (6, 5)

    batch_ar = th.arange(batch_size, device=device)[:, None, None]
    recv_ar = th.arange(n_agents, device=device)[None, :, None]
    send_ar = sender_indices[None, :, :]  # (1, 6, 5)

    batch_idx = batch_ar.expand(batch_size, n_agents, 5)
    recv_idx = recv_ar.expand(batch_size, n_agents, 5)
    send_idx = send_ar.expand(batch_size, n_agents, 5)

    messages_received = msg[batch_idx, send_idx, :]  # (batch, 6, 5, msg_dim)
    messages_received_flat = messages_received.reshape(batch_size, n_agents, -1)  # (batch, 6, 5*msg_dim)

    # Current obs (latest step)
    o_now = o[:, -1, :, :]  # (batch, 6, 39)

    messages_o = th.cat([o_now, messages_received_flat], dim=-1)  # (batch, 6, 39+5*msg_dim)

    return messages_o
