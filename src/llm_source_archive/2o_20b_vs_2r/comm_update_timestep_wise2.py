import torch as th

def message_design_instruction():
    """
    Message Design Instruction (Temporal Augmented Protocol):

    - **Sender**: Only Overseers (agents 20 and 21) send messages. Banelings (0–19) remain silent.
    - **Message Content (per Overseer)**:
        - For each of the past 10 timesteps (including current):
            - **Own last action** (one-hot, 8 dims): o[..., 170:178]
            - **Roach 0**:
                - Visibility flag (1 dim): o[..., 4]
                - Relative X (1 dim): o[..., 6]
                - Relative Y (1 dim): o[..., 7]
            - **Roach 1**:
                - Visibility flag (1 dim): o[..., 12]
                - Relative X (1 dim): o[..., 14]
                - Relative Y (1 dim): o[..., 15]
        - **Initial spawn anchor**: Own relative position at t=0 (o[..., 0, 162], o[..., 0, 163]) (if available; else zeros).
        - **Sender identity**: 22-dimensional one-hot vector (o[..., -1, 178:200])
    - **Message Structure (per Overseer)**:
        - For each of 9 past steps: [last_action(8), Roach0(3), Roach1(3)] = 14 dims x 9 = 126
        - Initial anchor: 2 dims
        - **Total: 128 dims per Overseer**
    - **Communication Protocol**:
        - **Broadcast**: Each Overseer broadcasts to all Banelings and the other Overseer.
        - Each Baneling receives both (concatenated: 256 dims).
        - Each Overseer receives only the other's (128 dims, zero-padded).
    - **Rationale**:
        - Current step info (Roach, Action, SenderID) is available in comm_init/comm_update.
        - This protocol provides *past* history (t-9 to t-1) to augment the current context.
    """
    return (
        "Temporal Augmented Communication Protocol (Update):\n"
        "- Only Overseers (agents 20 and 21) send messages. Each message contains, for each of the *past 9 steps* (excluding current): own last action (one-hot, 8 dims), Roach 0 (visibility, relX, relY, 3 dims), Roach 1 (visibility, relX, relY, 3 dims), "
        "for a total of 14 dims per step (126 total). In addition, the message includes the Overseer's initial spawn anchor (relative position at t=0, 2 dims). "
        "Each message is thus 128 dims. "

    )

def communication(o):
    """
    o: torch.Tensor of shape (batch, T, 22, 200)
    Returns: torch.Tensor of shape (batch, 22, 200+message_dim)
    """
    # Agent indices
    overseer_ids = [20, 21]
    baneling_ids = list(range(20))  # agents 0-19

    batch_size, T, n_agents, obs_dim = o.shape
    device = o.device

    # Get last 10 timesteps (assume T >= 10, else pad with zeros at front)
    window = 10
    if T < window:
        # Pad at front
        pad = window - T
        pad_shape = (batch_size, pad, n_agents, obs_dim)
        pad_tensor = th.zeros(pad_shape, dtype=o.dtype, device=device)
        o_pad = th.cat([pad_tensor, o], dim=1)
        o_last10 = o_pad[:, -window:, :, :]
    else:
        o_last10 = o[:, -window:, :, :]  # (batch, 10, 22, 200)

    # Extract Overseer observations: shape (batch, 10, 2, 200)
    overseer_obs = o_last10[:, :, overseer_ids, :]  # (batch, 10, 2, 200)

    # For each timestep, get:
    # - Last action (8 dims): 170:178
    # - Roach0: [4,6,7] (visible, relX, relY)
    # - Roach1: [12,14,15] (visible, relX, relY)
    # --> For each Overseer: (batch, 10, 14)
    last_action = overseer_obs[..., 170:178]  # (batch, 10, 2, 8)
    roach0 = overseer_obs[..., [4,6,7]]      # (batch, 10, 2, 3)
    roach1 = overseer_obs[..., [12,14,15]]   # (batch, 10, 2, 3)
    # Concatenate along last dim: (8+3+3=14)
    temporal_fields = th.cat([last_action, roach0, roach1], dim=-1)  # (batch, 10, 2, 14)
    
    # Use only PAST 9 steps
    # (batch, 9, 2, 14)
    temporal_fields_past = temporal_fields[:, :-1, :, :]

    # Reshape to (batch, 2, 9*14 = 126)
    temporal_fields_flat = temporal_fields_past.permute(0,2,1,3).reshape(batch_size, 2, 9*14)

    # Initial anchor: own relative position at t=0 (fields 162,163)
    # Use o[:, 0, overseer_ids, 162:164]
    initial_anchor = o[:, 0, overseer_ids, 162:164]  # (batch, 2, 2)

    # Compose message: (batch, 2, 128)
    msg = th.cat([temporal_fields_flat, initial_anchor], dim=-1)  # (batch, 2, 128)

    # Prepare empty message tensor for all agents
    message_dim = 256
    messages = th.zeros((batch_size, n_agents, message_dim), dtype=o.dtype, device=device)

    # For Banelings: receive both Overseers' messages (concat: 256 dims)
    both_msgs = msg.reshape(batch_size, -1)  # (batch, 256)
    messages[:, baneling_ids, :] = both_msgs.unsqueeze(1).expand(-1, len(baneling_ids), -1)

    # For Overseers: receive only the other Overseer's message (128 dims, zero-padded to 256)
    # For agent 20, receive agent 21's msg
    messages[:, 20, :128] = msg[:, 1, :]
    # For agent 21, receive agent 20's msg
    messages[:, 21, :128] = msg[:, 0, :]

    # Current observation for each agent at current step: o[:, -1, :, :] -> (batch, 22, 200)
    o_now = o[:, -1, :, :]  # (batch, 22, 200)

    # Concatenate messages to observations
    messages_o = th.cat([o_now, messages], dim=-1)  # (batch, 22, 200+256)
    return messages_o
