import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    - **Sender**: Only Overseers (agents 20 and 21) send messages. Banelings (agents 0-19) remain silent.
    - **Message Content (per Overseer)**:
        - For each visible Roach (Enemy0, Enemy1):
            - Visibility flag (1/0): o[..., 4], o[..., 12]
            - Relative X position: o[..., 6], o[..., 14]
            - Relative Y position: o[..., 7], o[..., 15]
        - **Movement possibility (intent)**: o[..., 0:4] (can move North, South, East, West)
        - **Last action (one-hot)**: o[..., 170:178] (no-op, stop, move N/S/E/W, attack enemy0, attack enemy1)
        - Sender identity: 22-dimensional one-hot vector (o[..., 178:200])
    - **Message Structure (per Overseer)**:
        [Move_N, Move_S, Move_E, Move_W, 
         Last_action_noop, stop, N, S, E, W, atk0, atk1]
        - Total: 4 + 8 = 12 dimensions per Overseer.
    - **Communication Protocol**:
        - **Broadcast**: Each Overseer broadcasts to all Banelings and the other Overseer.
        - Each Baneling receives both (concatenated: 24 dims).
        - Each Overseer receives only the other's (12 dims, zero-padded).
    - **Rationale**:
        - Provides behavioral context (movement intent, last action).
    """
    return (
        "Message Design (Update):\n"
        "- Only Overseers (agents 20 and 21) send messages. Each message contains: "
        "their own movement possibility (can move N/S/E/W; 4 values) and their last action (one-hot across 8 values). "
        "Each message is thus 12 dimensions. "
    )

def communication(o):
    """
    o: torch.Tensor of shape (batch, 22, 200)
    Returns: torch.Tensor of shape (batch, 22, 200+message_dim)
    """
    # Agent indices
    overseer_ids = [20, 21]
    baneling_ids = list(range(20))  # agents 0-19

    batch_size = o.shape[0]
    device = o.device

    # Extract Overseer observations: shape (batch, 2, 200)
    overseer_obs = o[:, overseer_ids, :]

    # Movement possibility
    move_fields = overseer_obs[:, :, 0:4]  # (batch, 2, 4)

    # Last action
    last_action_fields = overseer_obs[:, :, 170:178]  # (batch, 2, 8)

    # Compose message: (batch, 2, 12)
    msg = th.cat(
        [move_fields, last_action_fields], 
        dim=-1
    )  # (batch, 2, 12)

    # Prepare empty message tensor for all agents
    message_dim = 24
    messages = th.zeros((batch_size, 22, message_dim), dtype=o.dtype, device=device)

    # For Banelings: receive both Overseers' messages (concat)
    both_msgs = msg.reshape(batch_size, -1)  # (batch, 24)
    messages[:, baneling_ids, :] = both_msgs.unsqueeze(1).expand(-1, len(baneling_ids), -1)

    # For Overseers: receive only the other Overseer's message (pad to 24)
    # For agent 20, receive agent 21's msg
    messages[:, 20, :12] = msg[:, 1, :]
    # For agent 21, receive agent 20's msg
    messages[:, 21, :12] = msg[:, 0, :]

    # Concatenate messages to observations
    messages_o = th.cat([o, messages], dim=-1)  # (batch, 22, 200+24)
    return messages_o
