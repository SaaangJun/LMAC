import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    - **Sender**: Only Overseers (agents 20 and 21) send messages. Banelings (agents 0-19) and other agents remain silent.
    - **Message Content** (per Overseer):
        - For each visible Roach (Enemy0, Enemy1):
            - Visibility flag (1/0): o[..., 4] and o[..., 12]
            - Relative X position: o[..., 6] and o[..., 14]
            - Relative Y position: o[..., 7] and o[..., 15]
        - Sender identity: 22-dimensional one-hot vector (o[..., 178:200])
    - **Message Structure** (per Overseer): [E0_visible, E0_relX, E0_relY, E1_visible, E1_relX, E1_relY, SenderID (22 dims)]
        - Total: 6 + 22 = 28 dimensions per Overseer.
    - **Communication Protocol**:
        - **Broadcast**: Each Overseer broadcasts its message to all Banelings and to the other Overseer (but not to itself).
        - Each Baneling receives both Overseers' messages (concatenated: 56 dims).
        - Each Overseer receives only the other Overseer's message (28 dims).
        - All messages include sender identity for explicit grounding.
    - **Rationale**:
        - Banelings cannot observe Roaches. By receiving Overseer messages, they acquire actionable, uniquely held information (Roach positions), enabling coordinated convergence and attack.
        - Messages are minimal, sufficient, and non-redundant, containing only the essential task-relevant data.
        - Sender identity ensures messages are interpretable and properly attributed.
    """
    return (
        "Message Design:\n"
        "- Only Overseers (agents 20 and 21) send messages. Each Overseer broadcasts a message containing, for each visible Roach (Enemy0, Enemy1): visibility flag (1/0), relative X position, and relative Y position (6 values total), plus a 22-dimensional one-hot sender identity. "
        "Each message is thus 28 dimensions. Every Baneling receives both Overseers' messages (concatenated: 56 dims), while each Overseer receives the other Overseer's message (28 dims). "
        "Sender identity is always included for explicit grounding. This ensures Banelings receive uniquely held, actionable information needed for task success, while avoiding redundancy."
    )


def communication(o):
    """
    o: torch.Tensor of shape (batch, 22, 200)
    Returns: torch.Tensor of shape (batch, 22, 200+message_dim)
    """
    # Assume agents 20 and 21 are Overseers
    batch_size = o.shape[0]
    device = o.device

    # Indices for message content
    overseer_ids = [20, 21]
    baneling_ids = list(range(20))  # agents 0-19

    # Extract Overseer observations: shape (batch, 2, 200)
    overseer_obs = o[:, overseer_ids, :]

    # For each Overseer, extract message fields
    # Enemy0: visible (4), relX (6), relY (7)
    enemy0_fields = overseer_obs[:, :, [4, 6, 7]]  # (batch, 2, 3)
    # Enemy1: visible (12), relX (14), relY (15)
    enemy1_fields = overseer_obs[:, :, [12, 14, 15]]  # (batch, 2, 3)
    # SenderID: one-hot (178:200), 22 dims
    sender_id_fields = overseer_obs[:, :, 178:200]  # (batch, 2, 22)

    # Stack message: (batch, 2, 28)
    msg = th.cat([enemy0_fields, enemy1_fields, sender_id_fields], dim=-1)

    # Prepare empty message tensors for all agents
    # For Banelings: receive both Overseers' messages (56 dims)
    # For Overseers: receive only the other Overseer's message (28 dims, pad to 56 for consistency)
    message_dim = 56
    messages = th.zeros((batch_size, 22, message_dim), dtype=o.dtype, device=device)

    # For Banelings (agents 0-19): receive both Overseer messages
    both_msgs = msg.reshape(batch_size, -1)  # (batch, 56)
    messages[:, baneling_ids, :] = both_msgs.unsqueeze(1).expand(-1, len(baneling_ids), -1)

    # For Overseer 20 (index 20): receive Overseer 21's message (msg[:, 1, :])
    messages[:, 20, :28] = msg[:, 1, :]
    # For Overseer 21 (index 21): receive Overseer 20's message (msg[:, 0, :])
    messages[:, 21, :28] = msg[:, 0, :]

    # Concatenate messages to observations
    messages_o = th.cat([o, messages], dim=-1)  # (batch, 22, 200+56)
    return messages_o
