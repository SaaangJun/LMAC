import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    - **Sender**: Only the Overseer (agent at index 10) sends messages.
    - **Receivers**: All Banelings (agent indices 0~9) receive the message. The Overseer does not receive any messages.
    - **Message Content**:
        - `relative_x`: Overseer's observed relative X position of the enemy Roach (`o[..., 6]`).
        - `relative_y`: Overseer's observed relative Y position of the enemy Roach (`o[..., 7]`).
        - `is_visible`: Overseer's observed "is enemy visible" flag (`o[..., 4]`).
        - `sender_id`: One-hot vector of length 11, with the last position (index 10) set to 1 (identifying the Overseer as sender).
    - **Why**: Only the Overseer can observe the Roach; Banelings need this precise and actionable info to coordinate and attack. Including sender identity ensures explicit grounding and prevents ambiguity.
    - **Compactness**: Only essential info is sent; Banelings do not send messages, and Overseer receives none.
    - **Protocol**: Broadcast from Overseer to all Banelings, with explicit sender field.
    - **Shape**: Each agent's message is a 14-dimensional vector; for the Overseer (agent 10), this vector is all zeros.

    """
    return (
        "Message Structure: [relative_x, relative_y, is_visible, sender_one_hot (length 11)]. "
        "Only the Overseer (agent 10) sends, broadcasting the Roach's relative position and visibility. "
        "Sender identity is a one-hot of length 11 (index 10=1 for Overseer), grounding the message. "
        "Banelings 0-9 receive this message; the Overseer receives an all-zero message. "
        "This ensures Banelings get the minimum sufficient information for the task."
    )

def communication(o):
    """
    o: (batch, 11, 103)
    Returns: (batch, 11, 117)
    """
    # Device safety
    device = o.device
    batch_size = o.shape[0]
    n_agents = o.shape[1]
    obs_dim = o.shape[2]
    message_dim = 14

    # 1. Get Overseer's observation (agent index 10)
    overseer_obs = o[:, 10, :]  # (batch, 103)

    # 2. Extract required info from Overseer
    # These are all (batch,)
    relative_x = overseer_obs[:, 6].unsqueeze(1)      # (batch, 1)
    relative_y = overseer_obs[:, 7].unsqueeze(1)      # (batch, 1)
    is_visible = overseer_obs[:, 4].unsqueeze(1)      # (batch, 1)

    # 3. Create sender_id one-hot (always index 10)
    sender_id = th.zeros(batch_size, 11, device=device)
    sender_id[:, 10] = 1.0  # (batch, 11)

    # 4. Assemble message: (batch, message_dim)
    overseer_message = th.cat([relative_x, relative_y, is_visible, sender_id], dim=1)  # (batch, 14)

    # 5. Broadcast message to all agents
    # For agents 0-9: receive Overseer's message. For agent 10: receive zeros.
    messages = th.zeros(batch_size, n_agents, message_dim, device=device)

    # For Banelings (indices 0~9), set their message to Overseer's message
    messages[:, 0:10, :] = overseer_message.unsqueeze(1).expand(-1, 10, -1)
    # For Overseer (index 10), message remains zeros

    # 6. Concatenate messages to observations
    messages_o = th.cat([o, messages], dim=2)  # (batch, 11, 117)
    return messages_o
