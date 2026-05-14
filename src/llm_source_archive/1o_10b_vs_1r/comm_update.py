import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    **Sender**: Only the Overseer (agent 10) broadcasts a message.
    **Receivers**: All Banelings (agents 0–9) receive the message. The Overseer itself receives an all-zero message.

    **Message Content** (per batch):
    - [0] Roach relative X position (Overseer’s o[:,6])
    - [1] Roach relative Y position (Overseer’s o[:,7])
    - [2] Roach is_visible (Overseer’s o[:,4])
    - [3:10] Overseer's last action one-hot (fields 85–91, length 7): [No-op, Stop, Move N, Move S, Move E, Move W, Attack]
    - [10:30] For each Baneling (agents 0–9), Overseer’s relative X and Y to that Baneling as observed by Overseer:
        - For Baneling i: If visible, use fields 14+7*i (X) and 15+7*i (Y) from Overseer’s observation; else, zeros.
    - [30:41] Sender one-hot (length 11, index 10=1)

    **Why**: This message allows each Baneling to:
      - Know the Roach's relative position to the Overseer (as before)
      - Know the Overseer's recent movement intention (to predict movement)
      - If visible to Overseer, know the Overseer's position relative to self (enabling Baneling to reconstruct Roach's location in its own frame)
      - The sender field grounds the message.

    **Protocol**: Overseer broadcasts this message to all Banelings. Overseer receives all-zeros.

    **Shape**: Each message is 41-dimensional. Final enhanced observation tensor is (batch, 11, 144).

    **Redundancy Minimization**: Only new, actionable information is included; fields already present in previous methods are not repeated.
    """
    return (
        "Message Structure: [Overseer_last_action_one_hot (7), "
        "For each Baneling (10): Overseer's relative_X_to_Baneling (1), Overseer's relative_Y_to_Baneling (1)] = 27 dimensions. "
        "Only Overseer (agent 10) sends a message, broadcasting to all Banelings (agents 0–9). "
        "The message contains the Overseer's recent movement intent and, if visible, Overseer's relative position to each Baneling. "
    )

def communication(o):
    # o: (batch, 11, 103)
    device = o.device
    batch_size = o.shape[0]
    n_agents = o.shape[1]
    obs_dim = o.shape[2]
    message_dim = 27

    # 1. Get Overseer obs (agent 10)
    overseer_obs = o[:, 10, :]  # (batch, 103)

    # 2. Overseer last action one-hot (fields 85–91, length 7)
    overseer_last_action = overseer_obs[:, 85:92]  # (batch,7)

    # 3. Overseer’s relative X/Y to each Baneling, from its own obs
    # For Baneling i (i=0..9): fields 14+7*i (X), 15+7*i (Y)
    rel_xy_to_banelings = []
    for i in range(10):
        rel_x = overseer_obs[:, 14 + 7 * i].unsqueeze(1)  # (batch,1)
        rel_y = overseer_obs[:, 15 + 7 * i].unsqueeze(1)  # (batch,1)
        rel_xy_to_banelings.append(rel_x)
        rel_xy_to_banelings.append(rel_y)
    # (batch, 20)
    rel_xy_to_banelings = th.cat(rel_xy_to_banelings, dim=1)

    # 4. Assemble Overseer message: (batch, 27)
    overseer_message = th.cat([
        overseer_last_action,
        rel_xy_to_banelings
    ], dim=1)  # (batch, 27)

    # 5. Broadcast to all agents: Banelings get message, Overseer gets zero
    messages = th.zeros(batch_size, n_agents, message_dim, device=device)
    # For Banelings (0–9): give them the Overseer's message
    messages[:, 0:10, :] = overseer_message.unsqueeze(1).expand(-1, 10, -1)
    # For Overseer (10): zeros

    # 6. Concatenate to original obs
    messages_o = th.cat([o, messages], dim=2)  # (batch, 11, 130)
    return messages_o
