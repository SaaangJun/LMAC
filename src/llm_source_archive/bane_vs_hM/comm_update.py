import torch as th

def message_design_instruction():
    """
    Enhanced Communication Protocol Description:

    To address the unevenness in prediction accuracy and inconsistent inference observed in the previous protocol,
    this communication protocol extension focuses on explicitly sharing the sender's observational reliability
    and local movement capabilities, which are critical for synchronizing shared knowledge and trust.

    Key communicated information per agent (additional to previous protocol) includes:

    1. Visibility Flags (3 dims boolean):
       - Enemy0 (Hydralisk) visible (o[...,4])
       - Ally1 visible (o[...,16])
       - Ally2 visible (o[...,22])
       These flags explicitly indicate which key entities the sender currently observes, allowing receivers to
       assess the reliability of the positional and health info shared previously.

    2. Movement Possibility Flags (4 dims boolean):
       - Can move North (o[...,0])
       - Can move South (o[...,1])
       - Can move East  (o[...,2])
       - Can move West  (o[...,3])
       Sharing movement possibilities provides context on the sender's local environment constraints,
       supporting better inference of positional dynamics and intentions.

    3. Timestamp Counter (1 dim continuous normalized):
       - A timestep-relative counter normalized to [0,1] to help receivers detect staleness and align temporal info.
       - Since the environment timestep is not provided, we approximate this with the modulo of last action No-op counts
         or a placeholder zero. (Implementation here uses zeros as placeholder; can be replaced with real timestep input.)

    4. Last Action (7 dims one-hot):
       - The sender's previous action (No-op, Stop, Move N/S/E/W, Attack) at indices [31..37].


    Communication type: broadcast.

    Message dimension: 3 (visibility) + 4 (move flags) + 1 (timestamp) + 7 (last action) = 15 dims.

    This message complements the previous message by providing explicit reliability and context cues,
    reducing asymmetric inference and enabling consistent, synchronized understanding of critical state dimensions.

    The combined protocol (previous + this) supports precise, reliable coordination under partial observability
    with minimal communication overhead and maximal task relevance.
    """
    return message_design_instruction.__doc__


def communication(o):
    """
    Input:
        o: torch.Tensor of shape (batch_size=32, n_agents=3, obs_dim=42)
           Observation tensor per agent per batch.

    Output:
        messages_o: torch.Tensor of shape (32, 3, 42 + 15)
           Enhanced observation tensor with integrated new task-specific message appended.

    Message design (extension):
        For each agent, construct a message containing ONLY the new fields that were NOT included
        in the previous protocol (which covered relative enemy pos, own health, and sender id), plus last action:

        - Visibility flags for key entities (enemy0, ally1, ally2): indices [4,16,22]
        - Movement possibility flags (can move N/S/E/W): indices [0,1,2,3]
        - Timestamp placeholder: zeros tensor (shape: batch x agents x 1)
        - Last action (one-hot 7 dims): indices [31..37]

    Communication type: broadcast.

    Implementation notes:
        - Vectorized operations without explicit for-loops over batch or agents.
        - Uses same device as input tensor.
    """

    batch_size, n_agents, obs_dim = o.shape
    device = o.device

    # Extract visibility flags
    vis_enemy0 = o[:, :, 4:5]   # (batch, agents, 1)
    vis_ally1  = o[:, :, 16:17] # (batch, agents, 1)
    vis_ally2  = o[:, :, 22:23] # (batch, agents, 1)
    visibility = th.cat([vis_enemy0, vis_ally1, vis_ally2], dim=-1)  # (batch, agents, 3)

    # Movement possibility flags
    move_flags = o[:, :, 0:4]  # (batch, agents, 4)

    # Timestamp placeholder (zeros)
    timestamp = th.zeros(batch_size, n_agents, 1, device=device, dtype=o.dtype)


    # Last action one-hot (7 dims)
    last_actions = o[:, :, 31:38]  # (batch, agents, 7)

    # Concatenate all message parts -> message_dim = 15
    message = th.cat(
        [visibility, move_flags, timestamp, last_actions],
        dim=-1
    )  # (batch, agents, 15)

    # Broadcast messages: each agent receives messages from other two agents (excluding self)
    message_dim = message.shape[-1]

    # Expand for (sender, receiver)
    message_exp = message.unsqueeze(2).repeat(1, 1, n_agents, 1)  # (batch, sender, receiver, dim)

    # Mask self messages (sender == receiver)
    sender_ids = th.arange(n_agents, device=device).view(1, n_agents, 1)
    receiver_ids = th.arange(n_agents, device=device).view(1, 1, n_agents)
    self_mask = (sender_ids == receiver_ids).expand(batch_size, -1, -1)
    message_exp = message_exp.masked_fill(self_mask.unsqueeze(-1), 0.0)

    # Reorder to (batch, receiver, sender, dim)
    message_exp = message_exp.permute(0, 2, 1, 3)

    # Indices of other agents for each receiver (hardcoded for 3 agents)
    sender_indices = th.tensor([[1, 2],
                                [0, 2],
                                [0, 1]], device=device)  # (3, 2)
    sender_indices = sender_indices.unsqueeze(0).expand(batch_size, -1, -1)

    # Gather other agents' messages
    messages_from_others = th.gather(
        message_exp, 2,
        sender_indices.unsqueeze(-1).expand(-1, -1, -1, message_dim)
    )  # (batch, receiver, 2, dim)

    # Concatenate messages from two other agents: (batch, 3, 30)
    messages_concat = messages_from_others.reshape(batch_size, n_agents, -1)

    # Append to original observations: (batch, 3, 42 + 30) = (batch, 3, 72)
    messages_o = th.cat([o, messages_concat], dim=-1)

    return messages_o
