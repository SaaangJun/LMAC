import torch as th

def message_design_instruction():
    """
    Message Design Instruction for LBF (grid obs) Task
    --------------------------------------------------
    **Communication Objective:**
    Enhance agent coordination under partial observability by exchanging only non-redundant, task-relevant, and actionable information that the recipient cannot directly observe, with explicit sender identity.

    **Message Construction:**
    1. **Sender Identity (One-hot):**
       - 2D one-hot vector indicating which agent sent the message.
         - [1,0] for agent 0; [0,1] for agent 1.
         - This is essential for grounding and message disambiguation.

    2. **Recent Action (6D one-hot):**
       - Each agent communicates its own most recent action, as encoded in its own observation:
         - o[..., 27:33]: [last_action_0 (No-op), last_action_1 (North), last_action_2 (South), last_action_3 (West), last_action_4 (East), last_action_5 (Pick-up)]
       - This allows the recipient to infer likely agent intentions and avoid conflicts (such as colliding or duplicating effort), as agents cannot directly observe each other's last actions.

    3. **Local Food Observation (9D binary):**
       - Each agent shares the food_layer of its own local 3x3 grid (o[..., 9:18]).
       - This provides food location information outside the recipient’s own local view and helps coordinate efficient foraging, especially for food near the sender but outside the receiver's current grid.

    4. **Local Accessibility Map (9D binary):**
       - Each agent shares its own access_layer (o[..., 18:27]), indicating which cells in its local grid are accessible.
       - This enables recipients to infer possible movement options for the sender and better anticipate/help with path planning, especially in partially observable or blocked environments.

    **Total Message Dimension:** 2 (sender id) + 6 (last action) + 9 (food) + 9 (access) = **26**

    **Communication Pattern:** Peer-to-peer (each agent receives a message from the other agent in the current scenario).

    **Why this information?**
    - **Last Action:** Not visible to others, crucial for predicting teammates’ next moves and avoiding coordination failures.
    - **Local Food & Access:** The most important environmental features for the task (food collection, pathing) that are not always within the other agent's local observation due to the limited 3x3 grid. Sharing these extends the effective field of view.
    - **Sender Identity:** Ensures correct attribution of observed intent and environmental information, which is necessary for correct state update and planning.

    **No redundant or self-observable data is transmitted.**
    """

    return (
        "Message Design for LBF Task:\n"
        "Each agent sends the following to its peer in the same scenario:\n"
        "1. Sender identity (2D one-hot).\n"
        "2. Its own last action (6D one-hot, o[...,27:33]).\n"
        "3. Its own local food_layer (9D, o[...,9:18]).\n"
        "4. Its own local access_layer (9D, o[...,18:27]).\n"
        "Total message dimension: 26.\n"
        "Messages are exchanged peer-to-peer (agent 0 receives from 1 and vice versa, per scenario).\n"
        "See docstring for rationale."
    )


def communication(o):
    """
    Implements peer-to-peer message exchange for LBF (grid obs) MARL scenario.
    Each agent receives a message from the other agent in its batch scenario, with the following structure:
        [sender_id (2D one-hot), last_action (6D one-hot), food_layer (9D), access_layer (9D)] = 26D

    Args:
        o: torch.Tensor, shape (batch, 2, 39)

    Returns:
        messages_o: torch.Tensor, shape (batch, 2, 65) = (batch, 2, 39+26)
    """
    # Device & dtype consistency
    device = o.device
    dtype = o.dtype
    batch = o.shape[0]
    n_agents = o.shape[1]

    # Build message for each agent (shape: batch, 2, 26)
    # Sender identity (2D one-hot)
    sender_ids = th.eye(n_agents, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)  # (batch, 2, 2)

    # Last action (6D one-hot)
    last_action = o[..., 27:33]  # (batch, 2, 6)

    # Local food_layer (9D)
    food_layer = o[..., 9:18]    # (batch, 2, 9)

    # Local access_layer (9D)
    access_layer = o[..., 18:27] # (batch, 2, 9)

    # Concatenate message fields -> (batch, 2, 26)
    message = th.cat([sender_ids, last_action, food_layer, access_layer], dim=-1)

    # Transpose messages so that each agent receives from its peer
    # For 2 agents: swap along agent axis
    peer_message = message.flip(1)  # (batch, 2, 26): agent 0 gets agent 1's message and vice versa

    # Concatenate received message to each agent's own observation
    messages_o = th.cat([o, peer_message], dim=-1)  # (batch, 2, 39+26)

    return messages_o
