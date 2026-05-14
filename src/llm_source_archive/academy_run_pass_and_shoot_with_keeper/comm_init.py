import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Task Context:
    Two attackers coordinate near the penalty box edge to score.
    - Attacker 0 (Winger): Has the ball, closely marked by a defender.
    - Attacker 1 (Center): Unmarked, facing goal, ready to shoot.
    Key: Winger must pass at the right moment for a clean shot.

    Communication Goal:
    Share only essential, non-redundant information that the other agent cannot easily infer.
    Each agent sends:
      1. Self position and direction (to localize themselves).
      2. Ball relative position & direction (for the winger, ball possession info; for center, ball location).
      3. Last action one-hot vector (to infer intentions and potential next moves).
      4. Enemy defender (center-back) relative position and direction (only the winger is marked and sees defender closely).
      5. Enemy goalkeeper relative position (both see goalkeeper, but winger’s view might be more occluded; still useful).
      6. Sender identity (one-hot vector of length 2).

    Why these?
    - Position & direction: Critical for spatial coordination (where and how moving).
    - Ball info: Essential for pass timing and shot preparation.
    - Last action: Reveals intent (e.g., preparing to pass, shoot, dribble).
    - Defender info: Winger’s defender marking is vital for center to anticipate pressure.
    - Goalkeeper info: Relevant for shot timing and aiming.
    - Sender ID: To distinguish source of message, enabling correct interpretation.

    Communication Protocol:
    - Peer-to-peer: each agent receives the other's message.
    - Messages are concatenated to original observations for joint input.
    - No trainable components; only direct extraction and concatenation.
    - Compactness: Message length = 2 (pos) + 2 (dir) + 3 (ball pos+dir) + 4 (last actions) + 4 (defender pos+dir) + 2 (goalkeeper pos) + 2 (sender ID) = 19 dims approx.
      (Last action is 15 dims, so we include all 15 for clarity and sufficiency, not partial.)
    - To keep clarity and sufficiency, we include ALL 15 last action one-hot dims, since last action is categorical and crucial to infer intent.

    Final message per agent:
    - Ego pos (2)
    - Ego dir (2)
    - Ball relative pos (2) + ball direction (3)
    - Last action one-hot (15)
    - Enemy center-back relative pos (2) + direction (2)
    - Enemy goalkeeper relative pos (2)
    - Sender ID one-hot (2)

    Total message dims = 2 + 2 + 2 + 3 + 15 + 2 + 2 + 2 + 2 = 32 dims

    This message balances sufficiency and compactness, avoids redundancy (each agent sends only self info and enemies relative to self),
    and provides actionable, explicit info to the other agent for coordination.

    """

    return (
        "Message Design:\n"
        "- Sender identity: one-hot vector (2 dims) indicating agent ID.\n"
        "- Sender ego absolute position (2 dims) and direction (2 dims).\n"
        "- Ball relative position (2 dims) and ball direction (3 dims) from sender's perspective.\n"
        "- Sender last action one-hot vector (15 dims) to convey intent.\n"
        "- Enemy center-back relative position (2 dims) and direction (2 dims) as perceived by sender.\n"
        "- Enemy goalkeeper relative position (2 dims) from sender's perspective.\n\n"
        "Rationale:\n"
        "- Ego position/direction: spatial situational awareness.\n"
        "- Ball info: critical for timing passes and shots.\n"
        "- Last action: reveals behavioral intent, aiding prediction.\n"
        "- Enemy defender and goalkeeper info: essential for anticipating threats and shot timing.\n"
        "- Sender ID: disambiguates message source.\n\n"
        "Communication is peer-to-peer: each agent receives the other's message.\n"
        "Messages are concatenated with original observations for downstream processing.\n"
        "No trainable components; direct tensor extraction ensures computational efficiency.\n"
        "Designed to minimize redundancy and maximize task-relevant information exchange."
    )


def communication(o):
    """
    Input:
        o: torch.Tensor with shape (32, 2, 43)
           Batch size = 32, 2 agents, 43 obs dims per agent.

    Output:
        Tensor with shape (32, 2, 43 + message_dim)
        where message_dim = 32 (see message_design_instruction).

    Communication Protocol:
      - Peer-to-peer: each agent receives the other's message.
      - Message constructed solely from sender's own observation.
      - Concatenate received message to each agent's observation along last dim.

    Implementation Details:
      - Use vectorized operations, no explicit for-loops over batch or agents.
      - Device consistency with input tensor.
      - Handle sparse zero observations gracefully.
    """

    device = o.device
    batch_size = o.shape[0]
    n_agents = o.shape[1]

    # Indices for message components:
    # Ego position absolute: [0,1]
    # Ego direction: [4,5]
    # Ball relative pos: [16,17]
    # Ball direction: [19,20,21]
    # Last action one-hot: [22:37] inclusive (16 dims: 22 to 37)
    #   Actually dims 22 to 40 inclusive (19 dims)? Recheck:
    #   From description:
    #   22: Idle
    #   23-40: other last actions (19 dims total from 22 to 40)
    #   Counting: 22 to 40 inclusive = 19 dims
    #   So last action one-hot length = 19 dims (22 to 40 inclusive)
    # Enemy center-back relative pos: [10,11]
    # Enemy center-back direction: [14,15]
    # Enemy goalkeeper relative pos: [8,9]
    # Sender ID: one-hot vector length 2

    # Extract slices:
    ego_pos = o[:, :, 0:2]            # (batch, agent, 2)
    ego_dir = o[:, :, 4:6]            # (batch, agent, 2)
    ball_rel_pos = o[:, :, 16:18]     # (batch, agent, 2)
    ball_dir = o[:, :, 19:22]         # (batch, agent, 3)
    last_action = o[:, :, 22:41]      # (batch, agent, 19)
    centerback_rel_pos = o[:, :, 10:12]   # (batch, agent, 2)
    centerback_dir = o[:, :, 14:16]        # (batch, agent, 2)
    goalkeeper_rel_pos = o[:, :, 8:10]     # (batch, agent, 2)

    # Construct sender identity one-hot vector:
    # agent_id = 0 or 1
    # Create tensor shape (1,2,2) with one-hot along last dim
    sender_id = th.eye(n_agents, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, agent, 2)

    # Concatenate all message parts for each agent:
    # message shape (batch, agent, 2+2+2+3+19+2+2+2) = (batch, agent, 32)
    message = th.cat([
        ego_pos,
        ego_dir,
        ball_rel_pos,
        ball_dir,
        last_action,
        centerback_rel_pos,
        centerback_dir,
        goalkeeper_rel_pos,
        sender_id
    ], dim=-1)  # (batch, agent, 32)

    # Communication: each agent receives the OTHER agent's message.
    # We swap the agent dimension to simulate peer-to-peer message passing.
    # For agent 0, receive message from agent 1; for agent 1, receive message from agent 0.
    received_message = message[:, [1, 0], :]  # Swap agents

    # Concatenate received message with original observation along last dim
    messages_o = th.cat([o, received_message], dim=-1)  # (batch, 2, 43 + 32)

    return messages_o
