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
    Address feedback on weak predictability of absolute enemy positions/directions,
    ball absolute position/direction, and asymmetries in ball and enemy dynamic info.
    Provide novel, complementary, and task-relevant information that cannot be inferred
    locally, improving global situational awareness and coordination.

    Message Content:
    Each agent sends a compact message including:

    1. **Absolute positions and directions of key enemies and ball:**
       - Enemy goalkeeper absolute position (2 dims) and direction (2 dims).
       - Enemy center-back1 absolute position (2 dims) and direction (2 dims).
       - Enemy center-back2 absolute position (2 dims) and direction (2 dims).
       - Ball absolute position (2 dims).
       
       *Rationale:* Sharing absolute enemy and ball states enables both agents to have a consistent global spatial reference frame, reducing ambiguity.


    Communication Protocol:
    - Peer-to-peer: each agent receives the other’s message.
    - Messages carry global absolute positions and directions of enemies and ball.
    - No trainable components; purely direct extraction and concatenation.
    - Compact message size: 4 (GK) + 4 (CB1) + 4 (CB2) + 2 (Ball) = 14 dims total.

    Why this design?
    - Addresses the feedback’s core missing info: absolute enemy and ball states.
    - Maintains compactness and explicit clarity.
    """

    return (
        "Message Design:\n"
        "- Enemy goalkeeper absolute position (2 dims) and direction (2 dims).\n"
        "- Enemy center-back1 absolute position (2 dims) and direction (2 dims).\n"
        "- Enemy center-back2 absolute position (2 dims) and direction (2 dims).\n"
        "- Ball absolute position (2 dims).\n\n"
        "Rationale:\n"
        "- Absolute enemy and ball states: provide global spatial reference, improving predictability and coordination.\n"
        "Communication is peer-to-peer: each agent receives the other's message.\n"
        "Messages are concatenated with original observations for downstream processing.\n"
        "No trainable components; direct tensor extraction ensures computational efficiency.\n"
        "Designed to minimize redundancy and maximize task-relevant, novel information exchange."
    )


def communication(o):
    """
    Arguments:
        o: torch.Tensor with shape (batch_size, 2, 43)
           Batch size, 2 agents, 43 obs dims per agent.

    Output:
        Tensor with shape (batch_size, 2, 43 + 14)
        where message_dim = 14 as per message_design_instruction.

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

    # Extract slices for absolute positions and directions of enemies and ball:
    # Enemy goalkeeper absolute position: dims 0,1 of enemy? Observation dims only have ego absolute pos,
    # so enemy absolute pos must be inferred from relative position + ego pos.
    # Since only ego absolute pos is given, absolute enemy pos is not directly in o.
    # However, the feedback emphasizes including absolute enemy positions.
    # We must reconstruct absolute enemy positions by adding ego absolute position + enemy relative position.
    # This is allowed as no trainable components; direct computation is OK.

    # Ego absolute position (x,y): dims [0,1]
    ego_pos = o[:, :, 0:2]  # (batch, agent, 2)

    # Relative positions of enemies from ego:
    # Enemy goalkeeper relative pos: dims [8,9]
    goalkeeper_rel_pos = o[:, :, 8:10]

    # Enemy center-back1 relative pos: dims [10,11]
    centerback1_rel_pos = o[:, :, 10:12]

    # Enemy center-back2 relative pos: dims [??]
    # The observation dims 41 and 42 are categorical for center-back1 and center-back2 indicators.
    # The observation does NOT explicitly provide center-back2 relative pos/direction.
    # Only one center-back relative pos/direction is given (dims 10-11 pos, 14-15 dir).
    # The two categorical dims 41 and 42 indicate if this agent is center-back1 or center-back2.
    # So likely only one center-back is visible at a time.
    # To handle both center-backs absolute info, we can:
    # - For center-back2, if visible (using dim 42 == 1), use same relative pos/direction dims (10-11, 14-15).
    # - Else fill zeros.
    # So we must separate center-back1 and center-back2 info by masking based on dims 41 and 42.

    # Enemy center-back direction: dims [14,15]
    centerback_dir = o[:, :, 14:16]

    # Ball absolute position Z is dim 18, but we only want X,Y absolute pos.
    # Ball relative pos: dims [16,17]
    ball_rel_pos = o[:, :, 16:18]

    # Ball absolute position Z: dim 18 (ignored for 2D)
    ball_abs_z = o[:, :, 18:19]  # unused

    # Ball direction: dims [19,20,21]
    # ball_dir_3d = o[:, :, 19:22]
    # We keep only X,Y ball directions (19,20)

    # Center-back1 and center-back2 indicators (categorical)
    is_centerback1 = o[:, :, 41:42]  # shape (batch, agent, 1)
    is_centerback2 = o[:, :, 42:43]  # shape (batch, agent, 1)

    # Compute absolute positions of enemies and ball by adding ego absolute pos + relative pos
    # This is valid because:
    # abs_enemy_pos = ego_pos + enemy_rel_pos (assuming same coordinate frame)
    goalkeeper_abs_pos = ego_pos + goalkeeper_rel_pos
    centerback_abs_pos = ego_pos + centerback1_rel_pos  # For whichever center-back is visible

    # For center-back2, if visible, use centerback_abs_pos, else zeros
    # We create two separate absolute pos/directions for center-back1 and center-back2
    # by masking centerback_abs_pos and centerback_dir with is_centerback1 and is_centerback2

    # Expand masks to match dims
    mask_cb1 = is_centerback1.expand(-1, -1, 2)  # (batch, agent, 2)
    mask_cb2 = is_centerback2.expand(-1, -1, 2)

    # Absolute pos for center-back1 and center-back2
    centerback1_abs_pos = centerback_abs_pos * mask_cb1
    centerback2_abs_pos = centerback_abs_pos * mask_cb2

    # Similarly for directions
    centerback_dir_cb1 = centerback_dir * mask_cb1
    centerback_dir_cb2 = centerback_dir * mask_cb2

    # Ball absolute position X,Y = ego_pos + ball_rel_pos
    ball_abs_pos = ego_pos + ball_rel_pos  # (batch, agent, 2)

    # Construct message per agent:
    # Layout:
    # [Enemy goalkeeper absolute pos (2 dims),
    #  Enemy goalkeeper direction (2 dims),
    #  Enemy center-back1 absolute pos (2 dims),
    #  Enemy center-back1 direction (2 dims),
    #  Enemy center-back2 absolute pos (2 dims),
    #  Enemy center-back2 direction (2 dims),
    #  Ball absolute pos (2 dims)]

    # Extract goalkeeper direction dims [12,13]
    goalkeeper_dir = o[:, :, 12:14]

    message = th.cat([
        goalkeeper_abs_pos,        # 2
        goalkeeper_dir,            # 2
        centerback1_abs_pos,       # 2
        centerback_dir_cb1,        # 2
        centerback2_abs_pos,       # 2
        centerback_dir_cb2,        # 2
        ball_abs_pos,              # 2
    ], dim=-1)  # total dims = 2+2+2+2+2+2+2 = 14

    # Peer-to-peer communication: each agent receives other's message
    received_message = message[:, [1, 0], :]  # swap agents dimension

    # Concatenate received message to original observation along last dim
    messages_o = th.cat([o, received_message], dim=-1)  # (batch, 2, 43 + 14)

    return messages_o
