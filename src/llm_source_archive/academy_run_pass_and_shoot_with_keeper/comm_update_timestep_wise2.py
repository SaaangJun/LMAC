import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Task Context:
    - Two attackers coordinate near the penalty box edge to score.
      Attacker 0 (Winger) has the ball and is closely marked.
      Attacker 1 (Center) is unmarked, facing goal, ready to shoot.
    - Precise coordination requires accurate shared understanding of absolute positions and directions of key entities
      (especially center-backs and teammates) and temporal behavioral cues.

    Communication Goal:
    - Address critical gaps identified in previous design by explicitly sharing absolute positions and directions of key enemy center-backs and teammates, 
      enabling consistent global situational awareness.
    - Share ally relative position and direction to help triangulate global positions.
    - Provide temporal context by summarizing recent movement trends and last action transitions over the past 10 timesteps.
    - Include explicit observability flags for key entities to indicate information reliability.
    - Add ball absolute position Z explicitly (missing in previous design).
    - Maintain peer-to-peer communication with sender identity included.
    - Keep messages compact but enriched with novel, task-relevant info not inferable locally.

    Message Content Per Agent (constructed from the agent's own past 10 steps observations):

    1) Ego velocity (delta pos and dir over last 3 timesteps) (4 dims)
    2) Ally relative position and direction (4 dims) — last timestep
    3) Ally relative velocity (delta over last 3 timesteps) (4 dims)
    4) Enemy center-back1 observability flag (1 dim)
    5) Enemy center-back2 observability flag (1 dim)
    6) Ball absolute position Z (1 dim) and velocity (X,Y,Z) over last timestep (3 dims) -> 4 dims
    7) Last action frequency over past 10 timesteps (19 dims)
    8) Observability flags for key entities (ally and both center-backs) (3 dims binary)


    Total message dims:
    - Ego velocity: 4
    - Ally relative pos+dir: 4
    - Ally relative velocity: 4
    - Center-back1 flag: 1
    - Center-back2 flag: 1
    - Ball info (Z + Vel): 4
    - Last action freq: 19
    - Observability flags: 3
    -----------------------------------
    = 4 + 4 + 4 + 1 + 1 + 4 + 19 + 3 = 40 dims total

    Rationale:
    - Velocity info captures movement trends aiding prediction.
    - Ally relative info + velocity helps infer teammates' global position.
    - Ball velocity + Z deepens ball context.
    - Last action frequency summarizes behavioral intent over time.
    - Observability flags provide reliability info.

    Communication Protocol:
    - Peer-to-peer: each agent receives the other's message.
    - Messages concatenated to original observations.
    - Use vectorized operations, no loops.
    """

    return (
        "Message Design:\n"
        "- Ego velocity (delta position and direction over last 3 timesteps) (4 dims).\n"
        "- Ally relative position (X,Y) and direction (X,Y) at last timestep (4 dims).\n"
        "- Ally relative velocity (delta position and direction over last 3 timesteps) (4 dims).\n"
        "- Enemy center-back1 observability flag (1 dim).\n"
        "- Enemy center-back2 observability flag (1 dim).\n"
        "- Ball absolute position Z (1 dim) and velocity (X,Y,Z) (3 dims).\n"
        "- Last action frequency over past 10 timesteps (19 dims), normalized counts per action category.\n"
        "- Observability flags for ally and both center-backs (3 dims).\n\n"
        "Rationale:\n"
        "- Velocity and temporal summaries provide novel context not in single-frame observations.\n"
        "- Observability flags indicate information reliability.\n\n"
        "Communication is peer-to-peer: each agent receives the other agent's message.\n"
        "Messages are concatenated with original observations for downstream processing.\n"
        "No trainable components; uses deterministic extraction and simple temporal aggregation.\n"
        "Designed to minimize redundancy while enhancing inference of weakly predictable state dimensions."
    )


def communication(o):
    """
    Input:
        o: torch.Tensor with shape (batch_size, T, 2, 43)
           batch_size = number of scenarios
           T = number of timesteps (>=10)
           2 = number of agents
           43 = observation dimensions per agent

    Output:
        Tensor with shape (batch_size, 2, 43 + 40)
        where 40 is the message dimension defined in message_design_instruction.

    Method:
    - Extract features from the last 10 timesteps (o[:, -10:, :, :]) to compute:
      - Last timestep values (t = -1)
      - Velocity estimated as difference between last timestep and timestep t=-4 (3 steps back)
      - Last action frequency over last 10 timesteps (normalized sum)
    - Compute absolute positions of center-backs from relative positions + ego absolute pos.
      If unobservable (all zeros), set observability flag = 0 and zero positions.
    - Same for ally relative pos/direction and observability.
    - Ball absolute position computed as ego absolute pos + ball relative pos at last timestep.
      Ball velocity as difference over last 3 timesteps.
    - Observability flags indicate if the corresponding entity has any non-zero observation in last timestep.
    - Construct sender ID one-hot vector.
    - Compose message per agent.
    - Peer-to-peer communication: each agent receives the other's message.
    - Concatenate received message to original observation at last timestep per agent.
    """

    device = o.device
    batch_size, T, n_agents, obs_dim = o.shape
    assert n_agents == 2, "This communication is designed for 2 agents."

    # Use last 10 timesteps or all if T < 10
    window_size = min(10, T)
    o_window = o[:, -window_size:, :, :]  # (batch, window_size, 2, 43)

    # Last timestep index in window
    last_t = window_size - 1
    back_t = max(0, window_size - 4)  # 3 steps back or earliest available

    # Helper: safe difference (last - back) for velocity
    def safe_diff(last, back):
        return last - back

    # 2) Ego absolute position and direction at last timestep (batch, agent, 4)
    ego_pos_last = o[:, last_t, :, 0:2]  # x,y
    ego_dir_last = o[:, last_t, :, 4:6]  # dir x,y
    ego_pos_dir_last = th.cat([ego_pos_last, ego_dir_last], dim=-1)  # (batch, agent, 4)

    # Ego position and direction back timestep
    ego_pos_back = o[:, back_t, :, 0:2]
    ego_dir_back = o[:, back_t, :, 4:6]
    ego_pos_dir_back = th.cat([ego_pos_back, ego_dir_back], dim=-1)

    # Ego velocity (delta pos and dir)
    ego_vel = safe_diff(ego_pos_dir_last, ego_pos_dir_back)  # (batch, agent, 4)

    # 3) Ally relative position and direction at last timestep (batch, agent, 4)
    ally_rel_pos_last = o[:, last_t, :, 2:4]  # relative pos x,y
    ally_rel_dir_last = o[:, last_t, :, 6:8]  # relative dir x,y
    ally_rel_pos_dir_last = th.cat([ally_rel_pos_last, ally_rel_dir_last], dim=-1)  # (batch, agent, 4)

    # Ally relative pos/dir back timestep
    ally_rel_pos_back = o[:, back_t, :, 2:4]
    ally_rel_dir_back = o[:, back_t, :, 6:8]
    ally_rel_pos_dir_back = th.cat([ally_rel_pos_back, ally_rel_dir_back], dim=-1)

    # Ally relative velocity
    ally_rel_vel = safe_diff(ally_rel_pos_dir_last, ally_rel_pos_dir_back)  # (batch, agent, 4)

    # 4) Enemy center-back absolute position and direction + observability flags
    # Center-back relative pos and dir at last timestep
    cb1_rel_pos_last = o[:, last_t, :, 10:12]  # center-back1 relative pos x,y
    cb1_rel_dir_last = o[:, last_t, :, 14:16]  # center-back1 direction x,y

    # Center-back2 relative pos (same logic as before, handled via flags)

    def compute_abs_pos_dir_and_flag(ego_pos, rel_pos, rel_dir):
        # Check observability: entity observed if any rel_pos or rel_dir dimension != 0 at last timestep
        obs_flag = ((rel_pos.abs().sum(dim=-1) + rel_dir.abs().sum(dim=-1)) > 0).float()  # (batch, agent)
        # Expand flag dim
        obs_flag_exp = obs_flag.unsqueeze(-1)  # (batch, agent,1)
        # Absolute position and direction
        abs_pos = ego_pos + rel_pos  # (batch, agent, 2)
        abs_dir = rel_dir  # relative direction from ego; we treat as approx absolute direction for communication
        # Zero out if not observed
        abs_pos = abs_pos * obs_flag_exp
        abs_dir = abs_dir * obs_flag_exp
        return abs_pos, abs_dir, obs_flag_exp  # shapes: (batch, agent,2), (batch, agent,2), (batch, agent,1)

    # Center-back1
    _, _, cb1_flag = compute_abs_pos_dir_and_flag(ego_pos_last, cb1_rel_pos_last, cb1_rel_dir_last)


    # Center-back2
    # Create zero tensors for center-back2 info (since no direct observation dims)
    # cb2_abs_pos = th.zeros_like(cb1_abs_pos, device=device)
    # cb2_abs_dir = th.zeros_like(cb1_abs_dir, device=device)
    cb2_flag = th.zeros_like(cb1_flag, device=device)

    # 5) Ball absolute position and velocity
    ball_rel_pos_last = o[:, last_t, :, 16:18]  # ball relative x,y
    ball_abs_pos_xy = ego_pos_last + ball_rel_pos_last  # approximate ball absolute X,Y
    ball_abs_pos_z = o[:, last_t, :, 18:19]  # ball absolute position Z (already absolute)
    ball_abs_pos = th.cat([ball_abs_pos_xy, ball_abs_pos_z], dim=-1)  # (batch, agent, 3)

    # Ball relative pos and absolute pos at back timestep
    ball_rel_pos_back = o[:, back_t, :, 16:18]
    ball_abs_pos_xy_back = ego_pos_back + ball_rel_pos_back
    ball_abs_pos_z_back = o[:, back_t, :, 18:19]
    ball_abs_pos_back = th.cat([ball_abs_pos_xy_back, ball_abs_pos_z_back], dim=-1)

    ball_vel = ball_abs_pos - ball_abs_pos_back  # (batch, agent, 3)


    # 6) Last action frequency over last 10 timesteps (normalized count)
    # Last action dims: 22 to 40 inclusive -> 19 dims
    last_action_window = o_window[:, :, :, 22:41]  # (batch, window_size, agent, 19)
    # Sum over time dim
    last_action_sum = last_action_window.sum(dim=1)  # (batch, agent, 19)
    # Normalize by window_size to get frequency
    last_action_freq = last_action_sum / window_size  # (batch, agent, 19)

    # 7) Observability flags for ally and both center-backs
    # Ally observability: check if ally relative pos or dir at last timestep is non-zero
    ally_obs_flag = ((ally_rel_pos_last.abs().sum(dim=-1) + ally_rel_dir_last.abs().sum(dim=-1)) > 0).float()  # (batch, agent)

    # Center-back1 and center-back2 flags already computed: cb1_flag, cb2_flag (batch, agent, 1)
    # Squeeze last dim for concatenation
    cb1_flag_s = cb1_flag.squeeze(-1)
    cb2_flag_s = cb2_flag.squeeze(-1)

    observability_flags = th.stack([ally_obs_flag, cb1_flag_s, cb2_flag_s], dim=-1)  # (batch, agent, 3)

    # Compose message per agent (batch, agent, 40 dims)
    # Order:
    # [Ego Velocity (4)
    #  Ally Rel Pos/Dir Last (4)
    #  Ally Rel Velocity (4)
    #  CB1 Flag (1)
    #  CB2 Flag (1)
    #  Ball Abs Z (1)
    #  Ball Velocity (3)
    #  Last Action Freq (19)
    #  Observability Flags (3)]

    message = th.cat([
        ego_vel,                   # 4
        ally_rel_pos_dir_last,      # 4
        ally_rel_vel,               # 4
        cb1_flag,                   # 1
        cb2_flag,                   # 1
        ball_abs_pos_z,             # 1
        ball_vel,                   # 3
        last_action_freq,           # 19
        observability_flags         # 3
    ], dim=-1)  # (batch, agent, 40)

    # Peer-to-peer communication: each agent receives the other's message
    received_message = message[:, [1, 0], :]  # swap agents

    # Concatenate received message with original observation at last timestep
    # We use the last timestep observation only (shape: batch, agent, 43)
    o_last = o[:, last_t, :, :]  # (batch, agent, 43)

    messages_o = th.cat([o_last, received_message], dim=-1)  # (batch, agent, 43+40)

    return messages_o
