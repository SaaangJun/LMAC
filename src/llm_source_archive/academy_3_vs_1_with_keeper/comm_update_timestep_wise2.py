import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Task Context:
    Three attackers (agents) face a defender and goalkeeper near the box.
    The central attacker initially holds the ball and must decide to dribble or pass.
    Coordination requires resolving ambiguities in absolute positions and movement trends of teammates and opponents,
    especially under partial observability and sparse direct information.

    Communication Protocol:
    - Broadcast communication: each agent sends a message to all others.
    - Messages include a sender identity (one-hot vector of length 3).
    - Each agent receives messages from the other two agents only.
    - Messages are concatenated to original observations, resulting in shape (batch_size, 3, 48 + message_dim).

    Key Improvements over previous design:
    1) **Explicit teammate absolute positions and directions**:
       - Each agent shares the absolute positions and directions of *both* teammates as perceived,
         allowing teammates to cross-validate and reconstruct global positions more accurately.
       - This helps resolve ambiguities when teammates are out of direct sight or obscured.

    2) **Visibility/Confidence flags**:
       - For each teammate and key opponents (goalkeeper, center-back, ball), the sender includes a binary visibility flag
         indicating whether that entity is currently observed (non-zero in observation).
       - This guides receivers in weighting or trusting the relayed positional info.

    3) **Short-term temporal velocity estimates**:
       - For ego agent and teammates (from ego's observation), we compute simple velocity approximations
         based on position differences over the last 10 steps to convey movement trends.
       - This temporal context helps predict future positions and disambiguate sudden movements.

    4) **Absolute ball position and direction**:
       - Explicit absolute ball position and direction from ego's observation (currently missing) are shared.
       - This ensures consistent ball state awareness across agents.

    5) **Relative positions normalized to a common reference frame**:
       - To reduce interpretation discrepancies, relative positions of teammates and opponents are normalized by field dimensions implicitly,
         but since raw observations are normalized [-1,1], we keep as is, trusting shared absolute positions and velocities to disambiguate.

    6) **Sender identity**:
       - One-hot encoding to identify message origin, enabling receivers to correctly associate information.

    Message Content (per agent):
    - For each teammate (2 teammates): 
      - Absolute position (X, Y) (2x2=4 dims)
      - Absolute direction (X, Y) (2x2=4 dims)
      - Visibility flag (1 dim per teammate, 2 dims)
      - Velocity estimate (X, Y) over last 10 steps (2x2=4 dims)
    - For key opponents (goalkeeper and center-back):
      - Visibility flags (2 dims)
    - Ball:
      - Absolute position (X, Y) (2 dims)
      - Absolute direction (X, Y) (2 dims)
      - Visibility flag (1 dim)
      - Velocity estimate (X, Y) over last 10 steps (2 dims)
    - Ego agent:
      - Velocity estimate (X, Y) (2 dims)
    - Sender one-hot ID (3 dims)

    Total message dimension:
    - Teammates: 4 + 4 + 2 + 4 = 14 dims
    - Opponent visibility: 2 dims
    - Ball: 2 + 2 + 1 + 2 = 7 dims
    - Ego velocity: 2 dims
    - Sender ID: 3 dims
    -------------------------
    Sum = 14 + 2 + 7 + 2 + 3 = 28 dims per message

    Rationale:
    - Sharing teammates' absolute positions and directions (not just ego's) fills gaps in global spatial info.
    - Visibility flags signal if info is reliable or stale.
    - Velocity estimates provide temporal context, improving movement prediction and coordination.
    - Absolute ball info removes ambiguity in ball state.
    - Ego velocity helps teammates infer sender's recent movement pattern (important for decision making).
    - Sender ID ensures explicit grounding.

    This design addresses feedback by providing novel, complementary information that cannot be reliably inferred locally,
    especially for weakly predictable absolute positions and directions of teammates and key opponents.

    Communication is broadcast with each agent receiving combined messages from the other two agents.

    """

    return message_design_instruction.__doc__


def communication(o: th.Tensor) -> th.Tensor:
    """
    Args:
        o: Observation tensor with shape (batch=32, timesteps=T, agents=3, obs_dim=48)
           Contains time history (up to current step) for all agents.

    Returns:
        Tensor of shape (batch_size, 3, 48 + 2*message_dim=48+34=82)
        with original observations at current timestep concatenated with messages from other agents.
    """

    device = o.device
    batch_size, T, num_agents, obs_dim = o.shape
    assert num_agents == 3 and obs_dim == 48, "Unexpected input shape"

    # We'll build messages based on the last 10 timesteps (T dimension),
    # but only output messages for current timestep (last in T).

    # Index for current timestep (last timestep)
    cur_t = T - 1

    # Extract current obs for all agents: shape (batch, agents, obs_dim)
    o_cur = o[:, cur_t, :, :]  # (batch, 3, 48)

    # Helper: function to compute velocity approx over last 10 steps for given position dims
    # velocity = (pos_current - pos_past) / dt, dt = number of steps (T-1)
    # To be robust: if T<2, velocity=0
    def compute_velocity(pos_hist: th.Tensor) -> th.Tensor:
        # pos_hist: (batch, T, agents, 2)
        if T < 2:
            return th.zeros((batch_size, num_agents, 2), device=device)
        else:
            # Velocity = (pos_current - pos_earliest) / (T-1)
            vel = (pos_hist[:, -1, :, :] - pos_hist[:, 0, :, :]) / (T - 1)
            return vel  # (batch, agents, 2)

    # --- 1) Compute absolute positions and velocities for all agents (ego + teammates) ---
    # Ego absolute position (X,Y): dims 0,1
    ego_abs_pos_hist = o[:, :, :, 0:2]  # (batch, T, 3, 2)
    ego_abs_vel = compute_velocity(ego_abs_pos_hist)  # (batch, 3, 2)

    # Ego absolute direction (X,Y): dims 6,7 at current timestep
    ego_abs_dir = o_cur[:, :, 6:8]  # (batch, 3, 2)

    # --- 2) Visibility flags ---
    # Visibility is determined by checking if the observed relative positions or absolute positions are non-zero.
    # For teammates and opponents, we check if relative positions are non-zero at current timestep.

    # Visibility of teammates from ego perspective:
    # For agent i, teammates are (1 and 2) excluding i.
    # Relative positions of Ally1: dims 2,3
    # Relative positions of Ally2: dims 4,5
    # We'll create a helper function to get teammate relative pos and visibility per agent.

    # Extract relative positions of Ally1 and Ally2 from each agent's perspective
    # o_cur shape: (batch, 3, 48)
    ally1_rel_pos = o_cur[:, :, 2:4]  # (batch,3,2)
    ally2_rel_pos = o_cur[:, :, 4:6]  # (batch,3,2)

    # Visibility = 1 if norm of relative pos > 0 (means visible), else 0
    ally1_vis = (ally1_rel_pos.abs().sum(dim=2) > 1e-5).float()  # (batch,3)
    ally2_vis = (ally2_rel_pos.abs().sum(dim=2) > 1e-5).float()  # (batch,3)

    # --- 3) Opponent visibility flags ---
    # Enemy goalkeeper relative pos: dims 12,13
    gk_rel_pos = o_cur[:, :, 12:14]
    gk_vis = (gk_rel_pos.abs().sum(dim=2) > 1e-5).float()  # (batch,3)

    # Enemy center-back relative pos: dims 14,15
    cb_rel_pos = o_cur[:, :, 14:16]
    cb_vis = (cb_rel_pos.abs().sum(dim=2) > 1e-5).float()  # (batch,3)

    # --- 4) Ball absolute position and direction ---
    # Ball absolute position dims: 22 (Z) is vertical, so use X,Y absolute from ego perspective:
    # Given only ball relative position (20,21) and ball absolute Z (22),
    # The absolute X,Y of ball are not directly given.
    # However, since ego absolute pos is known, and ball relative pos is relative to ego,
    # we can compute ball absolute position as: ego_abs_pos + ball_rel_pos (approximate)
    # This assumes relative pos is ball_pos - ego_pos normalized in [-1,1], consistent across agents.

    # Extract ball relative pos and direction at current timestep
    ball_rel_pos = o_cur[:, :, 20:22]  # (batch,3,2)
    ball_abs_z = o_cur[:, :, 22:23]    # (batch,3,1) (vertical pos, less relevant here but keep for info)
    ball_dir = o_cur[:, :, 23:25]      # (batch,3,2)

    # Compute ball absolute position approx: ego_abs_pos + ball_rel_pos
    ball_abs_pos = ego_abs_pos_hist[:, cur_t, :, :] + ball_rel_pos  # (batch,3,2)

    # Compute ball velocity over last 10 steps similarly:
    # Ball relative pos over time:
    ball_rel_pos_hist = o[:, :, :, 20:22]  # (batch, T, 3, 2)
    ball_abs_pos_hist = ego_abs_pos_hist + ball_rel_pos_hist  # (batch, T, 3, 2)
    ball_vel = compute_velocity(ball_abs_pos_hist)  # (batch, 3, 2)

    # --- 5) Teammates' absolute positions and directions as perceived by ego ---
    # For each agent (ego), teammates are the other two agents:
    # We want to extract from ego's observation the teammates' absolute positions and directions as perceived.

    # From ego perspective:
    # Ally1 relative pos: dims 2,3 (relative X,Y)
    # Ally2 relative pos: dims 4,5 (relative X,Y)
    # Ally1 direction: dims 8,9
    # Ally2 direction: dims 10,11

    # To get teammates' absolute positions from ego's perspective:
    # teammate_abs_pos = ego_abs_pos + teammate_rel_pos

    # Prepare containers for teammates' absolute pos and dir per agent:
    # Shape: (batch, 3 agents, 2 teammates, 2 dims)

    teammates_abs_pos = th.zeros(batch_size, num_agents, 2, 2, device=device)
    teammates_abs_dir = th.zeros(batch_size, num_agents, 2, 2, device=device)
    teammates_vel = th.zeros(batch_size, num_agents, 2, 2, device=device)
    teammates_vis = th.zeros(batch_size, num_agents, 2, device=device)

    # For each agent, fill teammates info:
    for agent_id in range(num_agents):
        # Indices of teammates
        tm = [x for x in range(num_agents) if x != agent_id]
        # Ally1 rel pos and dir from ego perspective
        ally1_rel_pos_agent = o_cur[:, agent_id, 2:4]  # (batch,2)
        ally1_dir_agent = o_cur[:, agent_id, 8:10]     # (batch,2)
        ally1_vis_agent = (ally1_rel_pos_agent.abs().sum(dim=1) > 1e-5).float()  # (batch,)

        # Ally2 rel pos and dir from ego perspective
        ally2_rel_pos_agent = o_cur[:, agent_id, 4:6]  # (batch,2)
        ally2_dir_agent = o_cur[:, agent_id, 10:12]    # (batch,2)
        ally2_vis_agent = (ally2_rel_pos_agent.abs().sum(dim=1) > 1e-5).float()  # (batch,)

        # Compute absolute pos of teammates w.r.t ego agent
        ego_pos_agent = o_cur[:, agent_id, 0:2]  # (batch,2)

        tm1_abs_pos = ego_pos_agent + ally1_rel_pos_agent  # (batch,2)
        tm2_abs_pos = ego_pos_agent + ally2_rel_pos_agent  # (batch,2)

        # Store
        teammates_abs_pos[:, agent_id, 0, :] = tm1_abs_pos
        teammates_abs_pos[:, agent_id, 1, :] = tm2_abs_pos

        teammates_abs_dir[:, agent_id, 0, :] = ally1_dir_agent
        teammates_abs_dir[:, agent_id, 1, :] = ally2_dir_agent

        teammates_vis[:, agent_id, 0] = ally1_vis_agent
        teammates_vis[:, agent_id, 1] = ally2_vis_agent

    # --- 6) Compute velocity estimates for teammates from ego perspective ---
    # For each agent and each teammate, compute velocity by difference of absolute positions over last 10 steps.

    # To do this, we need to reconstruct teammates' absolute positions over time from ego perspective:
    # ego_abs_pos_hist: (batch,T,3,2)
    # ally1_rel_pos_hist: (batch,T,3,2)
    # ally2_rel_pos_hist: (batch,T,3,2)

    ally1_rel_pos_hist = o[:, :, :, 2:4]  # (batch,T,3,2)
    ally2_rel_pos_hist = o[:, :, :, 4:6]  # (batch,T,3,2)

    teammates_vel_hist = th.zeros(batch_size, num_agents, 2, 2, device=device)  # (batch, agent, teammate_idx, 2)

    for agent_id in range(num_agents):
        ego_pos_hist = o[:, :, agent_id, 0:2]  # (batch,T,2)
        # Ally1 absolute pos over time:
        ally1_abs_hist = ego_pos_hist + ally1_rel_pos_hist[:, :, agent_id, :]  # (batch,T,2)
        # Ally2 absolute pos over time:
        ally2_abs_hist = ego_pos_hist + ally2_rel_pos_hist[:, :, agent_id, :]  # (batch,T,2)

        # Velocity over last 10 steps:
        if T < 2:
            vel1 = th.zeros((batch_size, 2), device=device)
            vel2 = th.zeros((batch_size, 2), device=device)
        else:
            vel1 = (ally1_abs_hist[:, -1, :] - ally1_abs_hist[:, 0, :]) / (T - 1)
            vel2 = (ally2_abs_hist[:, -1, :] - ally2_abs_hist[:, 0, :]) / (T - 1)

        teammates_vel_hist[:, agent_id, 0, :] = vel1
        teammates_vel_hist[:, agent_id, 1, :] = vel2

    # --- 7) Ball visibility from ego perspective ---
    # Ball relative pos norm > 0 means visible
    ball_vis = (ball_rel_pos.abs().sum(dim=2) > 1e-5).float()  # (batch, 3)

    # --- 8) Ego velocity (already computed) ---
    # ego_abs_vel (batch, 3, 2)

    # --- 9) Compose messages per agent ---
    # For each agent, message contains:

    # a) Teammates absolute directions (2 teammates * 2 dims) -> 4 dims
    # b) Teammates visibility flags (2 dims)
    # c) Teammates velocity estimates (2 teammates * 2 dims) -> 4 dims
    # d) Opponent visibility flags: goalkeeper (1 dim), center-back (1 dim) -> 2 dims
    # e) Ball visibility flag (1 dim)
    # f) Ball velocity estimate (2 dims)
    # g) Ego velocity (2 dims)
    # (Removed: Teammates Abs Pos, Ball Abs Pos, Ball Dir, Sender ID)

    # Extract opponent visibility flags at current timestep (already computed: gk_vis, cb_vis)
    # Extract ball absolute position and direction from ego perspective (already computed: ball_abs_pos, ball_dir)

    # Compose message tensors:

    # Reshape teammates dims to (batch, agents, feature)
    tm_abs_dir_msg = teammates_abs_dir.reshape(batch_size, num_agents, -1)  # 4 dims
    tm_vis_msg = teammates_vis  # 2 dims
    tm_vel_msg = teammates_vel_hist.reshape(batch_size, num_agents, -1)  # 4 dims

    # Opponent visibility flags
    opp_vis_msg = th.stack([gk_vis, cb_vis], dim=2)  # (batch, agents, 2)

    # Ball info
    ball_vis_msg = ball_vis.unsqueeze(2)  # (batch, agents, 1)
    ball_vel_msg = ball_vel           # (batch, agents, 2)

    # Ego velocity
    ego_vel_msg = ego_abs_vel  # (batch, agents, 2)


    # Concatenate all parts along last dim
    messages = th.cat([
        tm_abs_dir_msg,     # 4
        tm_vis_msg,         # 2
        tm_vel_msg,         # 4
        opp_vis_msg,        # 2
        ball_vis_msg,       # 1
        ball_vel_msg,       # 2
        ego_vel_msg,        # 2
    ], dim=2)  # (batch, 3, 17)

    # --- 10) Prepare received messages for each agent ---
    # Each agent receives messages from the other two agents only, concatenated.

    # Mask to exclude self messages (3x3)
    mask = (1 - th.eye(num_agents, device=device)).bool()  # (3,3)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, agents, agents)

    # Expand messages for broadcasting
    messages_exp = messages.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (batch, receiver_agent, sender_agent, 17)
    mask_exp = mask.unsqueeze(-1)  # (batch, receiver_agent, sender_agent, 1)
    masked_messages = messages_exp * mask_exp.float()  # zero out self messages

    # Indices of other agents per agent (hardcoded)
    other_agents = th.tensor([[1, 2], [0, 2], [0, 1]], device=device)  # (3,2)

    batch_idx = th.arange(batch_size, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, num_agents, 2)
    agent_idx = th.arange(num_agents, device=device).unsqueeze(-1).expand(num_agents, 2)
    agent_idx = agent_idx.unsqueeze(0).expand(batch_size, -1, -1)

    gathered_messages = masked_messages[batch_idx, agent_idx, other_agents.unsqueeze(0).expand(batch_size, -1, -1), :]  # (batch,3,2,17)

    # Concatenate messages from other two agents along feature dim
    received_messages = gathered_messages.reshape(batch_size, num_agents, -1)  # (batch,3,34)

    # --- 11) Final output: concatenate original current obs with received messages ---
    messages_o = th.cat([o_cur, received_messages], dim=2)  # (batch, 3, 48+34=82)

    return messages_o
