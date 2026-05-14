import torch as th

def message_design_instruction():
    """
    Message Design Instruction:

    Task Context:
    - Three attackers (agents) face one defender and goalkeeper near the box.
    - The central attacker holds the ball initially and must decide to dribble or pass.
    - Coordination under defensive pressure requires clear spatial awareness and anticipation.

    Communication Protocol:
    - Broadcast communication: each agent sends a message to all others.
    - Each message includes explicit sender identity (one-hot, length=3).
    - Agents receive concatenated messages from other agents (excluding own).
    - Messages concatenated to original observations: output shape (batch, 3, 48 + message_dim).

    Communication Design Rationale & Improvements Over Previous Version:

    1. Addressing Feedback:
       - Previous messages lacked absolute positions of ball and key enemies/teammates.
       - Without absolute positional references, agents cannot reconstruct teammates' or ball's global positions.
       - This leads to poor predictability of absolute positions, critical for spatial coordination.

    2. Novel Information Shared:
       a) Absolute Ball Position (1D Z coordinate previously missing):
          - Include ball absolute position Z (o[:,:,22]) to complement ball relative XY.
          - Enables receivers to better estimate ball's 3D location.

       b) Absolute Positions of Key Enemies (Goalkeeper & Center-back):
          - Share absolute positions of goalkeeper and center-back (derived using sender's absolute position + enemy relative position).
          - This provides global enemy locations, crucial for spatial planning and avoiding defensive pressure.

       c) Absolute Positions of Visible Allies:
          - Each agent computes absolute positions of visible allies by adding:
            sender's absolute position + ally relative position.
          - Sharing these absolute ally positions helps receivers better localize teammates globally,
            improving spatial situational awareness and coordinated movement.

       d) Center-Midfielder Identities:
          - Include the 3 binary indicators (o[:,:,45:48]) to clarify agent roles.
          - Helps receivers disambiguate teammates and interpret positional info correctly.

       e) Recent Movement Intent & Temporal Context:
          - Share ego agent's direction (already included previously, keep here for completeness).
          - Add a simple temporal context: last action timestamp surrogate by encoding last action types as a compact numeric vector.
          - Since no explicit time info is available, encode last action summary as a numeric vector weighted by action importance.
          - This helps align temporal dynamics and intentions across agents.

    3. Message Construction Details:
       - Absolute positions: 
         * Ball absolute 3D pos: (X,Y from ball relative + ego abs pos, and Z from o[:,:,22])
         * Enemy absolute pos: ego_abs_pos + enemy_rel_pos (goalkeeper and center-back)
         * Ally absolute pos: ego_abs_pos + ally_rel_pos (Ally1 and Ally2)
       - Last action summary as numeric weighted sum to encode temporal intention compactly.
       - Sender identity as one-hot (3 dims).

    4. Message Dimension Calculation:
       - Ball absolute position (3 dims): X, Y, Z
       - Enemy absolute positions (4 dims): goalkeeper (X,Y), center-back (X,Y)
       - Ally absolute positions (4 dims): Ally1 (X,Y), Ally2 (X,Y)
       - Ego direction (2 dims): X, Y
       - Last action summary numeric (1 dim): weighted sum encoding last tactical action importance
       - Center-midfielder identity (3 dims)
       - Sender ID one-hot (3 dims)
       = 3 + 4 + 4 + 2 + 1 + 3 + 3 = 20 dims per message.

    5. Why This Matters:
       - Absolute positional info of ball and agents improves global situational awareness.
       - Center-midfielder flags help interpret positional data correctly.
       - Numeric last action summary helps temporal alignment and prediction of intentions.
       - This design balances compactness and sufficiency, focusing on novel info not inferable locally.

    6. Communication Protocol:
       - Broadcast messages.
       - Each agent receives concatenated messages from other two agents (total 40 dims received).
       - Output shape: (batch_size, 3, 48 + 40 = 88).

    Summary:
    This message design explicitly shares absolute global spatial cues and compact behavioral intentions,
    addressing prior shortcomings in absolute position predictability and temporal alignment,
    thus enhancing coordination and decision-making under partial observability.

    """
    return message_design_instruction.__doc__


def communication(o: th.Tensor) -> th.Tensor:
    """
    Arguments:
        o: Observation tensor of shape (batch_size=32, n_agents=3, obs_dim=48)

    Returns:
        Tensor of shape (batch_size, 3, 48 + 30) with appended messages from other agents.
    """
    device = o.device
    batch_size, num_agents, obs_dim = o.shape
    assert num_agents == 3 and obs_dim == 48, "Input tensor must have shape (batch, 3, 48)"

    # Extract ego absolute position (X, Y): (batch, 3, 2)
    ego_abs_pos = o[:, :, 0:2]

    # Extract ally relative positions (Ally1 and Ally2): (batch, 3, 4)
    allies_rel_pos = o[:, :, 2:6]

    # Extract enemy relative positions (Goalkeeper and Center-back): (batch, 3, 4)
    enemies_rel_pos = o[:, :, 12:16]

    # Extract ball relative position (X, Y): (batch, 3, 2)
    ball_rel_pos = o[:, :, 20:22]

    # Extract ball absolute position Z (batch, 3, 1)
    ball_abs_pos_z = o[:, :, 22:23]

    # Extract last action one-hot (19 dims: 26 to 44)
    last_action_oh = o[:, :, 26:45]  # (batch, 3, 19)

    # Extract center-midfielder identity flags (3 dims): (batch, 3, 3)
    cm_flags = o[:, :, 45:48]

    # --- Compute absolute positions of ball and entities per agent perspective ---

    # Ball absolute position XY = ego_abs_pos + ball_rel_pos
    ball_abs_pos_xy = ego_abs_pos + ball_rel_pos  # (batch, 3, 2)
    ball_abs_pos = th.cat([ball_abs_pos_xy, ball_abs_pos_z], dim=2)  # (batch, 3, 3)

    # Enemy absolute positions (goalkeeper and center-back)
    # Goalkeeper absolute pos = ego_abs_pos + goalkeeper relative pos
    goalkeeper_abs_pos = ego_abs_pos + enemies_rel_pos[:, :, 0:2]  # (batch, 3, 2)
    centerback_abs_pos = ego_abs_pos + enemies_rel_pos[:, :, 2:4]  # (batch, 3, 2)
    enemy_abs_pos = th.cat([goalkeeper_abs_pos, centerback_abs_pos], dim=2)  # (batch, 3, 4)

    # Ally absolute positions (Ally1 and Ally2)
    ally1_abs_pos = ego_abs_pos + allies_rel_pos[:, :, 0:2]  # (batch, 3, 2)
    ally2_abs_pos = ego_abs_pos + allies_rel_pos[:, :, 2:4]  # (batch, 3, 2)
    ally_abs_pos = th.cat([ally1_abs_pos, ally2_abs_pos], dim=2)  # (batch, 3, 4)

    # --- Last Action Summary Numeric Encoding ---
    # We encode last action as a weighted sum reflecting tactical importance and temporal relevance.
    # Define weights for key last actions to produce a single scalar encoding:
    # Dribble (43), LongPass(35), HighPass(36), ShortPass(37), Shot(38), Sprint(39)
    # Index mapping in last_action_oh: 26 to 44, so:
    # Dribble = index 43 - 26 = 17
    # LongPass = 35 - 26 = 9
    # HighPass = 36 - 26 = 10
    # ShortPass = 37 - 26 = 11
    # Shot = 38 - 26 = 12
    # Sprint = 39 - 26 = 13

    weights = th.tensor([0.0]*19, device=device)
    weights[17] = 1.0   # Dribble
    weights[9] = 0.8    # LongPass
    weights[10] = 0.8   # HighPass
    weights[11] = 0.8   # ShortPass
    weights[12] = 1.2   # Shot (higher importance)
    weights[13] = 0.5   # Sprint (lower importance)

    # (batch, 3, 19) * (19,) -> (batch, 3, 19)
    weighted_actions = last_action_oh * weights.view(1, 1, -1)

    # Sum weighted actions to get a single scalar per agent
    last_action_scalar = weighted_actions.sum(dim=2, keepdim=True).clamp(0, 1)  # (batch, 3, 1)

    # --- Sender Identity (Removed, duplicated in comm_init) ---
    # identity_mat = th.eye(num_agents, device=device)
    # sender_id = identity_mat.unsqueeze(0).repeat(batch_size, 1, 1)

    # --- Construct message per agent (15 dims) ---
    # Order:
    # Ball absolute pos (3)
    # Enemy absolute pos (4)
    # Ally absolute pos (4)
    # Last action scalar (1)
    # Center-midfielder flags (3)

    messages = th.cat([
        ball_abs_pos,      # 3
        enemy_abs_pos,     # 4
        ally_abs_pos,      # 4
        last_action_scalar,# 1
        cm_flags,          # 3
    ], dim=2)  # (batch, 3, 15)

    # --- Prepare received messages for each agent ---
    # Each agent receives messages from the other two agents concatenated (2*15=30 dims)

    # Mask to exclude self messages (False on diagonal)
    mask = (1 - th.eye(num_agents, device=device)).bool()  # (3,3)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)   # (batch, 3, 3)

    messages_exp = messages.unsqueeze(1).expand(-1, num_agents, -1, -1)  # (batch, receiver_agent, sender_agent, 15)
    mask_exp = mask.unsqueeze(-1)  # (batch, receiver_agent, sender_agent, 1)

    masked_messages = messages_exp * mask_exp.float()  # zero out self messages

    # Hardcoded other agents indices per receiver
    other_agents = th.tensor([[1, 2], [0, 2], [0, 1]], device=device)  # (3, 2)
    batch_idx = th.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, num_agents, 2)  # (batch,3,2)
    agent_idx = th.arange(num_agents, device=device).view(1, -1, 1).expand(batch_size, -1, 2)  # (batch,3,2)

    gathered_messages = masked_messages[batch_idx, agent_idx, other_agents.unsqueeze(0).expand(batch_size, -1, -1), :]  # (batch,3,2,15)

    # Concatenate messages from two other agents along last dim
    received_messages = gathered_messages.reshape(batch_size, num_agents, -1)  # (batch, 3, 30)

    # --- Final output: concatenate original obs with received messages ---
    messages_o = th.cat([o, received_messages], dim=2)  # (batch, 3, 48 + 30 = 78)

    return messages_o
