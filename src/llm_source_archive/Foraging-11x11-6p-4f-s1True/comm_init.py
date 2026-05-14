import torch as th

def message_design_instruction():
    """
    Message Design Instruction for LBF (grid obs) MARL Communication Protocol

    1. **Objective**: Under partial observability (3x3 local view), agents must coordinate to efficiently locate and collect food, avoiding redundant movement and collisions.
    
    2. **Key Information Not Locally Observable**:
        - Actual grid positions of food and other agents (beyond local 3x3).
        - Recent movement intent of other agents (to avoid collision, plan coordinated approaches).
    
    3. **Message Content Structure (per agent per time step):**
        - **Sender Identity**: 6-dim one-hot, indicating which agent sent the message. (Dims 0-5)
        - **Sender's Estimated Absolute Position**: 2-dim integer or normalized [x, y] in grid (if available; if not, use a placeholder or relative position).
            - *Assumption*: If absolute position is not observable, protocol omits this. If available, it's included for global mapping.
        - **Observed Food Locations (Local to Sender)**: 9-dim binary (food_layer[0,0] ... food_layer[2,2]); each entry 1 if food present in corresponding local cell.
        - **Observed Accessible Locations (Local to Sender)**: 9-dim binary (access_layer[0,0] ... access_layer[2,2]); helps others infer possible movement.
        - **Sender's Last Action**: 6-dim one-hot (last_action_0 ... last_action_5); helps others infer movement intent.
    
    4. **Total Message Dimension**: 
        - If absolute position is unavailable: 6 (sender id) + 9 (food) + 9 (access) + 6 (last action) = **30**
        - If absolute position is available: +2 = **32**
        - In LBF, agents may not know their absolute position, so **30** is chosen.
    
    5. **Communication Mode**: 
        - **Broadcast**: Each agent sends its message to all others. Each agent receives a message from every other agent (except itself), thus for 6 agents, each agent receives 5 messages per step.
        - For computational efficiency, all messages are stacked, and each agent's input is augmented with the concatenated messages from the other agents (excluding its own).
    
    6. **Why these fields?**
        - **Sender ID**: For grounding; agents can distinguish which message comes from which peer.
        - **Food Layer**: Maximally informative for coordination—agents can aggregate all food sightings to build a global food map.
        - **Access Layer**: Lets agents infer movement possibilities of others, essential for avoiding deadlocks/collisions.
        - **Last Action**: Encodes movement intent, further reduces coordination ambiguity.
        - **Absolute Position**: If available, allows mapping of local observations to the global grid; otherwise omitted.
    
    7. **No Redundancy**: 
        - No agent repeats its own locally observable information—only receives others' views.
        - All fields are either not observable by the receiver or are critical for global reasoning.
    
    8. **Message Integration**:
        - For each agent, the concatenated messages from other agents (5x30=150 dims) are appended to its original 39-dim observation.
        - **Final Enhanced Observation Shape**: (batch, agents, 39 + 5*30 = 189)
    """
    return (
        "Each agent broadcasts a 30-dimensional message composed of: "
        "[6-dim sender one-hot] + [9-dim food_layer (local)] + [9-dim access_layer (local)] + [6-dim last_action (one-hot)]. "
        "Each agent receives the messages from all other agents (excluding itself), concatenates them in agent-id order, "
        "and appends the resulting 150-dimensional vector to its own 39-dim observation, yielding a final observation vector "
        "of shape (batch, agents, 189). This protocol ensures agents share only critical, non-redundant information required "
        "for coordination under partial observability."
    )

def communication(o):
    """
    o: torch tensor, shape (batch, agents, 39)
    Returns: torch tensor, shape (batch, agents, 189)
    """
    # Device handling
    device = o.device
    batch_size, n_agents, obs_dim = o.shape
    assert obs_dim == 39 and n_agents == 6, "Unexpected input shape"
    
    # 1. Build message for each agent
    # Sender ID (6-dim one-hot)
    sender_ids = th.eye(n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, agents, 6)
    # Food layer: dims 9-17 (inclusive)
    food_layer = o[:, :, 9:18]   # (batch, agents, 9)
    # Access layer: dims 18-26 (inclusive)
    access_layer = o[:, :, 18:27]  # (batch, agents, 9)
    # Last action: dims 27-32 (inclusive)
    last_action = o[:, :, 27:33]   # (batch, agents, 6)
    # Concatenate to form message (batch, agents, 30)
    message = th.cat([sender_ids, food_layer, access_layer, last_action], dim=-1)  # (batch, agents, 30)
    
    # 2. For each agent, gather messages from other agents (NOT self)
    # We can vectorize this by masking or advanced indexing
    # First, build a mask for "other agents" (True iff i != j)
    mask = ~th.eye(n_agents, dtype=th.bool, device=device)  # (agents, agents)
    # For batch, expand mask to (batch, agents, agents)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, agents, agents)
    # Expand message to (batch, agents, agents, 30): for each receiver (axis 1), all senders (axis 2)
    message_exp = message.unsqueeze(1).expand(-1, n_agents, -1, -1)  # (batch, agents, agents, 30)
    # Apply mask to select "other agents" for each receiver
    # For each agent i, get all messages from agents j != i, ordered by agent id
    # mask: (batch, agents, agents), message_exp: (batch, agents, agents, 30)
    # Output: (batch, agents, n_agents-1, 30)
    messages_from_others = message_exp[mask].view(batch_size, n_agents, n_agents-1, 30)
    # Reshape to (batch, agents, 150)
    messages_cat = messages_from_others.reshape(batch_size, n_agents, (n_agents-1)*30)
    
    # 3. Concatenate to original obs
    enhanced_obs = th.cat([o, messages_cat], dim=-1)  # (batch, agents, 189)
    return enhanced_obs
