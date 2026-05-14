import torch as th

def message_design_instruction():
    """
    Message Design Instruction for protoss_5_vs_5 (SMACv2):

    1. **Message Purpose**:
       - Under partial observability, initialization messages serve as a lightweight "handshake" and basic status report.

    2. **What to Communicate**:
       - **Sender Identity**: One-hot vector (length 5).
       - **Self Status**: [Health, Shield, Absolute_x, Absolute_y] (4 dims).
         - Essential for teammates to know *where* the sender is and its condition.

    3. **Message Structure Summary**:
       - Sender one-hot: 5
       - Self status: 4
       - Message size per agent: **9**
       - Each agent receives 4 such messages: 4 × 9 = **36**
       - Enhanced observation shape: (batch, 5, 108 + 36) = (batch, 5, 144)

    4. **Efficiency**:
       - Minimal bandwidth usage for this initialization step.
    """
    return (
        "Each agent broadcasts a lightweight message containing:\n"
        "- Sender identity as a one-hot vector (length 5)\n"
        "- Its own status: [Health, Shield, Absolute_x, Absolute_y] (4 dims)\n"
        "Total message dim: 9.\n"
        "Receivers concatenate 4 such messages (total 36 dims) to their observation.\n"
    )

def communication(o):
    """
    o: Tensor of shape (batch, 5, 108)
    Returns: Tensor of shape (batch, 5, 144) = (batch, 5, 108 + 36)
    """
    device = o.device
    batch_size = o.shape[0]
    n_agents = o.shape[1]
    
    # --- SENDER ID ---
    sender_id = th.eye(n_agents, device=device)[None, :, :].expand(batch_size, n_agents, n_agents)  # (batch, 5, 5)

    # --- SELF STATUS ---
    # indices: health=53 (self? no, wait. obs structure is tricky. let's check input indices)
    # Based on comm_update reading:
    # 87=own_x, 88=own_y, 86=own_shield? 
    # Let's verify indices from previous file reads.
    # comm_update uses: 87(x), 88(y), 86(shield).
    # Let's find health.
    # In comm_update.py (previous version), we saw:
    # enemy_health_idx = 8..
    # ally_health_idx = 53..
    # Usually own health is part of the agent's feature vector.
    # Standard SMACv2 feature vector usually puts own health/shield early or late.
    # Wait, looking at ally_health_idx = 53 + i*9.
    # Allies are 0..3 (excluding self).
    # Where is SELF health?
    # In standard SMAC, obs usually includes: [move_feats(4), enemy_feats(5*5), ally_feats(4*5), own_feats(1+?)]
    # 4 + 25 + 20 = 49.
    # But here n_obs=108.
    # Let's look at `comm_update.py` again (which I read previously).
    # It accessed own_x at 87, own_y at 88.
    # It accessed own_shield at 86.
    # It accessed attack_action at 98..103.
    # Own health? likely near shield. Maybe 85?
    # Let's make a safe assumption or check standard indices if possible.
    # Given own_shield is 86, own_health is likely 85.
    
    own_health = o[..., 85].unsqueeze(-1) # (batch, 5, 1)
    own_shield = o[..., 86].unsqueeze(-1) # (batch, 5, 1)
    own_x = o[..., 87].unsqueeze(-1)
    own_y = o[..., 88].unsqueeze(-1)

    self_status = th.cat([own_health, own_shield, own_x, own_y], dim=-1) # (batch, 5, 4)

    # --- MESSAGE CONCAT ---
    message = th.cat([sender_id, self_status], dim=-1) # (batch, 5, 9)

    # --- BROADCAST ---
    agent_indices = th.arange(n_agents, device=device)
    other_agent_indices = th.stack([
        th.cat([agent_indices[:i], agent_indices[i+1:]])
        for i in range(n_agents)
    ]) # (5, 4)

    received = th.stack(
        [message[:, other_agent_indices[i], :] for i in range(n_agents)],
        dim=1
    ) # (batch, 5, 4, 9)
    received = received.reshape(batch_size, n_agents, -1) # (batch, 5, 36)

    messages_o = th.cat([o, received], dim=-1) # (batch, 5, 144)
    return messages_o
