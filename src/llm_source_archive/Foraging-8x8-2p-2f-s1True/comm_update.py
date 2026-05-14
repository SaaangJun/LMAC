import torch as th

def message_design_instruction():
    """
    Enhanced Message Design for LBF (grid obs) Task
    --------------------------------------------------
    **Objective:**  
    Address the information bottleneck identified in prior communication methods by enabling agents to infer unpredictable agent and access layers, and to propagate knowledge about the environment beyond their instantaneous local field of view.  
    The new protocol focuses on sharing **agent-relative positional cues** and **behavioral intent**—information that is not directly observable by the peer, and that is essential for effective coordination and prediction under partial observability.

    **Message Structure (per agent, per timestep):**
    1. **Estimated Absolute Position (2D, normalized):**
       - Each agent shares its own absolute position (row, col) in the grid, **normalized to [0,1]** (e.g., [row/H, col/W]), if available in the environment.  
         If not directly available, the agent shares its *relative position estimate* (e.g., initial spawn or via odometry).
         This allows the peer to reconstruct the sender's position in the global grid, enabling inference of agent layer state outside local view and supporting spatial coordination.

    **Total Message Dimension:**  
    2 (position) = **2**

    **Communication Pattern:**  
    Peer-to-peer (each agent receives a message from the other agent in the same batch).

    **Rationale for Each Field:**
    - **Absolute Position:** Enables global reconstruction of the agent layer, crucial for coordination and for inferring out-of-view agent states.

    **Redundancy Minimization:**
    - *Agent layer* is not shared directly, but is made inferable via position and intent.

    **Protocol Summary:**
    - Each agent receives a peer message containing: absolute (or estimated) position.
    - Messages are concatenated to the original observation, outputting a tensor of shape (batch, n_agents, 39+2).

    **Extensibility:**
    - If absolute positions are not available, use relative positions or odometry-based estimates.

    --------------------------------------------------
    Message fields:
      [absolute_position (2D normalized)]
    Total Message Dimension: 2
    Peer-to-peer exchange per scenario.
    """

    return (
        "Enhanced Message Design for LBF Task:\n"
        "Each agent sends to its peer:\n"
        "1. Its own absolute (or estimated) position in the grid (2D, normalized to [0,1]).\n"
        "Total message dimension: 2.\n"
        "Messages are exchanged peer-to-peer (agent 0 receives from 1 and vice versa, per scenario).\n"
        "Redundant info (Sender ID, Movement Intent, Access Layer) removed.\n"
        "This design enables global inference of agent positions."
    )

def communication(o, grid_shape=(8,8)):
    """
    Implements peer-to-peer message exchange for LBF (grid obs) MARL scenario, with enhanced global positional signals.
    Each agent receives a message from its peer in the current scenario, with the following structure:
        [absolute_position (2D normalized)] = 2D

    Args:
        o: torch.Tensor, shape (batch, 2, 39)
        grid_shape: tuple (H, W), the global grid shape; used for position normalization.

    Returns:
        messages_o: torch.Tensor, shape (batch, 2, 41) = (batch, 2, 39+2)
    """
    device = o.device
    dtype = o.dtype
    batch = o.shape[0]
    n_agents = o.shape[1]
    H, W = grid_shape

    # 2. Absolute position (assume available in o[..., 33:35] as (row, col) if included; else, estimate)
    # For this example, we estimate absolute position by assuming each agent knows its spawn (0,0 or 0,1),
    # and can integrate its movement using last actions. If not feasible, replace with zeros or a placeholder.
    # For illustration, we'll set positions to zeros (agents must provide absolute positions in the environment).
    # In practice, use environment info or agent's internal odometry.

    # Placeholder: zeros (batch, 2, 2)
    abs_pos = th.zeros((batch, n_agents, 2), device=device, dtype=dtype)

    # Optionally: If absolute position is stored in observation (e.g., o[..., 39:41]), use that:
    # abs_pos = o[..., 39:41]

    # Normalize (if not already): divide by grid shape (H, W) to get [0,1] range
    abs_pos_norm = abs_pos.clone()
    abs_pos_norm[..., 0] = abs_pos[..., 0] / max(H-1, 1)
    abs_pos_norm[..., 1] = abs_pos[..., 1] / max(W-1, 1)

    # 3. Movement intent (Removed: subset of last action)
    # movement_intent = o[..., 27:32]  # (batch, 2, 5)

    # Concatenate message fields -> (batch, 2, 2)
    message = abs_pos_norm

    # Peer-to-peer: swap along agent axis (for 2 agents)
    peer_message = message.flip(1)  # (batch, 2, 2)

    # Concatenate received message to each agent's own observation
    messages_o = th.cat([o, peer_message], dim=-1)  # (batch, 2, 39+2)

    return messages_o
