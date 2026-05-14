import torch as th

def message_design_instruction():
    """
    Improved Message Design Instruction for LBF (grid obs):

    Purpose:
    - Enable agents to coordinate and avoid redundant behavior by sharing **absolute position**.
    - This information is not available to other agents from local observations and is critical for inferring global state, planning, and preventing collisions.

    What to Communicate:
    - **Absolute Position**: 2 floats (x, y), normalized to [0, 1] based on grid size.
    - **Carrying Food Flag** (optional): 1-dim binary indicating if the agent is currently carrying food.

    Communication Protocol:
    - **Broadcast**: Each agent broadcasts its message to all others.
    - Each agent receives messages from all other agents (excluding itself), ordered by sender ID.
    - For each agent, the 5 incoming messages are concatenated and flattened.
    - The message vector is concatenated to the original 39-dim observation.

    Rationale:
    - **Absolute position** enables agents to localize others, infer global food/agent distribution, and avoid collisions.
    - **Carrying food** (optional) supports division of labor and path planning.

    Efficiency:
    - All operations are vectorized, avoiding explicit Python loops for batch and agent dimensions.

    Message Structure (per message): [position(2), carrying_food(1)] = 3 dims (or 2 if no carrying_food)
    For 5 other agents: 5 x 3 = 15 dims per agent (or 10).
    Final output: (batch, 6, 39 + 15) = (batch, 6, 54)
    """
    return (
        "Each agent broadcasts a message containing:\n"
        "- Its absolute (x, y) position in the grid, normalized to [0,1] (2 dims)\n"
        "- (Optional) 1-dim flag if the agent is carrying food\n"
        "Messages are sent to all other agents (broadcast, 5 senders per agent),\n"
        "concatenated (skipping self), flattened, and appended to original observation.\n"
    )


def communication(o, pos_x_idx=0, pos_y_idx=1, carrying_food_idx=None):
    """
    Efficient message exchange for LBF (grid obs) under improved protocol.

    Args:
        o: torch.Tensor of shape (batch, 6, 39)
        pos_x_idx: Index in o[..., :] for agent's absolute x-position (float, normalized to [0,1])
        pos_y_idx: Index in o[..., :] for agent's absolute y-position (float, normalized to [0,1])
        carrying_food_idx: Index in o[..., :] for carrying food flag (binary), or None if not available
    Returns:
        messages_o: torch.Tensor of shape (batch, 6, 54) (assuming carrying_food present -> 3 dims per msg)
                     or (batch, 6, 49) (if no carrying_food -> 2 dims per msg)
    """
    device = o.device
    batch_size, n_agents, obs_dim = o.shape
    assert n_agents == 6 and obs_dim == 39, "Expected shape (batch, 6, 39)"
    
    # 1. Extract message fields for all agents
    pos_x = o[..., pos_x_idx:pos_x_idx+1]   # (batch, 6, 1)
    pos_y = o[..., pos_y_idx:pos_y_idx+1]   # (batch, 6, 1)
    

    if carrying_food_idx is not None:
        carrying_food = o[..., carrying_food_idx:carrying_food_idx+1]  # (batch, 6, 1)
        msg = th.cat([pos_x, pos_y, carrying_food], dim=-1)  # (batch, 6, 3)
    else:
        msg = th.cat([pos_x, pos_y], dim=-1)  # (batch, 6, 2)

    msg_dim = msg.shape[-1]

    # 2. Build all-to-all message matrix (excluding self)
    agent_indices = th.arange(n_agents, device=device)
    sender_indices = th.stack([
        th.cat([agent_indices[:i], agent_indices[i+1:]], dim=0)
        for i in range(n_agents)
    ], dim=0)  # (6, 5)

    batch_ar = th.arange(batch_size, device=device)[:, None, None]
    recv_ar = th.arange(n_agents, device=device)[None, :, None]
    send_ar = sender_indices[None, :, :]  # (1, 6, 5)

    batch_idx = batch_ar.expand(batch_size, n_agents, 5)
    recv_idx = recv_ar.expand(batch_size, n_agents, 5)
    send_idx = send_ar.expand(batch_size, n_agents, 5)

    messages_received = msg[batch_idx, send_idx, :]  # (batch, 6, 5, msg_dim)
    messages_received_flat = messages_received.reshape(batch_size, n_agents, -1)  # (batch, 6, 5*msg_dim)
    messages_o = th.cat([o, messages_received_flat], dim=-1)  # (batch, 6, 39+5*msg_dim)

    return messages_o
