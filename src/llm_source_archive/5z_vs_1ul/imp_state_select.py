import numpy

def select_important_state():
    # Step-by-step reasoning:
    # 1. **Task Analysis**: Success in 5z_vs_1ul relies on *synchronized group engagement* (avoid staggered attacks),
    #    *agent survival*, and *efficient focus fire* on the Ultralisk.
    # 2. **Key Requirements**:
    #    - Agents must know *where* all teammates are (positions) to group up.
    #    - Agents must know *when* all teammates are ready to attack (weapon cooldowns).
    #    - Agents must be aware of their own and teammates' health/shield to avoid isolated weak units.
    #    - Agents need the enemy's position to coordinate approach and engagement.
    #    - Actions history is less important for *current* coordination, but could be useful for learning.
    #
    # 3. **Partial Observability**: 
    #    - Each agent can locally observe its own state and maybe a few neighbors, 
    #      but cannot globally observe all agents and the enemy.
    #    - *Global* positions, cooldowns, and health/shield are hard to infer without communication.
    #
    # 4. **Critical dimensions** (for communication/coordination):
    #    - All agents' (Zealots) positions: s[2], s[3], s[7], s[8], ..., s[22], s[23]
    #    - All agents' weapon cooldowns: s[1], s[6], s[11], s[16], s[21]
    #    - All agents' health: s[0], s[5], s[10], s[15], s[20]
    #    - All agents' shields: s[4], s[9], s[14], s[19], s[24]
    #    - Enemy (Ultralisk) position: s[26], s[27]
    #    - Enemy health: s[25]
    #
    # 5. **Not included**: Last actions (s[28:63]) are less critical for *immediate* coordination,
    #    unless used for intent inference or credit assignment.
    #
    # 6. **Summary**: The most important dimensions are those that allow all agents to
    #    synchronize movement and attacks, maximize survival, and focus fire.

    important_dims = []

    # Zealot agents' health, weapon cooldown, positions, and shields
    for i in range(5):
        agent_offset = i * 5
        important_dims.append(agent_offset + 0)  # health
        important_dims.append(agent_offset + 1)  # weapon_cooldown
        important_dims.append(agent_offset + 2)  # x
        important_dims.append(agent_offset + 3)  # y
        important_dims.append(agent_offset + 4)  # shield

    # Enemy Ultralisk's health and position
    important_dims.append(25)  # Ultralisk health
    important_dims.append(26)  # Ultralisk x
    important_dims.append(27)  # Ultralisk y

    # Final dimension list, in order:
    # [0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24, 25,26,27]
    # This covers all agents' health, cooldown, position, shield, and the enemy's health and position.

    return important_dims
