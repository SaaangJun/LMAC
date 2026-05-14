import numpy

def select_important_state():
    # Task-driven hypothesis reasoning:
    # 1. **Role Inference & Team Composition**:
    #    - Agents need to infer their own roles (damage dealer or support) and adapt strategies.
    #    - Unit type bits for all allies and enemies are critical for understanding team composition,
    #      which is not directly observable by a single agent.
    #    - These are: s[4:7], s[11:14], s[18:21], s[25:28], s[32:35] for allies,
    #      and s[38:41], s[44:47], s[50:53], s[56:59], s[62:65] for enemies.
    # 2. **Coordination based on Positions**:
    #    - Start positions are randomized ("surrounded" or "reflected").
    #    - Absolute positions of all units (allies & enemies) are important for coordination and tactics.
    #    - These are: s[2:4], s[9:11], s[16:18], s[23:25], s[30:32] for allies,
    #      and s[36:38], s[42:44], s[48:50], s[54:56], s[60:62] for enemies.
    # 3. **Health for Focus Fire & Support**:
    #    - Health values of all units are vital for targeting and healing decisions.
    #    - These are: s[0], s[7], s[14], s[21], s[28] for allies,
    #      and s[35], s[41], s[47], s[53], s[59] for enemies.
    # 4. **Cooldowns**:
    #    - Cooldowns are mainly relevant for tactical timing, but less critical for global coordination
    #      compared to the above. We'll deprioritize cooldowns as they're more locally observable.

    important_dims = []

    # Ally unit type bits
    for i in range(5):  # 5 allies
        important_dims.extend([4 + 7*i, 5 + 7*i, 6 + 7*i])

    # Enemy unit type bits
    for i in range(5):  # 5 enemies
        important_dims.extend([38 + 6*i, 39 + 6*i, 40 + 6*i])

    # Ally absolute positions (x, y)
    for i in range(5):
        important_dims.extend([2 + 7*i, 3 + 7*i])

    # Enemy absolute positions (x, y)
    for i in range(5):
        important_dims.extend([36 + 6*i, 37 + 6*i])

    # Ally health
    for i in range(5):
        important_dims.append(0 + 7*i)

    # Enemy health
    for i in range(5):
        important_dims.append(35 + 6*i)

    # Remove possible duplicates (shouldn't be any, but just in case)
    important_dims = sorted(set(important_dims))

    # In summary: 
    # - We prioritize unit type bits, absolute positions, and health for both allies and enemies,
    #   as these are hard to fully observe from a single agent's perspective but are critical for 
    #   coordinated strategy and role inference in this scenario.
    return important_dims
