import numpy

def select_important_state():
    # In protoss_5_vs_5 (SMACv2), agents must coordinate under partial observability.
    # Task success requires focusing on:
    # 1. **Ally status**: Each agent's own and allies' health, shield, position (x,y), and cooldown are vital for survival, focus fire, and support.
    # 2. **Enemy status**: Health, shield, and position of visible enemies are crucial for target selection and attack coordination.
    # 3. **Unit types**: In this symmetric scenario, unit types are less critical, as all units are likely the same (but we can include them if generalizing).
    # 4. **Cooldown**: Only allies' cooldowns are available (not enemies'), and are important for attack timing.

    # Hypothesis:
    # The following state dimensions are most important for coordinated combat:
    # - For each ally (including self): health, shield, cooldown, position (x, y)
    # - For each enemy: health, shield, position (x, y)
    # We ignore unit_type_bits for now, as in this scenario all units are symmetric.

    important_dims = []

    # Ally indices
    for i in range(5):  # 5 allies
        base = i * 8
        # health, cooldown, x, y, shield
        important_dims.extend([
            base + 0,  # health
            base + 1,  # cooldown
            base + 2,  # x
            base + 3,  # y
            base + 4,  # shield
        ])
        # (unit_type_bits ignored as all protoss units are the same in this symmetric scenario)

    # Enemy indices
    for i in range(5):  # 5 enemies
        base = 40 + i * 7
        # health, x, y, shield
        important_dims.extend([
            base + 0,  # health
            base + 1,  # x
            base + 2,  # y
            base + 3,  # shield
        ])
        # (enemy unit_type_bits ignored as all protoss units are the same in this symmetric scenario)

    # Explanation:
    # These dimensions are selected because:
    # - Health and shield are needed for focus fire, target selection, and survivability.
    # - Position (x, y) is essential for movement, spatial reasoning, and coordination.
    # - Cooldown (for allies) is vital for timing attacks and maximizing DPS.
    # - Other features (unit_type_bits) are less important in this symmetric scenario.
    return important_dims
