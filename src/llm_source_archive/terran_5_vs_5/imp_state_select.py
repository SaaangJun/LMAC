import numpy

def select_important_state():
    # Step-by-step reasoning:
    # 1. **Task**: The goal is to win a symmetric 5v5 battle with partial observability.
    # 2. **Critical factors for success**:
    #    - Knowing **which units (allies/enemies) are alive** and their **current health** (to focus fire or protect weak units).
    #    - **Positions** (x, y) of allies and enemies (to coordinate movement, encircle, or retreat).
    #    - **Cooldowns** of allies (to coordinate attacks or know when to cover for each other).
    #    - **Unit types** may be less critical in this symmetric scenario (all units likely same type), but included if not guaranteed.
    # 3. **Partial observability**: Individual agents may not see all units, so **communication** is needed to share unseen but critical info—especially enemy positions/health.
    # 4. **Most important dimensions**:
    #    - All **health** and **positions** (x, y) for both allies and enemies (indices: health, x, y for all units).
    #    - All **ally cooldowns** (to coordinate actions).
    #    - **Unit type bits** can be omitted if all units are the same, but if not, include them for generality.

    important_dims = []

    # Allies: 5 units, each has health, cooldown, x, y (indices given in blocks of 7: 0-6, 7-13, ...)
    for i in range(5):
        base = i * 7
        important_dims.append(base + 0)  # health
        important_dims.append(base + 1)  # cooldown
        important_dims.append(base + 2)  # x
        important_dims.append(base + 3)  # y
        # If unit type bits are important (e.g., for asymmetric scenarios), uncomment:
        # important_dims.extend([base+4, base+5, base+6])

    # Enemies: 5 units, each has health, x, y (indices: 35 + 6*i, 36+6*i, 37+6*i, ...)
    for i in range(5):
        base = 35 + i * 6
        important_dims.append(base + 0)  # health
        important_dims.append(base + 1)  # x
        important_dims.append(base + 2)  # y
        # If unit type bits are important, uncomment:
        # important_dims.extend([base+3, base+4, base+5])

    # Explanation:
    # - Health and position (x, y) for all units are crucial for targeting, movement, and coordination.
    # - Ally cooldown is important for timing attacks and defense.
    # - Unit type bits are commented out but can be included for generalization.
    # - These dimensions are hard to fully observe locally and thus benefit most from communication.

    return important_dims
