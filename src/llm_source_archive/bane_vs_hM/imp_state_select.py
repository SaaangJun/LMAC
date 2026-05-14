import numpy

def select_important_state():
    # Step-by-step reasoning:
    #
    # 1. **Task Analysis**: The Banelings must arrive at the intersection (or target) simultaneously for a successful attack.
    #    - If any Baneling arrives early, it is eliminated—synchronization is critical.
    #    - The agents cannot see each other, so information about *all agents' positions* (X, Y) is vital for coordination,
    #      but not locally observable.
    #
    # 2. **State Variable Filtering**:
    #    - **Agent positions**: [s[2], s[3], s[8], s[9], s[14], s[15]] (absolute X/Y for all 3 agents).
    #    - **Agent health**: [s[0], s[6], s[12]] (if an agent is dead, coordination is impossible).
    #    - **Enemy Hydralisk position**: [s[19], s[20]] (target location, but likely fixed at intersection).
    #    - **Enemy Hydralisk health**: [s[18]] (for task completion, but less relevant for the coordination itself).
    #    - **Other features** (shields, weapon cooldowns, unit types, Medivac, etc.): mostly irrelevant for the coordination problem as described.
    #
    # 3. **Crucial for Coordination**:
    #    - **Agent positions** (s[2], s[3], s[8], s[9], s[14], s[15]): Needed to infer if agents will arrive together.
    #    - **Agent health** (s[0], s[6], s[12]): Dead agents can't coordinate; critical to know if all are alive.
    #
    # 4. **Optional** (less critical, but may help):
    #    - **Hydralisk position** (s[19], s[20]): If not fixed, needed for path planning.
    #    - **Hydralisk health** (s[18]): To check if task is already complete.
    #
    # 5. **Final selection**:
    #    - Core for coordination: agent positions and healths.
    #    - Optionally, include Hydralisk position if not fixed (but can be omitted if always known).
    #
    # Therefore, the most important state dimensions are:
    # [0] Agent 1 health
    # [2] Agent 1 X
    # [3] Agent 1 Y
    # [6] Agent 2 health
    # [8] Agent 2 X
    # [9] Agent 2 Y
    # [12] Agent 3 health
    # [14] Agent 3 X
    # [15] Agent 3 Y
    # [18] Hydralisk health
    # [19] Hydralisk X
    # [20] Hydralisk Y

    important_dims = [0, 2, 3, 6, 8, 9, 12, 14, 15, 18, 19, 20]
    # For pure coordination, the above 9 are sufficient

    return important_dims
