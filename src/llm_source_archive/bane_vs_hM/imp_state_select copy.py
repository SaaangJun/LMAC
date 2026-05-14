import numpy

def select_important_state():
    """
    Explanation:
    The task requires three Banelings to simultaneously explode on the Hydralisk at the central junction.
    Key factors for success:
    - Precise positioning of each Baneling (agents 0,1,2) to coordinate the simultaneous strike.
      Thus, each agent's absolute X and Y coordinates are crucial.
    - The Hydralisk's position is the target location for the explosion, so its coordinates must also be known.
    - Health of the Hydralisk and the Medivac is indirectly important: 
      - Hydralisk health indicates if the attack succeeded.
      - Medivac health affects healing rate, but since Banelings cannot see health of all units and have limited view,
        the Medivac's exact health may be less critical to each individual agent's immediate action.
      Still, including Hydralisk health is helpful for coordination.
    - Agents' own health might influence their decision to attack or not (e.g., if very low health), so include agents' health.
    - Weapon cooldowns and shields are always zero, so they can be ignored.
    - Unit type is constant or categorical but not informative for agents (all Banelings same type, enemies distinct but fixed).
    
    Since agents have limited field of view and no direct health info of others, 
    sharing positions and own health is vital for coordination.
    
    Therefore, the selected dimensions are:
    - Each agent's health and absolute position (X,Y)
    - Hydralisk's health and absolute position (X,Y)
    
    We exclude shields, cooldowns, and unit types.
    Medivac info is less critical for immediate coordination.
    """

    # Indices chosen:
    # Agent 0 health: s[0]
    # Agent 0 X,Y: s[2], s[3]
    # Agent 1 health: s[6]
    # Agent 1 X,Y: s[8], s[9]
    # Agent 2 health: s[12]
    # Agent 2 X,Y: s[14], s[15]
    # Hydralisk health: s[18]
    # Hydralisk X,Y: s[19], s[20]

    important_dims = [
        0, 2, 3,
        6, 8, 9,
        12, 14, 15,
        18, 19, 20
    ]

    return important_dims
