import numpy

def select_important_state():
    """
    Task-driven Hypothesis:
    The central midfielder (agent holding the ball) must decide to dribble or pass to attackers on wings or center.
    Coordination requires awareness of:
    - Own position and movement (to decide dribbling or passing options)
    - Teammates' positions and directions (to identify good passing lanes and runs)
    - Enemy defender and goalkeeper positions and directions (to evaluate defensive pressure)
    - Ball position and direction (to understand ball control and potential pass/dribble trajectory)

    Since this is a multi-agent setting with partial observability,
    dimensions that are difficult for a single agent to perceive but critical for coordination should be prioritized.
    For example, teammates’ directions and enemy directions help infer intentions and potential movements.
    
    Therefore, the important dimensions selected are:
    - All three teammates' positions (s[0:6]) and directions (s[6:12]) to capture spatial and movement info.
    - Enemy goalkeeper and center-back positions (s[12:16]) and directions (s[16:20]) to assess defensive threat.
    - Ball position and direction (s[20:26]) for ball control and dynamics.
    
    This selection excludes redundant info (e.g., if the agent is the central midfielder1, its own position/direction might be always known),
    but here we include all teammates since coordination depends on all.
    """

    # Indices selected:
    # Teammates positions: s[0..5]
    teammates_positions = list(range(0, 6))
    # Teammates directions: s[6..11]
    teammates_directions = list(range(6, 12))
    # Enemy goalkeeper and center-back positions: s[12..15]
    enemy_positions = list(range(12, 16))
    # Enemy goalkeeper and center-back directions: s[16..19]
    enemy_directions = list(range(16, 20))
    # Ball position and direction: s[20..25]
    ball_info = list(range(20, 26))

    important_dims = teammates_positions + teammates_directions + enemy_positions + enemy_directions + ball_info
    return important_dims
