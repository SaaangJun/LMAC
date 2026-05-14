import numpy

def select_important_state():
    # === Step-by-step reasoning ===
    # Task: Banelings (agents 1-10) must converge on and attack the Roach (enemy), but 
    # they do not know the Roach's or Overseer's (agent 11) absolute positions directly.
    # Only the Overseer (agent 11) can observe the Roach's position.
    #
    # For effective coordination:
    # - The Overseer must communicate both its own position (as reference) and the Roach's position.
    # - Banelings must know both to localize and converge on the Roach.
    #
    # Therefore, the state dimensions encoding:
    #   (a) The Overseer's absolute position      --> s[62] (X), s[63] (Y)
    #   (b) The Roach's (enemy) absolute position --> s[67] (X), s[68] (Y)
    # are the most important for communication and decision-making.
    #
    # Optionally, including the Overseer's and Roach's health can help (e.g., if the Roach is dead),
    # but for the core task of localization and convergence, positions are most critical.

    important_dims = [
        62, # Agent 11 (Overseer) absolute X
        63, # Agent 11 (Overseer) absolute Y
        67, # Enemy (Roach) absolute X
        68, # Enemy (Roach) absolute Y
    ]
    # Optionally, to be robust, add health info:
    # 60, # Agent 11 health
    # 66, # Enemy health

    # Brief explanation:
    # The Overseer's and Roach's absolute positions are crucial for the Banelings to localize and coordinate an attack.
    # These dimensions are not directly observable by Banelings, so they must be efficiently encoded and shared.
    # Health can be included for robustness but is not strictly necessary for localization.
    return important_dims
