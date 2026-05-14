import numpy

def select_important_state():
    """
    Identify the most important state dimensions for coordinated multi-agent RL in the described scenario.

    Step-by-step reasoning:
    1. **Task Analysis**:
        - The goal is for Banelings (agents 1-20) to coordinate and attack Roaches (Enemy1, Enemy2), but only Overseer agents (21,22) can directly observe the Roaches' exact positions.
        - Banelings cannot perceive Roach positions themselves, so the Overseers' job is to encode and communicate this information.
        - The critical bottleneck for cooperation and reward is thus the accurate, shared knowledge of Roach (Enemy1/2) positions.

    2. **State Dimension Mapping**:
        - Roach positions: Enemy1 (`s[133]`,`s[134]`), Enemy2 (`s[138]`,`s[139]`).
        - Overseer positions: Agent21 (`s[122]`,`s[123]`), Agent22 (`s[128]`,`s[129]`).
        - (Optionally, Overseer health/shield may be useful for robustness, but position is primary.)

    3. **Communication Relevance**:
        - Since Banelings are silent, the critical information to encode/transmit is the Roach (Enemy) position.
        - Overseer positions may be needed if message encoding is relative to self.

    4. **Action History**:
        - Agent last actions are less critical for the core communication, unless planning to infer intent or predict future state, which is secondary for this bottleneck.

    5. **Selection**:
        - **Must-have**: Enemy Roach positions: `s[133]`,`s[134]` (Enemy1 X,Y), `s[138]`,`s[139]` (Enemy2 X,Y).
        - **Optional**: Overseer positions: `s[122]`,`s[123]` (Agent21 X,Y), `s[128]`,`s[129]` (Agent22 X,Y).
        - (If only one Roach is relevant per Overseer, select accordingly.)

    6. **Return**: Return the indices as a list.

    """
    # Indices for Enemy1 (Roach 1) and Enemy2 (Roach 2) absolute X, Y coordinates
    important_dims = [
        133,  # Enemy1 absolute X
        134,  # Enemy1 absolute Y
        138,  # Enemy2 absolute X
        139,  # Enemy2 absolute Y
        # Optionally, Overseer positions (for relative encoding/decoding)
        122,  # Agent21 absolute X
        123,  # Agent21 absolute Y
        128,  # Agent22 absolute X
        129,  # Agent22 absolute Y
    ]
    # The first four are **critical** for task success (Roach positions);
    # the last four are useful if you want to encode Roach position relative to Overseer location.
    return important_dims
