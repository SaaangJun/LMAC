class PromptTemplates:

    @staticmethod
    def get_step1_info():
        return {
            'step_num': 1,
            'step_name': 'Recovery Enhancement',
            'step_goal': 'Ensure each agent can recover state dimensions accurately',
            'step_instruction': 'Individual agent enhancement through improved communication',
            'specific_instruction': """Considering the provided feedback, you should refine the communication approach to reduce uneven prediction across agents and timesteps. When certain agents can infer a state dimension while others cannot, this asymmetry must be explicitly addressed.""",
            'enhancement_goals': [
                'Improve individual agent prediction accuracy for critical state dimensions',
                'Ensure each agent receives information needed for better state inference',
                'Address agent-specific limitations in state dimension recognition'
            ],
            'analysis_logic': """
**Analysis Method (Phase 1 Focus - Recovery)**:
- **Baseline Comparison**: Compare 'with_communication' vs 'without_communication'. Identify dimensions where communication significantly increases 'agent_success_rates'.
- **Identify Ignorance**: Focus on dimensions where `agent_success_rates` are generally low (< 0.6) across ALL agents.
- **Isolate Weak Agents**: Identify specific agents that fail to recognize the state locally while others succeed.
"""
        }

    @staticmethod
    def get_step2_info():
        return {
            'step_num': 2,
            'step_name': 'Imbalance Mitigation',
            'step_goal': 'Achieve consistent state recovery across all agents',
            'step_instruction': 'Address inconsistencies and reduce information asymmetry',
            'specific_instruction': """You should improve upon the current communication approach by emphasizing the agents' positional information and behavioral patterns. You must address weakly predictable state dimensions that cannot be inferred from local observations.""",
            'enhancement_goals': [
                'If one agent can recover a state dimension, all agents should be able to do so consistently',
                'Address information imbalance (High Variance) across agents',
                'Focus on dimensions where agents show inconsistent prediction performance'
            ],
            'analysis_logic': """
**Analysis Method (Step 2 Focus - Imbalance Mitigation)**:
- **Analyze Variance (Crucial)**: Check the 'variance_in_[step]' fields. High variance indicates **Information Asymmetry** (Sharing Failure).
- **Identify Information Sources**: Look for dimensions where specific agents achieve high accuracy (>0.8) while others fail. These 'source' agents possess the critical observations that need to be broadcasted.
- **Synchronization Gap**: If 'without_communication' is low but 'with_communication' has high variance, the source agent knows the info but isn't sharing it effectively.
"""
        }


    @staticmethod
    def _get_description_I_T(task_description, obs_shape, obs_dim_desc, detail_content, important_dims=None, task_additional_description=""):
        """
        [Component 1] Task Description (I_T)
        Specifies cooperative goal, environment characteristics, and information structure.
        Includes Important State Dimensions (Task Information Structure).
        """
        state_description = ""
        if important_dims:
            state_description = f"""
**Reasoning Tokens - Important State Dimensions**:
Based on previous analysis, the following state dimensions were identified as critical:
{important_dims}
{task_additional_description}
These dimensions require inter-agent communication for effective coordination under partial observability.
"""

        return f"""
**Task Description and Environment Characteristics**:

**Task Description**:
{task_description}

**State Information**:
{state_description}

**Observation Information**:
- Observation tensor shape: {obs_shape}
- {obs_dim_desc}
- Each dimension meaning: {detail_content}

"""

    @staticmethod
    def _get_instruction_I_P(obs_shape, indexing_example, additional_msg_prompt):
        """
        [Component 2] Protocol Instruction (I_P)
        Specifies required input-output format and design objectives.
        """
        return f"""
**Protocol Design Instructions**:

**Communication Design Key Principles**:
1. **State Reconstruction & Knowledge Gap Bridging:**
- **Analyze Semantic Relationship**: Before designing, map local observations to global state variables to identify what is missing (Knowledge Gap).
- **Target Partial Observability**: In POMDPs, global states are hidden. Do not access them directly. Instead, share correlated local features that allow others to infer the missing context.

2. **Uniqueness, Sufficiency & Compactness**:
- Share only essential information not already known or easily inferred by others.
- Ensure sufficiency for coordination while strictly minimizing redundancy.

3. **Contextual and Interaction-Aware**:
- Prioritize self-perceived behavioral data (e.g., movement possibilities, recent actions) to compensate for partial visibility.

4. **Explicitness and Clarity**:
- Avoid abstraction; critical task information must be explicit and interpretable.

5. **Structured Output**:
- Output shape: ({obs_shape.split(',')[0].strip('(')}, {obs_shape.split(',')[1].strip()}, {obs_shape.split(',')[2].strip().rstrip(')')} + message_dim).

6. **Communication Protocol**:
- Messages must be transmitted to other agents, ensuring they receive the information.
- The received message is then appended to the recipient's observation vector.
- Do NOT include the message in the sender's own observation.

7. **Computational Efficiency**:
- No trainable components; minimize loops for batch efficiency.

**Observation Access Pattern**:
For example: {indexing_example}

**Protocol Requirements**: 
{additional_msg_prompt}

**Task**: Design a protocol using the identified critical dimensions to improve coordination and state recovery.

**Required Python Functions**:

1. `message_design_instruction()`:
- Returns a string explaining how the message aids global state reconstruction using critical dimensions.

2. `communication(o)`:
- Input: Observation `o`.
- Output: Enhanced observation with messages that reduce state uncertainty for other agents.
- Logic: Extract key information from `o` and format it for others' consumption.

**Constraints**:
- Executable, integration-ready Python code.
- **Vectorized operations only** (minimize for-loops) for efficiency.
- **Semantic Inference**: Analyze the semantic relationship between the required information and the available features in 'o'. You must exclusively utilize the existing features in 'o' to infer or approximate the target information, strictly prohibiting the assumption of any particular features that are not explicitly listed in the 'Observation Information'.

Let's think step by step. Below is an illustrative example of the expected output:

```python
import torch as th
def message_design_instruction():
    # Explain how this protocol aids state reconstruction via critical dimensions
    return message_description

def communication(o):
    # Implement protocol logic using vectorized operations.
    # Input o: {obs_shape}, Output: {obs_shape} + message_dim
    # Ensure device consistency.
    return messages_o
```
"""

    @staticmethod
    def get_reasoning_prompt_z0(detail_content_state, task_description):
        """
        [Step 0-1] Important State Reasoning Prompt (Generates z^(0))
        LLM acts as a 'Reasoning Agent' to select important states.
        Matches original: get_llm_d_prompt
        """
        return f"""
You are a reasoning agent designing an importance extractor function for multi-agent reinforcement learning.
=========================================================
The agents' task description is:
{task_description}
=========================================================
Important notes:
- The explanation of each state dimension is provided here:
{detail_content_state}
=========================================================
Your task:
Write a Python function named `select_important_state()` that:

**Task-driven Hypothesis (Initial Reasoning)**:  
- Based on the task description and the meaning of each state dimension, form an initial hypothesis about which dimensions are likely important for task success.  
- In this partially observable multi-agent environment, each agent only perceives a limited view of the global state. Therefore, dimensions that are hard to perceive individually but critical when inferred through inter-agent communication should be prioritized. These dimensions are assumed to contribute significantly to coordinated decision-making and ultimately to task success.

Let's think step by step. Below is an illustrative example of the expected output:

```python
import numpy
def select_important_state():
    # Your implementation here
    # A brief explanation as an in-code comment about why you selected these dimensions
    return important_dims  # e.g., [idx,...]
```
"""

    @staticmethod
    def get_input_prompt_x(important_dims, task_description, task_additional_description, detail_content, 
                           obs_shape, obs_dim_desc, indexing_example, additional_msg_prompt=""):
        """
        [Step 0-2] Input Prompt x (Combines I_T and I_P only)
        Strictly x = I_T + I_P as per paper.
        """
        I_T = PromptTemplates._get_description_I_T(
            task_description, obs_shape, obs_dim_desc, detail_content, 
            important_dims, task_additional_description
        )
        
        I_P = PromptTemplates._get_instruction_I_P(
            obs_shape, indexing_example, additional_msg_prompt
        )
        return f"""
You are a communication design agent for Multi-Agent Reinforcement Learning (MARL).
Your goal is to design a task-specific communication protocol that maximizes global state awareness and coordination efficiency.

=========================================================
[PART 1] Task Description (I_T)
=========================================================

{I_T}

=========================================================
[PART 2] Protocol Design Instruction (I_P)
=========================================================

{I_P}
"""

    @staticmethod
    def get_feedback_instruction_x_tilde(analysis_data, task_description, detail_content, obs_shape, 
                          obs_tensor_desc, obs_example, predictability_calc, 
                          timewise_additional_prompt, next_k_input_data, 
                          task_additional_description, previous_comm_protocol, json_data,
                          phase_info=None):
        """
        [Step k] Feedback Instruction x_tilde
        Guides the Analysis Agent to generate feedback c^(k).
        """
        I_T = PromptTemplates._get_description_I_T(task_description, obs_shape, obs_tensor_desc, detail_content)
        
        extras = []
        if timewise_additional_prompt and timewise_additional_prompt.strip():
            extras.append(timewise_additional_prompt.strip())
        if next_k_input_data and next_k_input_data.strip():
            extras.append(next_k_input_data.strip())
        if task_additional_description and task_additional_description.strip():
            extras.append(task_additional_description.strip())
        extras_block = "\n".join(extras)
    
        if phase_info is None:
            phase_info = PromptTemplates.get_phase1_info()

        goals = "\n".join(f"- {g}" for g in phase_info.get("enhancement_goals", []))
        
        phase_instruction_block = (
            f"**CURRENT PHASE CONTEXT**:\n"
            f"- Step: {phase_info['step_num']} ({phase_info['step_name']})\n"
            f"- Goal: {phase_info['step_goal']}\n"
            f"- Objective: {phase_info['step_instruction']}\n"
            f"- Specific Instruction: {phase_info['specific_instruction']}\n"
            f"- Focus Areas:\n{goals}\n{extras_block}"
        )

        criterion = (
            "**Important State Dimensions Performance**:\n"
            f"{json_data}\n\n"
            "**Evaluation Method**:\n"
            f"{predictability_calc}"
        )
        
        step_wise_analysis_part = f"""
You are an analysis agent tasked with improving communication strategies in a multi-agent reinforcement learning (MARL) system.
**Context**:
{I_T}

### STEP-WISE ANALYSIS INSTRUCTION

1. **Step Information & Guidelines**:
Conduct your analysis based on the objectives below.
{phase_instruction_block}
{phase_info['analysis_logic']}

2. **Previous Protocol Under Analysis**:
{previous_comm_protocol}

3. Performance Data Analysis**:
Analyze the discriminator results below based on the 'Guidelines' provided.
=========================================================
**Criterion**:
{criterion}
"""

        feedback_generation_part = f"""
### FEEDBACK GENERATION INSTRUCTION

Based on your analysis, generate the structured feedback.

**Expected Output Format (JSON)**:
Strictly output a single JSON object. Do not include markdown formatting (```json ... ```) outside the object if possible, or ensure it is clean.
{{
  "Evaluation": "Synthesize your analysis results. Explicitly mention the gaps identified (e.g., 'High Variance in Agent 0 pos', 'Low Mean Accuracy') using the Step {phase_info['step_num']} analysis method.",
  "Missing_Information_Hypothesis": "Hypothesis about what specific information (e.g., intent, location) is missing or inadequately communicated.",
  "Improvement_Suggestions": "Specific, actionable suggestions to modify the communication content/structure. These must directly address the identified gaps to achieve the {phase_info['step_name']} goal."
}}
"""
        return step_wise_analysis_part + feedback_generation_part

    @staticmethod 
    def get_protocol_update_prompt(task_description, detail_content, obs_shape, obs_dim_desc, 
                                   indexing_example, message_concat_axis, timestep_additional_prompt, 
                                   task_additional_prompt, additional_msg_prompt, feedback):

        I_T = PromptTemplates._get_description_I_T(task_description, obs_shape, obs_dim_desc, detail_content)
        I_P = PromptTemplates._get_instruction_I_P(obs_shape, indexing_example, additional_msg_prompt)
        
        return f"""
You are a communication design agent for Multi-Agent Reinforcement Learning (MARL).
Your goal is to design a task-specific communication protocol that allows agents to share only essential and non-redundant information to enhance coordination and decision-making.
Based on the task description and observation dimensions, identify which information should be exchanged and structure it to maximize task performance.

=========================================================
[PART 1] Task Description (I_T)
=========================================================

{I_T}

=========================================================
[PART 2] Protocol Design Instruction (I_P)
=========================================================

{I_P}

=========================================================
[PART 3] Feedback & Protocol Update Instruction
=========================================================
Here is the feedback from the previous communication protocol evaluation:
{feedback}

**Protocol Update Strategy**:
1. **Reflect Feedback**: Analyze the performance gaps identified above.
2. **Complement, Do Not Repeat**: The new message will be **concatenated** with the previous one.
   - **Constraint**: Do NOT include information that is already shared by the previous protocol.
   - **Action**: Design new message fields that provide **missing** or **refined** information (e.g., if 'Location' is already shared, add 'Velocity' or 'Intent').
3. **Design New Message**: Create a protocol that strictly extracts these complementary features from the local observation.

{timestep_additional_prompt}
{task_additional_prompt}

Let's think step by step. Below is an illustrative example of the expected output:

```python
import torch as th

def message_design_instruction():
    # Your message design instruction goes here
    return message_description

def communication(o):
    # input : {obs_shape}
    # Your communication implementation goes here
    # use same device as input to avoid CUDA/CPU mismatch
    # {message_concat_axis}
    # Strict Rule : Only concatenate new, non-overlapping fields into each agent’s observation; exclude any information already included in the previous protocol.
    return messages_o 
```
"""
    
    @staticmethod
    def get_error_augmentation_prompt(base_prompt, attempt, stage, exc, short_tb):
        return (
            f"{base_prompt}\n\n"
            f"---\n"
            f"[Retry context] Previous attempt #{attempt} FAILED\n"
            f"Stage: {stage}\n"
            f"Exception: {type(exc).__name__}\n"
            f"Details (last lines):\n```\n{short_tb}\n```\n"
            "Please fix the issue above. Output ONLY a single Python fenced block:\n"
            "```python\n# your fixed code\n```\n"
            "Requirements: provide functions `communication(o)` and `message_design_instruction()`; "
            "no trainable params; respect the required tensor shapes; avoid for-loops over batch/time dims."
        )
