import os
import json
import sys
import importlib.util
import traceback
import numpy as np
from .env_utils import EnvUtils
from .code_utils import CodeUtils
from .prompt_templates import PromptTemplates
from openai import OpenAI
import google.generativeai as gemini
from anthropic import Anthropic
import logging
import json
import datetime

logging.getLogger("openai").setLevel(logging.WARNING)


TOKEN_USAGE_LOG = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_tokens": 0,
    "total_retries": 0,
    "retries_by_phase": {
        "imp_state_selection": 0,
        "init_comm_generation": 0,
        "feedback_generation": 0,
        "comm_update_standard": 0,
        "comm_update_timestep": 0
    },
    "calls": []
}

def get_token_usage():
    return TOKEN_USAGE_LOG.copy()

def reset_token_usage():
    global TOKEN_USAGE_LOG
    TOKEN_USAGE_LOG = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_retries": 0,
        "retries_by_phase": {
            "imp_state_selection": 0,
            "init_comm_generation": 0,
            "feedback_generation": 0,
            "comm_update_standard": 0,
            "comm_update_timestep": 0
        },
        "calls": []
    }

def save_token_usage(save_path):
    with open(save_path, 'w') as f:
        json.dump(TOKEN_USAGE_LOG, f, indent=2)
    print(f"[TOKEN_USAGE] Saved to {save_path}")

    print(f"[TOKEN_USAGE] === Summary ===")
    print(f"[TOKEN_USAGE] Total retries: {TOKEN_USAGE_LOG['total_retries']}")
    for phase, count in TOKEN_USAGE_LOG['retries_by_phase'].items():
        if count > 0:
            print(f"[TOKEN_USAGE]   - {phase}: {count} retries")




class Communication:
    """
    Unified Communication class with separated prompt management
    Core functions: LLM_S, LLM_M_update, protocol_phase_0_step1/step2
    """
    
    def __init__(self, args, test_obs, temperature=0.6):
        self.args = args
        self.test_obs = test_obs
        self.temperature = temperature
        self.threshold = args.mse_thres
        
        self.env_utils = EnvUtils(args)
        self.code_utils = CodeUtils(args)
        self.prompt_templates = PromptTemplates()

        # Claude Configuration
        self.claude = Anthropic(api_key=self.args.claude_key)
        # GPT Configuration
        self.gpt = OpenAI(api_key=self.args.openai_key)

    def _call_llm(self, prompt, call_type="general", attempt=1, max_attempts=1):
        global TOKEN_USAGE_LOG
        
        is_retry = attempt > 1
        if is_retry:
            TOKEN_USAGE_LOG["total_retries"] += 1
            if call_type in TOKEN_USAGE_LOG["retries_by_phase"]:
                TOKEN_USAGE_LOG["retries_by_phase"][call_type] += 1
            else:
                TOKEN_USAGE_LOG["retries_by_phase"][call_type] = 1
        
        call_record = {
            "model": self.args.model,
            "call_type": call_type,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "is_retry": is_retry,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "timestamp": None,
            "success": True,
            "error": None
        }
        
        call_record["timestamp"] = datetime.datetime.now().isoformat()
        
        try:
            if self.args.model == "gemini-2.5-flash":
                gemini.configure(api_key=self.args.gemini_key)
                self.gemini_model = gemini.GenerativeModel("gemini-2.5-flash")
                resp = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature
                    }
                )
                
                if hasattr(resp, 'usage_metadata'):
                    call_record["input_tokens"] = getattr(resp.usage_metadata, 'prompt_token_count', 0)
                    call_record["output_tokens"] = getattr(resp.usage_metadata, 'candidates_token_count', 0)
                    call_record["total_tokens"] = getattr(resp.usage_metadata, 'total_token_count', 0)
                
                result = resp.text
                
            elif self.args.model == "claude-opus-4-20250514":
                resp = self.claude.messages.create(
                    model="claude-opus-4-20250514",
                    temperature=self.temperature,
                    max_tokens=3000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                
                if hasattr(resp, 'usage'):
                    call_record["input_tokens"] = getattr(resp.usage, 'input_tokens', 0)
                    call_record["output_tokens"] = getattr(resp.usage, 'output_tokens', 0)
                    call_record["total_tokens"] = call_record["input_tokens"] + call_record["output_tokens"]
                
                result = "".join(b.text for b in resp.content if b.type == "text")
                
            elif self.args.model in ["gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14", "o1-mini"]:
                kwargs = {
                    "model": self.args.model,
                    "messages": [{"role": "user", "content": prompt}],
                }

                if self.args.model not in {"o1", "o1-mini"}:
                    kwargs["temperature"] = self.temperature

                resp = self.gpt.chat.completions.create(**kwargs)
                
                if hasattr(resp, 'usage') and resp.usage is not None:
                    call_record["input_tokens"] = getattr(resp.usage, 'prompt_tokens', 0)
                    call_record["output_tokens"] = getattr(resp.usage, 'completion_tokens', 0)
                    call_record["total_tokens"] = getattr(resp.usage, 'total_tokens', 0)
                
                result = resp.choices[0].message.content
            else:
                raise NotImplementedError
                
        except Exception as e:
            call_record["success"] = False
            call_record["error"] = str(e)
            TOKEN_USAGE_LOG["total_input_tokens"] += call_record["input_tokens"]
            TOKEN_USAGE_LOG["total_output_tokens"] += call_record["output_tokens"]
            TOKEN_USAGE_LOG["total_tokens"] += call_record["total_tokens"]
            TOKEN_USAGE_LOG["calls"].append(call_record)
            raise e
        

        TOKEN_USAGE_LOG["total_input_tokens"] += call_record["input_tokens"]
        TOKEN_USAGE_LOG["total_output_tokens"] += call_record["output_tokens"]
        TOKEN_USAGE_LOG["total_tokens"] += call_record["total_tokens"]
        TOKEN_USAGE_LOG["calls"].append(call_record)
        

        retry_info = f" [RETRY {attempt}/{max_attempts}]" if is_retry else ""
        print(f"[TOKEN_USAGE]{retry_info} Type: {call_type} | Model: {self.args.model} | "
              f"Input: {call_record['input_tokens']} | "
              f"Output: {call_record['output_tokens']} | "
              f"Total: {call_record['total_tokens']} | "
              f"Cumulative: {TOKEN_USAGE_LOG['total_tokens']} | "
              f"Retries: {TOKEN_USAGE_LOG['total_retries']}")
        
        return result


    def LLM_S(self, results, imp_state=None, threshold=0.05, cur_communication_method=None, 
              timestep_wise=True, baseline_comparison=False, baseline_results=None):

        if not os.path.exists(self.code_utils.code_dir):
                os.makedirs(self.code_utils.code_dir, exist_ok=True)
        
        # Preprocess data using discriminator      
        try:
            if timestep_wise:
                analysis_data = self._process_timestep_data(results, baseline_comparison, baseline_results, imp_state)
                mode = "timestep"
                json_path = os.path.join(self.code_utils.code_dir, "feedback_data_summary_timewise.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            else:
                analysis_data = self._process_standard_data(results, baseline_comparison, baseline_results, imp_state)
                mode = "standard"
                json_path = os.path.join(self.code_utils.code_dir, "feedback_data_summary.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            # Generate prompt with phase-specific instruction 
            detail_content, task_description, total_dim = self.get_detail_content_and_task_desc(timewise=timestep_wise)
            obs_shape = f"({self.args.batch_size}, {self.args.n_agents}, {total_dim})"
            
            if timestep_wise:
                obs_shape = f"({self.args.batch_size}, T, {self.args.n_agents}, {total_dim})"
            
            prompt = self._create_analysis_prompt(
                analysis_data, task_description, detail_content, obs_shape, 
                cur_communication_method, mode
            )
            
            return prompt
            
        except Exception as e:
            print(f"Error in LLM_S: {e}")
            traceback.print_exc()
            return f"Error occurred: {str(e)}"

    def LLM_M_update(self, feedback, cur_communication_method, timestep_wise=False):
        batch_size = self.args.batch_size
        n_agents = self.args.n_agents

        detail_content, task_description, total_dim = self.get_detail_content_and_task_desc(timewise=timestep_wise)

        if timestep_wise:
            obs_shape = f"({batch_size}, T, {n_agents}, {total_dim}) - Contains information from the previous 10 steps up to the current step for each agent"
            obs_dim_desc = (
                f"- {batch_size}: Number of scenarios in the batch.\n"
                f"- T: Number of timesteps per episode.\n"
                f"- {n_agents}: Number of agents.\n"
                f"- {total_dim}: Number of observation dimensions per agent."
            )
            indexing_example = (
                "- o[2, 5, 0, :] = observation vector of agent 0 at timestep 5 in the 2nd batch\n"
                "- o[3, 4, 2, :] = observation vector of agent 2 at timestep 4 in the 3rd batch"
            )
            message_concat_axis = (
                "The output should be a tensor of shape (batch_size, n_agents, total_dim + message_dim). "
                "At each step, communication must aggregate the past 10 timesteps into a single representation for the current input, reducing (batch_size, T, n_agents, total_dim) to (batch_size, n_agents, total_dim + message_dim)."
            )
            timestep_additional_prompt = 'Your communication design should leverage temporal patterns across multiple timesteps, enabling agents to share novel, task-relevant information that enhances inference of weakly predictable state dimensions.'
        else:
            obs_shape = f"({batch_size}, {n_agents}, {total_dim})"
            obs_dim_desc = (
                f"- {batch_size}: Number of scenarios simultaneously processed.\n"
                f"- {n_agents}: Number of agents.\n"
                f"- {total_dim}: Number of observation dimensions per agent."
            )
            indexing_example = (
                "- o[2, 0, :] = observation vector of agent 0 in the 2nd batch\n"
                "- o[2, 2, :] = observation vector of agent 2 in the 2nd batch"
            )
            message_concat_axis = "The output should be a tensor of shape (batch_size, n_agents, total_dim + message_dim)."
            timestep_additional_prompt = ''

        task_additional_prompt = 'Caution!: 1. Each dimension in o is either a continuous value (0 to 1, normalized) or a categorical value (0 or 1, binary state), with Last action represented as a one-hot vector.'
        if self.args.message_dim_limit:
            additional_msg_prompt = f"The message dimension must not exceed {self.args.message_limit_dimension}. " \
                        f"Extract the most mission-critical information from the current state and encode it efficiently within this dimension to maximize communicative value."
        else:
            additional_msg_prompt = f""

        feedback = feedback +  "Current communication method: " + cur_communication_method
        
        return self.prompt_templates.get_protocol_update_prompt(
            task_description, detail_content, obs_shape, obs_dim_desc,
            indexing_example, message_concat_axis, timestep_additional_prompt,
            task_additional_prompt, additional_msg_prompt, feedback
        )

    def _process_timestep_data(self, results, baseline_comparison, baseline_results=None, imp_state=None):
        B, T, N, D = results.shape
        valid_mask = (results.abs().sum(dim=-1) != 0).float()
        filtered_names, _, _ = self.get_imp_state_names_and_units(imp_state)
        dimensions = []

        baseline_phase_data = {}
        if baseline_comparison and baseline_results is not None:
            baseline_valid_mask = (baseline_results.abs().sum(dim=-1) != 0).float()
            
            for dim_idx, state_idx in enumerate(imp_state):
                baseline_dim_data = baseline_results[:, :, :, dim_idx]  # [B, T, N]
                baseline_predictions = (baseline_dim_data < self.threshold).int()
                
                def get_baseline_phase_rates():
                    phase_results = {p: [] for p in ["early", "early-mid", "mid", "mid-late", "late"]}
                    
                    for b in range(B):
                        valid_len = int(baseline_valid_mask[b].sum(dim=0)[0].item())
                        if valid_len == 0:
                            continue
                        
                        p1 = round(valid_len * 0.2)
                        p2 = round(valid_len * 0.4)
                        p3 = round(valid_len * 0.6)
                        p4 = round(valid_len * 0.8)
                        
                        def baseline_phase_rate(start_t, end_t):
                            if start_t >= end_t:
                                return [0.0] * N
                            phase_mask = baseline_valid_mask[b, start_t:end_t, :]
                            phase_preds = baseline_predictions[b, start_t:end_t, :] * phase_mask.int()
                            counts = phase_mask.sum(dim=0)
                            counts[counts == 0] = 1.0
                            return (phase_preds.sum(dim=0) / counts).tolist()
                        
                        phase_results["early"].append(baseline_phase_rate(0, p1))
                        phase_results["early-mid"].append(baseline_phase_rate(p1, p2))
                        phase_results["mid"].append(baseline_phase_rate(p2, p3))
                        phase_results["mid-late"].append(baseline_phase_rate(p3, p4))
                        phase_results["late"].append(baseline_phase_rate(p4, valid_len))
                    
                    phase_avg = {}
                    phase_var = {}
                    for phase, vals in phase_results.items():
                        if not vals:
                            phase_avg[phase] = [0.0] * N
                            phase_var[phase] = 0.0
                        else:

                            phase_avg[phase] = [
                                round(sum(agent_vals) / len(agent_vals), 3)
                                for agent_vals in zip(*vals)
                            ]

                            agent_avg_rates = phase_avg[phase]
                            if len(agent_avg_rates) > 1:
                                mean_rate = sum(agent_avg_rates) / len(agent_avg_rates)
                                variance = sum((rate - mean_rate) ** 2 for rate in agent_avg_rates) / len(agent_avg_rates)
                                phase_var[phase] = round(variance, 4)
                            else:
                                phase_var[phase] = 0.0
                    
                    return phase_avg, phase_var
                
                baseline_avg, baseline_var = get_baseline_phase_rates()
                baseline_phase_data[state_idx] = {
                    'avg': baseline_avg,
                    'var': baseline_var
                }

        for dim_idx, state_idx in enumerate(imp_state):
            dim_name = filtered_names[dim_idx] if dim_idx < len(filtered_names) else f"dim_{state_idx}"

            dim_data = results[:, :, :, dim_idx]  
            predictions = (dim_data < self.threshold).int()
            def get_agent_phase_rates():
                phase_results = {p: [] for p in ["early", "early-mid", "mid", "mid-late", "late"]}

                for b in range(B):
                    valid_len = int(valid_mask[b].sum(dim=0)[0].item())
                    if valid_len == 0:
                        continue

                    p1 = round(valid_len * 0.2)
                    p2 = round(valid_len * 0.4)
                    p3 = round(valid_len * 0.6)
                    p4 = round(valid_len * 0.8)

                    def phase_rate(start_t, end_t):
                        if start_t >= end_t:
                            return [0.0] * N
                        phase_mask = valid_mask[b, start_t:end_t, :]  # [phase_len, N]
                        phase_preds = predictions[b, start_t:end_t, :] * phase_mask.int()
                        counts = phase_mask.sum(dim=0)                # [N]
                        counts[counts == 0] = 1.0
                        return (phase_preds.sum(dim=0) / counts).tolist()

                    phase_results["early"].append(phase_rate(0, p1))
                    phase_results["early-mid"].append(phase_rate(p1, p2))
                    phase_results["mid"].append(phase_rate(p2, p3))
                    phase_results["mid-late"].append(phase_rate(p3, p4))
                    phase_results["late"].append(phase_rate(p4, valid_len))

                phase_avg = {}
                phase_var = {}
                for phase, vals in phase_results.items():
                    if not vals:
                        phase_avg[phase] = [0.0] * N
                        phase_var[phase] = 0.0
                    else:
                        phase_avg[phase] = [
                            round(sum(agent_vals) / len(agent_vals), 3)
                            for agent_vals in zip(*vals)
                        ]
                        agent_avg_rates = phase_avg[phase]
                        if len(agent_avg_rates) > 1:
                            mean_rate = sum(agent_avg_rates) / len(agent_avg_rates)
                            variance = sum((rate - mean_rate) ** 2 for rate in agent_avg_rates) / len(agent_avg_rates)
                            phase_var[phase] = round(variance, 4)
                        else:
                            phase_var[phase] = 0.0
                
                return phase_avg, phase_var

            phase_avg, phase_var = get_agent_phase_rates()
            dimension_entry = {
                "dim": f"s[{state_idx}] - {dim_name}",
                "with_communication": {
                    "early": phase_avg["early"],
                    "variance_in_early": phase_var["early"],
                    "early-mid": phase_avg["early-mid"],
                    "variance_in_early-mid": phase_var["early-mid"],
                    "mid": phase_avg["mid"],
                    "variance_in_mid": phase_var["mid"],
                    "mid-late": phase_avg["mid-late"],
                    "variance_in_mid-late": phase_var["mid-late"],
                    "late": phase_avg["late"],
                    "variance_in_late": phase_var["late"]
                }
            }
            
            if baseline_comparison and state_idx in baseline_phase_data:
                baseline_data = baseline_phase_data[state_idx]
                dimension_entry["without_communication"] = {
                    "early": baseline_data['avg']["early"],
                    "variance_in_early": baseline_data['var']["early"],
                    "early-mid": baseline_data['avg']["early-mid"],
                    "variance_in_early-mid": baseline_data['var']["early-mid"],
                    "mid": baseline_data['avg']["mid"],
                    "variance_in_mid": baseline_data['var']["mid"],
                    "mid-late": baseline_data['avg']["mid-late"],
                    "variance_in_mid-late": baseline_data['var']["mid-late"],
                    "late": baseline_data['avg']["late"],
                    "variance_in_late": baseline_data['var']["late"]
                }
            else:
                dimension_entry["without_communication"] = {
                    "early": [0.0] * N,
                    "variance_in_early": 0.0,
                    "early-mid": [0.0] * N,
                    "variance_in_early-mid": 0.0,
                    "mid": [0.0] * N,
                    "variance_in_mid": 0.0,
                    "mid-late": [0.0] * N,
                    "variance_in_mid-late": 0.0,
                    "late": [0.0] * N,
                    "variance_in_late": 0.0
                }
            
            dimensions.append(dimension_entry)

        return {"dimensions": dimensions}

    def _process_standard_data(self, results, baseline_comparison, baseline_results=None, imp_state=None):
        filtered_names, _, _ = self.get_imp_state_names_and_units(imp_state)
        
        output_json = []
        threshold = self.threshold
        
        if hasattr(results, 'shape'):

            B, T, A, D = results.shape
            binary_predictions = (results < threshold).float()  # 
            valid_mask = (results != 0).float()
            binary_masked = binary_predictions * valid_mask
            valid_counts = valid_mask.sum(dim=(0, 1)) 
            valid_counts[valid_counts == 0] = 1.0 
            success_rates = binary_masked.sum(dim=(0, 1)) / valid_counts 
        else:
            squared_error = results.get('squared_error')
            B, T, A, D = squared_error.shape
            binary_predictions = (squared_error < threshold).float()
            valid_mask = (squared_error != 0).float()
            binary_masked = binary_predictions * valid_mask
            valid_counts = valid_mask.sum(dim=(0, 1))
            valid_counts[valid_counts == 0] = 1.0
            success_rates = binary_masked.sum(dim=(0, 1)) / valid_counts

        baseline_success_rates = None
        if baseline_results is not None:
            if hasattr(baseline_results, 'shape'):

                baseline_binary_predictions = (baseline_results < threshold).float()
                baseline_valid_mask = (baseline_results != 0).float()
                baseline_binary_masked = baseline_binary_predictions * baseline_valid_mask

                baseline_valid_counts = baseline_valid_mask.sum(dim=(0, 1))
                baseline_valid_counts[baseline_valid_counts == 0] = 1.0
                baseline_success_rates = baseline_binary_masked.sum(dim=(0, 1)) / baseline_valid_counts
            else:
                baseline_squared_error = baseline_results.get('squared_error')
                baseline_binary_predictions = (baseline_squared_error < threshold).float()
                baseline_valid_mask = (baseline_squared_error != 0).float()
                baseline_binary_masked = baseline_binary_predictions * baseline_valid_mask
                
                baseline_valid_counts = baseline_valid_mask.sum(dim=(0, 1))
                baseline_valid_counts[baseline_valid_counts == 0] = 1.0
                baseline_success_rates = baseline_binary_masked.sum(dim=(0, 1)) / baseline_valid_counts

        for d in range(len(imp_state)):
            agent_success_rates = [round(rate, 2) for rate in success_rates[:, d].tolist()]
            baseline_agent_success_rates = [0.0] * A  
            if baseline_success_rates is not None:
                baseline_agent_success_rates = [round(rate, 2) for rate in baseline_success_rates[:, d].tolist()]

            entry = {
                "dimension": f"s[..., {imp_state[d]}] ({filtered_names[d] if d < len(filtered_names) else f'dim_{imp_state[d]}'})",
                "with_communication": {
                    "agent_success_rates": agent_success_rates 
                },
                "without_communication": {
                    "agent_success_rates": baseline_agent_success_rates 
                }
            }
            output_json.append(entry)
        
        return output_json

    def _create_analysis_prompt(self, analysis_data, task_description, detail_content, 
                              obs_shape, cur_communication_method, mode):
        
        if self.args.env== "sc2" or self.args.env == "grf":
            task_additional_description = "Agent observations provide only relative, not absolute, coordinates and distances; thus, in environments with limited visibility of allies and enemies, agents should emphasize sharing self-perceived behavioral information such as movement possibilities and recent actions when available"
        else:
            task_additional_description = ""
            
        if mode == "timestep":
            phase_info = self.prompt_templates.get_step2_info()
            json_data = json.dumps(analysis_data.get("dimensions", analysis_data), indent=2)
            obs_example = f"- o[2, 5, 0, :] = observation vector of agent 0 at timestep 5 in the 2nd batch\n- o[3, 4, 2, :] = observation vector of agent 2 at timestep 4 in the 3rd batch"
            predictability_calc = "Each dimension shows agent prediction success rates across 5 episode phases (early/early-mid/mid/mid-late/late). Each phase contains agent-wise success rates as arrays like [0.12, 0.34, 0.53] where values range 0.0-1.0 (higher = better prediction accuracy). This reveals learning patterns and temporal adaptation across episode stages."
            timewise_additional_prompt = "Design communication messages that explicitly target the specific timesteps where prediction disagreements or failures occur, prioritizing the sharing of concrete, temporally aligned, and task-relevant features—especially those unobservable to individual agents—to maximize consistent inference of important state dimensions and enhance shared situational awareness."
            obs_tensor_desc = f"- {self.args.batch_size}: number of episodes per batch\n- T: number of timesteps per episode\n- {self.args.n_agents}: number of agents"
            next_k_input_data = f"In the next protocol generation, the message can be constructed using observations from a sequence of 10 timesteps."
            
            return self.prompt_templates.get_feedback_instruction_x_tilde(
                analysis_data, task_description, detail_content, obs_shape,
                obs_tensor_desc, obs_example, predictability_calc,
                timewise_additional_prompt, next_k_input_data,
                task_additional_description, cur_communication_method, json_data,
                phase_info=phase_info
            )
        else:
            phase_info = self.prompt_templates.get_step1_info()
            json_data = json.dumps(analysis_data, indent=2)
            obs_example = f"- o[2, 0, :] = observation vector of agent 0 in the 2nd batch\n- o[2, 2, :] = observation vector of agent 2 in the 2nd batch"
            predictability_calc = "Each dimension shows per-agent binary success based on an MSE threshold, with success rates obtained by averaging across batch and time, where values range from 0.0 to 1.0 (higher = better prediction accuracy); baseline no-communication comparison is included to highlight the benefits of communication."
            timewise_additional_prompt = ""
            obs_tensor_desc = f"- {self.args.batch_size}: number of episodes per batch\n- {self.args.n_agents}: number of agents"
            next_k_input_data = f"In the next protocol generation, the message can be constructed using the current observation"
            
            return self.prompt_templates.get_feedback_instruction_x_tilde(
                analysis_data, task_description, detail_content, obs_shape,
                obs_tensor_desc, obs_example, predictability_calc,
                timewise_additional_prompt, next_k_input_data,
                task_additional_description, cur_communication_method, json_data,
                phase_info=phase_info
            )
        

    def protocol_phase_0_step1_important_state(self, max_retries=15):

        if not os.path.exists(self.code_utils.code_dir):
            os.makedirs(self.code_utils.code_dir, exist_ok=True)
        
        # Generate important state reasoning prompt
        detail_content_state, task_description, total_dim = self.env_utils.get_state_detail_content_and_task_desc()
        base_prompt = self.prompt_templates.get_reasoning_prompt_z0(detail_content_state, task_description)
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Phase 0 Step 1] Important State - Attempt {attempt}/{max_retries}")
                
                # LLM call for important state reasoning z^(0)
                llm_response = self._call_llm(
                    base_prompt,
                    call_type="imp_state_selection",
                    attempt=attempt,
                    max_attempts=max_retries
                )
                
                # Extract important state function
                imp_state_code = self.code_utils.extract_code_block(llm_response)
                
                # Save important state function
                imp_state_path = os.path.join(self.code_utils.code_dir, f'imp_state_select.py')
                with open(imp_state_path, 'w') as f:
                    f.write(imp_state_code)
                
                # Validate important state output
                important_dims = load_imp_state(imp_state_path)
                
                print(f"[Phase 0 Step 1] SUCCESS - Important dims: {important_dims}")
                
                return imp_state_path, important_dims
                
            except Exception as e:
                print(f"[Phase 0 Step 1] Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    continue
                else:
                    print(f"❌ [Phase 0 Step 1] Failed after {max_retries} attempts")
                    raise e

    def protocol_phase_0_step2_init_comm(self, important_dims, max_retries=15):
        if not os.path.exists(self.code_utils.code_dir):
            os.makedirs(self.code_utils.code_dir, exist_ok=True)
        
        # Get task context for protocol generation
        batch_size = self.args.batch_size
        n_agents = self.args.n_agents
        detail_content, task_description, total_dim = self.env_utils.get_detail_content_and_task_desc()
        
        # Observation shape info
        obs_shape = f"({batch_size}, {n_agents}, {total_dim})"
        obs_dim_desc = (
            f"- {batch_size}: Number of scenarios simultaneously processed.\n"
            f"- {n_agents}: Number of agents.\n"
            f"- {total_dim}: Number of observation dimensions per agent."
        )
        indexing_example = (
            f"- o[2, 0, :] = observation vector of agent 0 in the 2nd batch\n"
            f"- o[2, 2, :] = observation vector of agent 2 in the 2nd batch"
        )
        
        if self.args.env== "sc2" or self.args.env == "grf":
            task_additional_description = " Agent observations provide only relative, not absolute, coordinates and distances; thus, in environments with limited visibility of allies and enemies, agents should emphasize sharing self-perceived behavioral information such as movement possibilities and recent actions when available. Each dimension in o is either a continuous value (0 to 1, normalized) or a categorical value (0 or 1, binary state), with Last action and Agent ID represented as a one-hot vector."
        else:
            task_additional_description = "Each dimension in o is either a continuous value (0 to 1, normalized) or a categorical value (0 or 1, binary state), with Last action and Agent ID represented as a one-hot vector."
        # Protocol generation instructions
        if self.args.message_dim_limit:
            additional_msg_prompt = f"The message dimension must not exceed {self.args.message_limit_dimension}. " \
                        f"Extract the most mission-critical information from the current state and encode it efficiently within this dimension to maximize communicative value."
        else:
            additional_msg_prompt = ""
        
        # Process important_dims - handle both formats (integers or descriptions)
        if important_dims and len(important_dims) > 0:
            if isinstance(important_dims[0], str) and "s[" in important_dims[0]:
                # Already formatted with descriptions
                important_dims_with_desc = important_dims
            else:
                # Convert integers to descriptions using get_imp_state_names_and_units
                filtered_names, _, _ = self.get_imp_state_names_and_units(important_dims)
                important_dims_with_desc = []
                for i, dim_idx in enumerate(important_dims):
                    dim_name = filtered_names[i] if i < len(filtered_names) else f"dim_{dim_idx}"
                    important_dims_with_desc.append(f"{dim_name},")
        else:
            important_dims_with_desc = []

        base_prompt = self.prompt_templates.get_input_prompt_x(
            important_dims_with_desc, task_description, task_additional_description,detail_content, obs_shape, 
            obs_dim_desc, indexing_example, additional_msg_prompt
        )
        
        code_file_path = os.path.join(self.code_utils.code_dir, 'comm_init.py')
        
        error_context = "" 
        for attempt in range(1, max_retries + 1):
            print(f"[comm_update] Try {attempt}/{max_retries} ...")
            stage = "llm_call"
            try:
                prompt = base_prompt + error_context

                llm_code = self._call_llm(
                    prompt,
                    call_type="init_comm_generation",
                    attempt=attempt,
                    max_attempts=max_retries
                )

                stage = "extract_code_block"
                code_block = self.code_utils.extract_code_block(llm_code)

                stage = "write_file"
                with open(code_file_path, 'w') as fp:
                    fp.write(code_block)
                print(f"[comm_update] LLM code written to: {code_file_path}")

                stage = "import_module"
                cur_module = self.code_utils.import_and_reload_module(code_file_path)

                stage = "validate_comm_function"
                message_dim = self.code_utils.validate_communication_function(cur_module, self.test_obs,)

                print(f"[Phase 0 Step 2] SUCCESS - Message dim: {message_dim}")
                    
                return code_file_path, cur_module, message_dim

            except Exception as e:
                comm_function_code = ""
                if 'code_block' in locals():
                    try:
                        import re
                        comm_pattern = r'def communication\(.*?\):.*?(?=def\s+\w+|\Z)'
                        comm_match = re.search(comm_pattern, code_block, re.DOTALL)
                        if comm_match:
                            comm_function_code = comm_match.group(0).strip()
                        else:
                            lines = code_block.split('\n')
                            comm_start = -1
                            for i, line in enumerate(lines):
                                if line.strip().startswith('def communication'):
                                    comm_start = i
                                    break
                            if comm_start >= 0:
                                comm_function_code = '\n'.join(lines[comm_start:])
                            else:
                                comm_function_code = code_block  # Fallback to full code_block
                    except Exception:
                        comm_function_code = code_block if 'code_block' in locals() else "Code extraction failed"
                
                error_context = "\n\n" + self.prompt_templates.get_error_augmentation_prompt(
                    comm_function_code, attempt, stage, e, self._short_tb()
                )
                
                try:
                    dbg = os.path.join(self.code_utils.code_dir, f"comm_update_attempt{attempt}_error.txt")
                    with open(dbg, "w") as f:
                        f.write(f"Stage: {stage}\n\n{self._short_tb()}\n\n")
                        if comm_function_code:
                            f.write("Communication function code:\n")
                            f.write(comm_function_code)
                    print(f"[comm_update] Saved debug to {dbg}")
                except Exception:
                    pass

        print("[comm_update] Failed after max retries.")
        return None, None, None

    def imp_state_generate(self, max_retries=15):
        if hasattr(self, '_phase_0_cache'):
            result = self._phase_0_cache
            important_dims = result['important_dims']
            filtered_names, _, _ = self.get_imp_state_names_and_units(important_dims)
            important_state_with_desc = []
            for i, dim_idx in enumerate(important_dims):
                dim_name = filtered_names[i] if i < len(filtered_names) else f"dim_{dim_idx}"
                important_state_with_desc.append(f"s[{dim_idx}] - {dim_name}")
            return result['imp_state_path'], important_state_with_desc
        
        try:
            imp_state_path, important_dims = self.protocol_phase_0_step1_important_state(max_retries)
            
            filtered_names, _, _ = self.get_imp_state_names_and_units(important_dims)
            important_state_with_desc = []
            for i, dim_idx in enumerate(important_dims):
                dim_name = filtered_names[i] if i < len(filtered_names) else f"dim_{dim_idx}"
                important_state_with_desc.append(f"s[{dim_idx}] - {dim_name}")
            
            return imp_state_path, important_dims
            
        except Exception as e:
            print(f"[imp_state_generate] Failed: {e}")
            raise e

    def init_comm_generate(self, max_retries=15, important_dims=None):
        if important_dims is not None:
            try:
                comm_path, comm_module, message_dim = self.protocol_phase_0_step2_init_comm(important_dims, max_retries)
                return comm_path, comm_module, message_dim
            except Exception:
                print("Step 2 failed, falling back to integrated approach")
        
        try:
            imp_state_path, important_dims = self.protocol_phase_0_step1_important_state(max_retries)
            comm_path, comm_module, message_dim = self.protocol_phase_0_step2_init_comm(important_dims, max_retries)
            
            self._phase_0_cache = {
                'imp_state_path': imp_state_path,
                'important_dims': important_dims,
                'comm_path': comm_path,
                'comm_module': comm_module,
                'message_dim': message_dim
            }
            
            return comm_path, comm_module, message_dim
            
        except Exception as e:
            print(f"[init_comm_generate] Failed: {e}")
            raise e

    def feedback_generate(self, feedback_data, imp_state, threshold=0.05, cur_communication_method=None, timestep_wise=False, max_retries=5, baseline_feedback_data=None):
        """Generate feedback using LLM_S analysis with optional baseline comparison
        Compatible with run_llm_final.py phase update functions
        """
        if not os.path.exists(self.code_utils.code_dir):
            os.makedirs(self.code_utils.code_dir, exist_ok=True)
        
        baseline_comparison = baseline_feedback_data is not None
        
        if baseline_comparison:
            prompt = self.LLM_S(
                results=feedback_data,
                imp_state=imp_state,
                threshold=threshold,
                cur_communication_method=cur_communication_method,
                timestep_wise=timestep_wise,
                baseline_comparison=True,
                baseline_results=baseline_feedback_data
            )
            if timestep_wise:
                print(f"[feedback_generate] Generated timestep-wise analysis with baseline comparison")
            else:
                print(f"[feedback_generate] Generated standard analysis with baseline comparison")
        else:
            prompt = self.LLM_S(
                results=feedback_data,
                imp_state=imp_state,
                threshold=threshold,
                cur_communication_method=cur_communication_method,
                timestep_wise=timestep_wise
            )
            if timestep_wise:
                print(f"[feedback_generate] Generated timestep-wise analysis without baseline")
            else:
                print(f"[feedback_generate] Generated standard analysis without baseline")
        
        if timestep_wise:
            feedback_path = os.path.join(self.code_utils.code_dir, 'feedback_timestep_wise.json')
        else:
            feedback_path = os.path.join(self.code_utils.code_dir, "comm_feedback.json")
            
        for attempt in range(1, max_retries+1):
            print(f"[feedback_generate] Try {attempt}/{max_retries} ...")
            try:
                feedback_text = self._call_llm(
                    prompt,
                    call_type="feedback_generation",
                    attempt=attempt,
                    max_attempts=max_retries
                )
                print("[feedback_generate] Feedback received.")
                with open(feedback_path, "w") as f:
                    f.write(feedback_text)
                return feedback_text, feedback_path
            except Exception as e:
                print(f"[feedback_generate] Error on attempt {attempt}:\n{traceback.format_exc()}")

        print("[feedback_generate] Failed after max retries.")
        return None, None

    def comm_update(self, feedback, cur_communication_method, timestep_wise=False, max_retries=15, phase=None):
        """Update communication method based on feedback
        Compatible with run_llm_final.py phase update functions
        """
        base_prompt = self.LLM_M_update(feedback, cur_communication_method, timestep_wise)

        if timestep_wise:
            code_file_path = os.path.join(self.code_utils.code_dir, f'comm_update_timestep_wise{phase}.py')
            call_type = f"comm_update_timestep_wise_{phase}"
            print(f"[comm_update] Generating timestep-wise communication update...")
        else:
            code_file_path = os.path.join(self.code_utils.code_dir, 'comm_update.py')
            call_type = "comm_update"
            print(f"[comm_update] Generating standard communication update...")

        error_context = "" 
        for attempt in range(1, max_retries + 1):
            print(f"[comm_update] Try {attempt}/{max_retries} ...")
            stage = "llm_call"
            try:
                prompt = base_prompt + error_context
                llm_code = self._call_llm(
                    prompt,
                    call_type=call_type,
                    attempt=attempt,
                    max_attempts=max_retries
                )

                stage = "extract_code_block"
                code_block = self.code_utils.extract_code_block(llm_code)

                stage = "write_file"
                with open(code_file_path, 'w') as fp:
                    fp.write(code_block)
                print(f"[comm_update] LLM code written to: {code_file_path}")

                stage = "import_module"
                cur_module = self.code_utils.import_and_reload_module(code_file_path)

                stage = "validate_comm_function"
                message_dim = self.code_utils.validate_communication_function(cur_module, self.test_obs, timestep_wise)
                print(f"[comm_update] message_dim: {message_dim}")
                
                if timestep_wise:
                    print(f"[comm_update] Timestep-wise update completed successfully")
                else:
                    print(f"[comm_update] Standard update completed successfully")
                    
                return code_file_path, cur_module, message_dim

            except Exception as e:
                # Extract only the communication function from code_block for error context
                comm_function_code = ""
                if 'code_block' in locals():
                    try:
                        import re
                        # Extract communication function definition
                        comm_pattern = r'def communication\(.*?\):.*?(?=def\s+\w+|\Z)'
                        comm_match = re.search(comm_pattern, code_block, re.DOTALL)
                        if comm_match:
                            comm_function_code = comm_match.group(0).strip()
                        else:
                            # Fallback: extract from "def communication" to end of code
                            lines = code_block.split('\n')
                            comm_start = -1
                            for i, line in enumerate(lines):
                                if line.strip().startswith('def communication'):
                                    comm_start = i
                                    break
                            if comm_start >= 0:
                                comm_function_code = '\n'.join(lines[comm_start:])
                            else:
                                comm_function_code = code_block  # Fallback to full code_block
                    except Exception:
                        comm_function_code = code_block if 'code_block' in locals() else "Code extraction failed"
                
                error_context = "\n\n" + self.prompt_templates.get_error_augmentation_prompt(
                    comm_function_code, attempt, stage, e, self._short_tb()
                )
                
                try:
                    dbg = os.path.join(self.code_utils.code_dir, f"comm_update_attempt{attempt}_error.txt")
                    with open(dbg, "w") as f:
                        f.write(f"Stage: {stage}\n\n{self._short_tb()}\n\n")
                        if comm_function_code:
                            f.write("Communication function code:\n")
                            f.write(comm_function_code)
                    print(f"[comm_update] Saved debug to {dbg}")
                except Exception:
                    pass

        print("[comm_update] Failed after max retries.")
        return None, None, None
    

    def get_imp_state_names_and_units(self, imp_state):
        return self.env_utils.get_imp_state_names_and_units(imp_state)
    
    def get_detail_content_and_task_desc(self, timewise=False):
        return self.env_utils.get_detail_content_and_task_desc(timewise)

    def get_state_detail_content_and_task_desc(self):
        return self.env_utils.get_state_detail_content_and_task_desc()
    
    def _short_tb(self, max_lines=25):
        tb = traceback.format_exc().splitlines()
        return "\n".join(tb[-max_lines:])


def load_imp_state(module_path):
    """Load important state dimensions from module file"""
    module_name = os.path.splitext(os.path.basename(module_path))[0] 
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, 'select_important_state'):
        raise AttributeError("No select_important_state() in LLM code")
    important_dims = module.select_important_state()
    important_dims = [i for x in important_dims for i in (x if isinstance(x, list) else [x])]
    if not isinstance(important_dims, (list, np.ndarray)):
        raise TypeError("select_important_state() did not return a list or np.ndarray")
    important_dims = list(map(int, important_dims))
    print(f"[imp_state_generate] important_dims: {important_dims}")
    return important_dims
