import argparse
import os
import time
import copy
import random
import pickle
import subprocess
import yaml
import copy
import wandb
from collections import deque
from multiprocessing import Process, set_start_method
import torch as th

from modules.meta.msg_converter_new import MsgConverter as MsgConverterNew
from modules.meta.discriminator_new import Discriminator as DiscriminatorNew
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from LLM.llm_core import load_imp_state
from LLM.llm_core import get_token_usage, reset_token_usage, save_token_usage

import time


PROJECT_ROOT = os.getcwd()
NAME = None 
ENV_CONFIG = None
MAP_NAME = None
ENV_KEY = None
EXTRA_ARGS = []
RUNNING_ALGORITHM_NAME = None
DATA_ROOT = None
BUFFER_DIR = None
TEST_BUFFER_DIR = None
T_MAX = 5050000
EPSILON_ANNEAL_TIME = 50000
TIMESTEP_WISE = True
MODEL = "gpt-4.1-2025-04-14"

OPENAI_KEY = "<your_openai_key>"
GEMINI_KEY = "<your_gemini_key>"
CLAUDE_KEY = "<your_claude_key>"

NUM_SEEDS = 5
GPU_IDS = [0, 1, 2, 3, 4]
MSE_THRES = None
META_LAMBDA = 1.0
RECON_LAMBDA = 1.0
MAX_DISC_STEPS = 1000 
USE_WANDB = True 
MSG_DIM_LIMIT = None
MESSAGE_LIMIT_DIMENSION = None
TIMESTEP_WISE_PHASES = 1 

def cleanup_flags(flag_paths):
    for path in flag_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"[Cleanup] Removed flag: {path}")

def load_merged_config():
    with open(os.path.join(PROJECT_ROOT, "src/config", "default.yaml"), "r") as f:
        base = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, "src/config", "algs", f"{CONFIG}.yaml"), "r") as f:
        alg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(PROJECT_ROOT, "src/config", "envs", f"{ENV_CONFIG}.yaml"), "r") as f:
        env = yaml.load(f, Loader=yaml.FullLoader)
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    merged = recursive_update(base, alg)
    merged = recursive_update(merged, env)
    return merged

def apply_extra_args_to_env_args(config_dict):
    extra_args = config_dict.get("extra_args", [])
    for arg in extra_args:
        if arg.startswith("env_args.") and "=" in arg:
            k, v = arg.split("=", 1)
            k = k.strip().replace("env_args.", "")
            v = v.strip()
            if v == "True":
                v = True
            elif v == "False":
                v = False
            else:
                try: v = int(v)
                except ValueError:
                    try: v = float(v)
                    except ValueError: pass
            config_dict["env_args"][k] = v

def check_buffer_availability(buffer_dir, required_files=10):
    if not os.path.exists(buffer_dir):
        return False
    files = [f for f in os.listdir(buffer_dir) if f.endswith('pkl')]
    return len(files) >= required_files

def train_discriminator(discriminator, buffer_dir, args, discriminator_name="original"):
    disc_losses = deque(maxlen=40)
    print(f"[{discriminator_name.upper()}] Starting discriminator training...")
    
    step_count = 0
    max_steps = getattr(args, 'max_disc_steps', MAX_DISC_STEPS)
    
    while step_count < max_steps:
        files = [f for f in os.listdir(buffer_dir) if f.endswith('pkl')]
        if len(files) < args.batch_size:
            print(f"[WAIT-{discriminator_name}] Not enough files to sample batch (have {len(files)}, need {args.batch_size})")
            time.sleep(2)
            continue
        
        sampled_files = random.sample(files, args.batch_size)
        batch = {"obs": [], "state": [], "mask": [], "actions_onehot": []}
        for fname in sampled_files:
            with open(os.path.join(buffer_dir, fname), "rb") as f:
                b = pickle.load(f)
                for k in batch.keys():
                    batch[k].append(b[k])
        
        disc_losses.append(discriminator.train(batch))
        step_count += 1
        print(f"[TRAIN-{discriminator_name}] Discriminator training... Step {step_count}/{max_steps} | Loss: {disc_losses[-1]:.6f}")
    
    print(f"[TRAIN-{discriminator_name}] Discriminator training completed ({max_steps} steps)")
    
    return disc_losses

def train_baseline_discriminator(discriminator, buffer_dir, args, discriminator_name="baseline"):
    disc_losses = deque(maxlen=40)
    print(f"[{discriminator_name.upper()}] Starting baseline discriminator training (without communication)...")
    
    step_count = 0
    max_steps = getattr(args, 'max_disc_steps', MAX_DISC_STEPS)
    while step_count < max_steps:
        files = [f for f in os.listdir(buffer_dir) if f.endswith('pkl')]
        if len(files) < args.batch_size:
            time.sleep(1)
            break
        
        sampled_files = random.sample(files, args.batch_size)
        batch = {"obs": [], "state": [], "mask": [], "actions_onehot": []}
        for fname in sampled_files:
            with open(os.path.join(buffer_dir, fname), 'rb') as f:
                b = pickle.load(f)
                for k in batch.keys():
                    batch[k].append(b[k])
        
        disc_losses.append(discriminator.baseline_train(batch))
        step_count += 1
        
        print(f"[TRAIN-{discriminator_name}] Baseline discriminator training... Step {step_count}/{max_steps} | Loss: {disc_losses[-1]:.6f}")
    
    print(f"[TRAIN-{discriminator_name}] Baseline discriminator training completed ({max_steps} steps)")
    
    return disc_losses

def evaluate_baseline_discriminator(discriminator, test_batch, save_dir, discriminator_name="baseline"):
    test_loss, squared_error = discriminator.baseline_evaluate(test_batch, save_dir)
    print(f"[EVAL-{discriminator_name}] Baseline state prediction loss (test): {test_loss:.6f}")
    return test_loss, squared_error

def evaluate_discriminator(discriminator, test_batch, save_dir, discriminator_name="original"):
    test_loss, squared_error = discriminator.evaluate(test_batch, save_dir)
    print(f"[EVAL-{discriminator_name}] State prediction loss (test): {test_loss:.6f}")
    return test_loss, squared_error

def timestep_wise_comm_phase(args, i, save_dir, important_state, test_batch, test_loss_baseline, prev_test_loss, prev_args_phase, prev_comm_paths):
    from run_llm_final import timestepwise_comm_update

    print("=" * 80)
    print(f"[Phase{i}] Timestep-wise Communication Generation & Final Validation")
    print("=" * 80)

    print(f"[Phase{i}-1] Generating timestep-wise communication with updated feedback...")
    print(f"[Phase{i}-1] Using feedback file: {save_dir}")

    start_time = time.time()

    comm_code_path_updated_timewise, feedback_path_timewise = timestepwise_comm_update(
        i, prev_comm_paths[-1], important_state, save_dir, prev_args_phase)
    
    end_time = time.time()
    comm_inference_time = end_time - start_time
    
    print(f"[Phase{i}-1] Timestep-wise communication generation time: {(comm_inference_time) / 60:.2f} minutes")
    print(f"[Phase{i}-2] Creating final multi-communication discriminator (all modules)...")
    multi_comm_paths_phase = prev_comm_paths + [comm_code_path_updated_timewise]
    msg_converter_phase = MsgConverterNew(args, multi_comm_paths_phase)

    converter_info_phase = msg_converter_phase.get_converter_info()
    print(f"[Phase{i}-2] Loaded {converter_info_phase['num_converters']} communication modules:")
    for j, (name, dim, path) in enumerate(zip(converter_info_phase['converter_names'],
                                              converter_info_phase['individual_dims'],
                                              converter_info_phase['paths'])):
        print(f"[Phase{i}-2]   {j + 1}. {name}: {dim} dims from {os.path.basename(path)}")
    print(f"[Phase{i}-2] Total combined message dimension: {converter_info_phase['total_dim']}")

    args_phase = copy.deepcopy(args)
    args_phase.obsmsg_dim = msg_converter_phase.obsmsg_dim

    discriminator_phase = DiscriminatorNew(args_phase, msg_converter_phase, discriminator_name=f"phase{i}")

    print(f"[Phase{i}-3] Training final discriminator for complete system validation...")
    start_time = time.time()
    disc_losses_phase = train_discriminator(discriminator_phase, BUFFER_DIR, args_phase, f"phase{i}")
    end_time = time.time()
    phase_train_time = end_time - start_time
    print(f"[Phase{i}-3] Phase{i} final discriminator training complete. Final loss: {disc_losses_phase[-1]:.6f}")
    print(f"[Phase{i}-3] Phase{i} discriminator training time: {(phase_train_time) / 60:.2f} minutes")

    test_loss_phase, squared_error_phase = evaluate_discriminator(
        discriminator_phase, test_batch, save_dir, f"phase{i}"
    )

    comm_effect_phase = test_loss_baseline - test_loss_phase
    comm_effect_pct_phase = (comm_effect_phase / test_loss_baseline) * 100 if test_loss_baseline > 0 else 0

    print(f"[Phase{i}] Performance comparison:")
    print(f"[Phase{i}]   Phase{i-1} (Updated): {prev_test_loss:.6f}")
    print(f"[Phase{i}]   Phase{i} (Final):   {test_loss_phase:.6f}")
    print(f"[Phase{i}]   Baseline: {test_loss_baseline:.6f}")

    return test_loss_phase, args_phase, discriminator_phase, multi_comm_paths_phase, comm_effect_phase, comm_effect_pct_phase


# ==========  LLM Communication Training Pipeline ==========
def main():
    print("="*80)
    print("[MAIN] Starting 3-Phase LLM Communication Training Pipeline")
    print("="*80)
    
    from run_llm_final import phase0_comm, phase1_comm_update_basic

    config_dict = load_merged_config()
    config_dict['name'] = NAME
    if ENV_CONFIG in ['sc2', 'sc2v2' ,'grf']:
        config_dict['env_args']['map_name'] = MAP_NAME
    else:
        config_dict['env_args']['key'] = ENV_KEY
    config_dict['t_max'] = T_MAX
    config_dict['epsilon_anneal_time'] = EPSILON_ANNEAL_TIME
    config_dict['model'] = MODEL
    config_dict['seed'] = 1234
    config_dict['env_args']['seed'] = 1234
    config_dict['device'] = "cuda" if th.cuda.is_available() else "cpu"
    config_dict['extra_args'] = EXTRA_ARGS
    config_dict['max_disc_steps'] = MAX_DISC_STEPS
    config_dict['use_wandb'] = USE_WANDB  
    config_dict['mse_thres'] = MSE_THRES
    config_dict['message_dim_limit'] = MSG_DIM_LIMIT
    config_dict['message_limit_dimension'] = MESSAGE_LIMIT_DIMENSION
    config_dict['buffer_dir'] = BUFFER_DIR 
    config_dict['test_buffer_dir'] = TEST_BUFFER_DIR
    config_dict['openai_key'] = OPENAI_KEY
    config_dict['gemini_key'] = GEMINI_KEY
    config_dict['claude_key'] = CLAUDE_KEY
    
    apply_extra_args_to_env_args(config_dict)

    from types import SimpleNamespace as SN
    args = SN(**config_dict)
    args.env = ENV_CONFIG

    print(f"[SETUP] Using individual wandb runs for each discriminator")

    runner = r_REGISTRY[args.runner](args=args, logger=Logger(None))
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info['obs_shape']
    args.obs_ext_dim = args.obs_shape + args.n_agents + args.n_actions

    print(f"[SETUP] Environment Info: {args.n_agents} agents, {args.n_actions} actions, obs_shape: {args.obs_shape}")

    for buf in [BUFFER_DIR, TEST_BUFFER_DIR]:
        if not os.path.exists(buf):
            raise FileNotFoundError(f"Buffer directory not found: {buf}. Cannot skip data collection without existing data.")
        files = [f for f in os.listdir(buf) if f.endswith('pkl')]
        if len(files) == 0:
            raise ValueError(f"Buffer directory is empty: {buf}. Cannot skip data collection without existing data.")
    print(f"[SETUP] Using existing buffer directories: {BUFFER_DIR}, {TEST_BUFFER_DIR}")

    save_dir = os.path.join(DATA_ROOT, "feedback_data")
    os.makedirs(save_dir, exist_ok=True)

    # ========== Phase0 : Important State Selection & Initial Communication Generation ==========
    print("=" * 80)
    print("[Phase0] Important State Selection & Initial Communication Generation")
    print("=" * 80)

    print("[Phase0-1] Generating important state and initial communication...")
    start_time = time.time()
    comm_code_path, important_state = phase0_comm(args)
    end_time = time.time()
    phase0_inference_time = end_time - start_time
    print(f"[Phase0-1] Communication module path: {comm_code_path}")
    print(f"[Phase0-1] Important state dimensions: {important_state} (count: {len(important_state)})")
    print(f"[Phase0-1] Important state and initial communication generation time: {(phase0_inference_time) / 60:.2f} minutes")
    
    print("[Phase0-2] Creating initial MsgConverter and Discriminator...")
    msg_converter_phase0 = MsgConverterNew(args, comm_code_path)
    args.obsmsg_dim = msg_converter_phase0.obsmsg_dim
    args.imp_state = important_state
    args.imp_state_dim = len(important_state)

    print("[Phase0-3] Data collection phase skipped (RL Collection Logic Removed) - Using existing buffer data...")
    if not check_buffer_availability(BUFFER_DIR, args.batch_size):
        raise ValueError(f"Insufficient data in buffer directory: {BUFFER_DIR}")
    if not check_buffer_availability(TEST_BUFFER_DIR, args.batch_size):
        raise ValueError(f"Insufficient data in test buffer directory: {TEST_BUFFER_DIR}")
    print(f"[Phase0-3]  Confirmed sufficient data in buffer directories")

    test_files = [f for f in os.listdir(TEST_BUFFER_DIR) if f.endswith('pkl')]
    test_batch_size = min(args.batch_size, len(test_files))
    sampled_test_files = random.sample(test_files, test_batch_size)
    test_batch = {k: [] for k in ["obs", "state", "mask", "actions_onehot"]}
    for file in sampled_test_files:
        with open(os.path.join(TEST_BUFFER_DIR, file), 'rb') as f:
            b = pickle.load(f)
            for k in test_batch.keys():
                test_batch[k].append(b[k])
    print(f"[SETUP] Prepared test batch with {len(sampled_test_files)} samples")

    print("[BASELINE] Training baseline discriminator once (without communication)...")
    discriminator_baseline = DiscriminatorNew(args, msg_converter_phase0, discriminator_name="baseline")
    
    start_time = time.time()

    disc_losses_baseline = train_baseline_discriminator(
        discriminator_baseline, BUFFER_DIR, args, "baseline"
    )

    end_time = time.time()
    baseline_train_time = end_time - start_time
    print(f"[BASELINE] Baseline discriminator training time: {(baseline_train_time) / 60:.2f} minutes")
    print(f"[BASELINE] Baseline discriminator training complete. Final loss: {disc_losses_baseline[-1]:.6f}")
    print("[BASELINE] Evaluating baseline performance (without communication)...")
    test_loss_baseline, squared_error_baseline = evaluate_baseline_discriminator(
        discriminator_baseline, test_batch, save_dir, "baseline"
    )
    print(f"[BASELINE] Global baseline test loss: {test_loss_baseline:.6f}")
    
    
    print("[Phase0-4] Training discriminator for initial communication validation...")
    discriminator_phase0 = DiscriminatorNew(args, msg_converter_phase0, discriminator_name="phase0_initial")

    start_time = time.time()

    disc_losses_phase0 = train_discriminator(discriminator_phase0, BUFFER_DIR, args, "phase0_initial")

    end_time = time.time()
    phase0_train_time = end_time - start_time

    print(f"[Phase0-4] Phase0 discriminator training time: {(phase0_train_time) / 60:.2f} minutes")
    print(f"[Phase0-4] Phase0 discriminator training complete. Final loss: {disc_losses_phase0[-1]:.6f}")

    print(f"[Phase0-4] Evaluating Phase0 discriminator...")
    test_loss_phase0, squared_error_phase0 = evaluate_discriminator(
        discriminator_phase0, test_batch, save_dir, "phase0_initial"
    )

    print(f"[Phase0]   Phase0 performance:")
    print(f"[Phase0]   Phase0 (Initial):     {test_loss_phase0:.6f}")
    print(f"[Phase0]   No-comm:    {test_loss_baseline:.6f}")
    print(f"[Phase0]   Phase0 Complete - Initial communication validated. Test loss: {test_loss_phase0:.6f}")

    # ========== Phase1: Communication Update Generation & Validation ==========
    print("=" * 80)
    print("[Phase2] Communication Update Generation & Validation")
    print("=" * 80)
    
    print("[Phase1-1] Generating communication update with LLM feedback...")

    start_time = time.time()
    comm_code_path_updated, feedback_path = phase1_comm_update_basic(
        comm_code_path, important_state, save_dir, args)
    end_time = time.time()
    phase1_inference_time = end_time - start_time
    
    print("[Phase1-2] Creating multi-communication discriminator (initial + update)...")
    multi_comm_paths_phase1 = [comm_code_path, comm_code_path_updated]
    msg_converter_phase1 = MsgConverterNew(args, multi_comm_paths_phase1)

    converter_info_phase1 = msg_converter_phase1.get_converter_info()
    print(f"[Phase1-2] Loaded {converter_info_phase1['num_converters']} communication modules:")
    for i, (name, dim, path) in enumerate(zip(converter_info_phase1['converter_names'], 
                                              converter_info_phase1['individual_dims'], 
                                              converter_info_phase1['paths'])):
        print(f"[Phase1-2]   {i+1}. {name}: {dim} dims from {os.path.basename(path)}")
    print(f"[Phase1-2] Total combined message dimension: {converter_info_phase1['total_dim']}")

    args_phase1 = copy.deepcopy(args)
    args_phase1.obsmsg_dim = msg_converter_phase1.obsmsg_dim

    discriminator_phase1 = DiscriminatorNew(args_phase1, msg_converter_phase1, discriminator_name="phase1_updated")

    print("[Phase1-3] Training discriminator for communication update validation...")

    start_time = time.time()

    disc_losses_phase1 = train_discriminator(discriminator_phase1, BUFFER_DIR, args_phase1, "phase1_updated")

    end_time = time.time()
    phase1_train_time = end_time - start_time

    print(f"[Phase1-3] Phase1 discriminator training complete. Final loss: {disc_losses_phase1[-1]:.6f}")

    test_loss_phase1, squared_error_phase1 = evaluate_discriminator(
        discriminator_phase1, test_batch, save_dir, "phase1_updated"
    )

    print(f"[Phase1] Performance comparison:")
    print(f"[Phase1]   Phase1 (Initial): {test_loss_phase0:.6f}")
    print(f"[Phase1]   Phase1 (Updated): {test_loss_phase1:.6f}")
    print(f"[Phase1]   No-comm: {test_loss_baseline:.6f}")
    print(f"[Phase1] Phase1 Complete - Communication update validated. Test loss: {test_loss_phase1:.6f}")
    print(f"[Phase1] Communication update inference time: {(phase1_inference_time) / 60:.2f} minutes") 
    print(f"[Phase1] Phase1 discriminator training time: {(phase1_train_time) / 60:.2f} minutes")


    # ========== Phase2: Timestep-wise Communication Generation & Validation ==========
    if TIMESTEP_WISE:
        timestep_wise_test_loss = []
        timestep_wise_args = []
        timestep_wise_disc = []
        timestep_wise_comm_effect = []
        timestep_wise_comm_effect_pct = []

        prev_test_loss = test_loss_phase1
        prev_args_phase = args_phase1
        com_code_paths = multi_comm_paths_phase1
        for i in range(2, 2 + TIMESTEP_WISE_PHASES):
            prev_test_loss, prev_args_phase, prev_disc_phase, com_code_paths, prev_comm_effect, prev_comm_effect_pct = timestep_wise_comm_phase(args, i, save_dir, important_state, test_batch, test_loss_baseline, prev_test_loss, prev_args_phase, com_code_paths)
            timestep_wise_test_loss.append(prev_test_loss)
            timestep_wise_args.append(prev_args_phase)
            timestep_wise_disc.append(prev_disc_phase)
            timestep_wise_comm_effect.append(prev_comm_effect)
            timestep_wise_comm_effect_pct.append(prev_comm_effect_pct)

    print("=" * 80)
    print("[SUMMARY] 2-Phase Training Pipeline Complete")
    print("=" * 80)
    
    print(f"[SUMMARY] Phase0 (Initial comm):    {test_loss_phase0:.6f}")
    print(f"[SUMMARY] Phase1 (Updated comm):    {test_loss_phase1:.6f}")
    if TIMESTEP_WISE:
        for i in range(TIMESTEP_WISE_PHASES):
            print(f"[SUMMARY] Phase{i+2} (Final comm):      {timestep_wise_test_loss[i]:.6f}")
    print(f"[SUMMARY] Baseline (Global):        {test_loss_baseline:.6f}")

    final_test_loss = timestep_wise_test_loss[-1] if TIMESTEP_WISE else test_loss_phase1
    final_baseline_loss = test_loss_baseline 
    final_comm_effect = final_baseline_loss - final_test_loss

    token_usage = get_token_usage()
    print("=" * 80)
    print("[TOKEN_USAGE] Final Token Usage Summary")
    print("=" * 80)
    print(f"[TOKEN_USAGE] Total Input Tokens:  {token_usage['total_input_tokens']}")
    print(f"[TOKEN_USAGE] Total Output Tokens: {token_usage['total_output_tokens']}")
    print(f"[TOKEN_USAGE] Total Tokens:        {token_usage['total_tokens']}")
    print(f"[TOKEN_USAGE] Total LLM Calls:     {len(token_usage['calls'])}")
    
    token_usage_path = os.path.join(save_dir, "token_usage.json")
    save_token_usage(token_usage_path)
    print(f"[TOKEN_USAGE] Token usage details saved to: {token_usage_path}")

    print("=" * 80)
    print("[SUMMARY] Time Summary")
    print("=" * 80)

    total_time_all = phase0_inference_time + baseline_train_time + phase0_train_time + phase1_inference_time + phase1_train_time
    if TIMESTEP_WISE:
        total_time_all += sum([phase1_inference_time for _ in range(TIMESTEP_WISE_PHASES)])
    
    print(f"[TIME] Phase0 Inference Time:        {(phase0_inference_time) / 60:.2f} minutes")
    print(f"[TIME] Baseline Discriminator Train: {(baseline_train_time) / 60:.2f} minutes")
    print(f"[TIME] Phase0 Discriminator Train:   {(phase0_train_time) / 60:.2f} minutes")
    print(f"[TIME] Phase1 Inference Time:       {(phase1_inference_time) / 60:.2f} minutes")
    print(f"[TIME] Phase1 Discriminator Train:  {(phase1_train_time) / 60:.2f} minutes")
    if TIMESTEP_WISE:
        for i in range(TIMESTEP_WISE_PHASES):
            print(f"[TIME] Phase{i+2} Inference Time:       {(phase1_inference_time) / 60:.2f} minutes")
    print(f"[TIME] Total Time for All Phases:   {(total_time_all) / 60:.2f} minutes")   

    wandb.finish() 
    
    print("=" * 80)
    print("[WANDB] Finishing all discriminator wandb runs...")
    print("=" * 80)
    
    if hasattr(discriminator_baseline, 'finish_wandb'):
        discriminator_baseline.finish_wandb()
    
    if hasattr(discriminator_phase0, 'finish_wandb'):
        discriminator_phase0.finish_wandb()
    
    if hasattr(discriminator_phase1, 'finish_wandb'):
        discriminator_phase1.finish_wandb()
    
    if TIMESTEP_WISE:
        for i in range(TIMESTEP_WISE_PHASES):
            if hasattr(timestep_wise_disc[i], 'finish_wandb'):
                timestep_wise_disc[i].finish_wandb()
    
    print("[WANDB] All discriminator wandb runs finished.")
    print("[SUMMARY] Ready for multi-seed RL training with optimized communication.\n")
    return com_code_paths, important_state, save_dir

# ========== Final Training: Communication MARL ==========
def final_parallel_train(com_code_paths, important_state):
    print("=" * 80)
    print("[FINAL] Multi-Seed RL Training with Optimized Communication")
    print("=" * 80)
    
    done_flag_paths = []
    procs = []

    for idx in range(NUM_SEEDS):
        name_idxed = f"LMAC_Final_seed{idx}"
        result_dir = os.path.join(DATA_ROOT, name_idxed)
        os.makedirs(result_dir, exist_ok=True)
        done_flag_path = os.path.join(result_dir, "done.flag")
        done_flag_paths.append(done_flag_path)
        cuda_id = GPU_IDS[idx % len(GPU_IDS)]
        run_name = f"({MAP_NAME}){NAME}" if MAP_NAME else f"({ENV_KEY}){NAME}"
        base_cmd = (
            f"cd '{PROJECT_ROOT}' && "
            f"CUDA_VISIBLE_DEVICES={cuda_id} python src/main_llm_final.py "
            f"--config='{CONFIG}' --env-config='{ENV_CONFIG}' "
            f"with t_max='{T_MAX}' "
            f"epsilon_anneal_time='{EPSILON_ANNEAL_TIME}' "
            f"name='{name_idxed}' save_dir='{result_dir}' "
            f"\"comm_code_paths={com_code_paths}\" "
            f"mse_thres='{MSE_THRES}' "
            f"meta_lambda='{META_LAMBDA}' "
            f"consistency_lambda='{CONSISTENCY_LAMBDA}' "
            f"wandb_project='{WANDB_PROJECT}' "
            f"wandb_team='{WANDB_TEAM}' "
            f"wandb_mode='{WANDB_MODE}' "
            f"wandb_save_model={WANDB_SAVE_MODEL} "
        )
        if WANDB_KEY:
            base_cmd += f"wandb_key='{WANDB_KEY}' "
            
        if ENV_CONFIG in ['sc2', 'grf']:
            base_cmd += f"env_args.map_name='{MAP_NAME}' "
        else:
            base_cmd += f"env_args.key='{ENV_KEY}' "
        base_cmd += (
            f"important_state='{important_state}' "
            f"running_algorithm_name='{run_name}' "
            f"{' '.join(EXTRA_ARGS)} "


            f"timestep_wise='{TIMESTEP_WISE}' "
        )
        log_path = os.path.join(result_dir, "train.log")
        full_cmd = f"{base_cmd} > '{log_path}' 2>&1"
        proc = subprocess.Popen(full_cmd, shell=True, executable="/bin/bash")
        procs.append(proc)
        print(f"[FINAL] RL experiment {idx+1}/{NUM_SEEDS} started on GPU {cuda_id} (log: {log_path})")

    print(f"[FINAL] Waiting for all {NUM_SEEDS} optimized RL experiments to finish ...")
    
    start_time = time.time()
    while not all(os.path.exists(flag) for flag in done_flag_paths):
        elapsed = time.time() - start_time
        completed = sum(1 for flag in done_flag_paths if os.path.exists(flag))
        print(f"[FINAL] Progress: {completed}/{NUM_SEEDS} completed ({elapsed/60:.1f} min elapsed)")
        time.sleep(3600)
        
    for proc in procs:
        proc.terminate()
        proc.wait()
    
    cleanup_flags(done_flag_paths)
    total_time = time.time() - start_time
    print(f"[FINAL] Multi-seed RL training with optimized communication complete. Total time: {total_time/60:.1f} min\n")

# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Multi-Phase RL Training with LLM Communication")

    parser.add_argument("--config", type=str, required=True, help="Algorithm config name. e.g., qmix_llm")
    parser.add_argument("--env", type=str, required=True, help="Environment name. e.g., sc2, grf, lbf,")
    parser.add_argument("--map", type=str, help="Map name for SC2/GRF environments. e.g., 5z_vs_1ul")
    parser.add_argument("--key", type=str, help="Environment key for lbf/mpe/overcooked/hallway_group")
    parser.add_argument("--t_max", type=int, default=5050000, help="Maximum number of training timesteps")
    parser.add_argument("--max_disc_steps", type=int, default=200, help="Maximum discriminator training steps")
    parser.add_argument("--model", type=str, default="gpt-4.1-2025-04-14", help="LLM Model (gpt-4.1-mini-2025-04-14, gemini-1.5-flash, claude-3-opus-20240229, gpt-4.1-mini)")

    parser.add_argument("--mse_thres", type=float, default=0.05, help="Discriminator MSE threshold")
    parser.add_argument("--meta_lambda", type=float, default=0.1, help="Weight for the meta loss")
    parser.add_argument("--recon_lambda", type=float, default=1.0, help="Weight for the reconstruction loss")
    parser.add_argument("--consistency_lambda", type=float, default=1.0, help="Weight for the consistency loss")
    
    parser.add_argument("--skip_llm", type=bool, default=False, help="Skip LLM steps and run RL only")

    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds for multi-seed RL training")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="LMAC", help="Wandb project name")
    parser.add_argument("--wandb_team", type=str, default="LMAC", help="Wandb team name")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Wandb mode (online/offline)")
    parser.add_argument("--wandb_key", type=str, default="", help="Wandb API key")
    parser.add_argument("--wandb_save_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save models to WandB")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[1,2,3,4,5], help="List of GPU IDs to use for multi-seed training")
    
    
    parser.add_argument("--players", type=int, default=6, help="(LBF only) Number of players")
    parser.add_argument("--sight", type=int, default=1, help="(LBF only) Sight range")
    parser.add_argument("--field_size", type=int, default=11, help="(LBF only) Field size")
    parser.add_argument("--max_food", type=int, default=4, help="(LBF only) Maximum number of food items")
    parser.add_argument("--force_coop", type=lambda x: (str(x).lower() == 'true'), default=False, help="(LBF only) Force cooperation (True/False)")
    

    parser.add_argument("--msg_dim_limit", type=bool, default=False, help="Limit message dimension")
    parser.add_argument("--message_limit_dimension", type=int, default=10, help="Maximum message dimension")
    parser.add_argument("--timestep_wise_phases", type=int, default=1, help="Number of timestep-wise phases")

    cmd_args = parser.parse_args()

    CONFIG = cmd_args.config
    ENV_CONFIG = cmd_args.env
    T_MAX = cmd_args.t_max
    MAX_DISC_STEPS = cmd_args.max_disc_steps
    MSE_THRES = cmd_args.mse_thres
    NUM_SEEDS = cmd_args.num_seeds
    GPU_IDS = cmd_args.gpu_ids
    META_LAMBDA = cmd_args.meta_lambda
    RECON_LAMBDA = cmd_args.recon_lambda
    CONSISTENCY_LAMBDA = cmd_args.consistency_lambda
    WANDB_PROJECT = cmd_args.wandb_project
    WANDB_TEAM = cmd_args.wandb_team
    WANDB_MODE = cmd_args.wandb_mode
    WANDB_KEY = cmd_args.wandb_key
    WANDB_SAVE_MODEL = cmd_args.wandb_save_model
    MSG_DIM_LIMIT = cmd_args.msg_dim_limit
    MESSAGE_LIMIT_DIMENSION = cmd_args.message_limit_dimension  
    MODEL = cmd_args.model
    TIMESTEP_WISE_PHASES = cmd_args.timestep_wise_phases


    print(f"Configuration: {CONFIG} | Environment: {ENV_CONFIG} | Seeds: {NUM_SEEDS} | GPUs: {GPU_IDS}")
    print(f"Data Collection: DISABLED (Using existing data)")

    if ENV_CONFIG in ['sc2', 'sc2v2', 'grf']:
        if not cmd_args.map:
            raise ValueError("For SC2/GRF, --map argument is required.")
        MAP_NAME = cmd_args.map
        ENV_KEY = None
        EXTRA_ARGS = [
            "env_args.sight_range=2",
            "env_args.shoot_range=2",
            "env_args.obs_all_health=False",
            "env_args.obs_enemy_health=False",
        ] if MAP_NAME and "bane_vs_hm" in MAP_NAME.lower() else []
        NAME = f"LMAC_{MODEL}_MSE_{MSE_THRES}"
        RUNNING_ALGORITHM_NAME = f"({MAP_NAME}){NAME}"
        
        if MSG_DIM_LIMIT:
            NAME = f"LMAC_{MODEL}_msg_dim_limit_{MESSAGE_LIMIT_DIMENSION}"
            RUNNING_ALGORITHM_NAME = f"({MAP_NAME}){NAME}"
            
        DATA_ROOT = os.path.join(PROJECT_ROOT, "data", NAME, MAP_NAME)
        print(f"SC2/GRF Environment: {MAP_NAME}")
        
    elif ENV_CONFIG == "lbf":
        players = cmd_args.players
        field_size = cmd_args.field_size
        max_food = cmd_args.max_food
        sight = cmd_args.sight
        force_coop = cmd_args.force_coop
        ENV_KEY = f"Foraging-{field_size}x{field_size}-{players}p-{max_food}f-s{sight}{force_coop}"
        EXTRA_ARGS = [
            f"env_args.sight={sight}",
            f"env_args.players={players}",
            f"env_args.field_size={field_size}",
            f"env_args.max_food={max_food}",
            f"env_args.force_coop={force_coop}"
        ]
        NAME = f"Final_{MODEL}MSE_{MSE_THRES}"
        RUNNING_ALGORITHM_NAME = f"({ENV_KEY}){NAME}"
        
        if MSG_DIM_LIMIT:
            NAME = f"Final_{MODEL}_msg_dim_limit_{MESSAGE_LIMIT_DIMENSION}"
            RUNNING_ALGORITHM_NAME = f"({MAP_NAME}){NAME}"
        
        DATA_ROOT = os.path.join(PROJECT_ROOT, "data", NAME, ENV_KEY)
        print(f"LBF Environment: {ENV_KEY}")

    if MAP_NAME:
        BUFFER_DIR = os.path.join(PROJECT_ROOT, "data", MAP_NAME)
    elif ENV_KEY:
        BUFFER_DIR = os.path.join(PROJECT_ROOT, "data", ENV_KEY)
    else:
        raise ValueError("MAP_NAME or ENV_KEY must be specified for buffer path")
    TEST_BUFFER_DIR = os.path.join(BUFFER_DIR, "test")
    print(f"[SETUP] Using existing buffer directory: {BUFFER_DIR}")

    print(f"Data root: {DATA_ROOT}")
    print(f"Buffer directories: {BUFFER_DIR}, {TEST_BUFFER_DIR}")

    all_flag_files = []
    for i in range(NUM_SEEDS):
        all_flag_files.append(os.path.join(DATA_ROOT, f"LMAC_init_seed{i}", "done.flag"))
        all_flag_files.append(os.path.join(DATA_ROOT, f"LMAC_Final_seed{i}", "done.flag"))
    for fpath in [
        os.path.join(DATA_ROOT, "phase2_done.flag"),
        os.path.join(DATA_ROOT, "phase3_done.flag"),
        os.path.join(DATA_ROOT, "phase4_done.flag"),
    ]:
        all_flag_files.append(fpath)
    cleanup_flags(all_flag_files)

    if cmd_args.skip_llm:
        print("[SKIP_LLM] Running RL-only mode...")
        llm_result = os.path.join(PROJECT_ROOT, "src", "llm_source_archive", MAP_NAME if MAP_NAME else ENV_KEY)
        com_code_paths = [
            os.path.join(llm_result, "comm_init.py"),
            os.path.join(llm_result, "comm_update.py"),
            os.path.join(llm_result, "comm_update_timestep_wise2.py"),
        ]
        imp_state_select = os.path.join(llm_result, "imp_state_select.py")
        for path in [*com_code_paths, imp_state_select]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path does not exist: {path}")
        important_state = load_imp_state(imp_state_select)
        final_parallel_train(com_code_paths, important_state)

    else:
        print("[FULL_PIPELINE] Running complete 3-Phase LLM + RL pipeline...")
        start_time = time.time()
        com_code_paths, important_state, feedback_data_path = main()
        final_parallel_train(com_code_paths, important_state)
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"[COMPLETE] Full pipeline completed in {total_time/60:.1f} minutes")
        print("=" * 80)
