"""
Code generation and validation utilities for LLM communication
"""
import os
import re
import sys
import importlib.util
import numpy as np


class CodeUtils:
    def __init__(self, args):
        self.args = args

        if args.env == "grf":
            short_name = {
                "academy_run_pass_and_shoot_with_keeper": "2_vs_2",
                "academy_3_vs_1_with_keeper": "3_vs_2",
            }
            map_name = short_name[args.env_args['map_name']]
            self.code_dir = os.path.join(os.getcwd(), 'src/llm_source', f'{args.name}', map_name)
        elif args.env in ["sc2", "sc2v2"]:
            map_name = args.env_args['map_name']
            self.code_dir = os.path.join(os.getcwd(), 'src/llm_source', f'{args.name}', map_name)
        else:
            key = args.env_args['key']
            self.code_dir = os.path.join(os.getcwd(), 'src/llm_source', f'{args.name}', key)

    def extract_code_block(self, llm_text):
        matches = re.findall(r"```(?:python)?\n([\s\S]*?)```", llm_text)
        if matches:
            return matches[0]

        lines = llm_text.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("def "):
                in_code = True
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines)

    def import_and_reload_module(self, module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  
        spec.loader.exec_module(module)
        return module

    def validate_important_state_function(self, module):
        if not hasattr(module, 'select_important_state'):
            raise AttributeError("No select_important_state() in LLM code")
        
        important_dims = module.select_important_state()
        important_dims = [i for x in important_dims for i in (x if isinstance(x, list) else [x])]
        
        if not isinstance(important_dims, (list, np.ndarray)):
            raise TypeError("select_important_state() did not return a list or np.ndarray")
        
        important_dims = list(map(int, important_dims))
        return important_dims

    def validate_communication_function(self, module, test_obs, timestep_wise=False):
        if not hasattr(module, 'communication'):
            raise AttributeError("No communication() in LLM code")
        
        if timestep_wise:
            test_obs_tw = test_obs.unsqueeze(1).repeat(1, 10, 1, 1)
            cur_com = module.communication(test_obs_tw)
            message_dim = cur_com.shape[-1] - test_obs_tw.shape[-1]
            B,T,N,D = test_obs_tw.shape
            
            if len(test_obs.shape) != len(cur_com.shape):
                raise ValueError(
                    f"Shape mismatch: Communication protocol's output must be ({B}, {N}, {D} + message_dim)., "
                    f"but got {cur_com.shape}. Note: the temporal dimension should be compressed to generate the message at the current timestep."
                    f"Input shape : ({B,T,N,D}), Output shape: ({B}, {N}, {D} + message_dim)"
                )
                
            if self.args.message_dim_limit and message_dim > self.args.message_limit_dimension * self.args.n_agents:
                raise ValueError(f"Message dimension {message_dim} exceeds the limit {self.args.message_limit_dimension}")


        else:
            cur_com = module.communication(test_obs)
            message_dim = cur_com.shape[-1] - test_obs.shape[-1]
    
            if cur_com.shape[:2] != test_obs.shape[:2]:
                raise ValueError("Shape mismatch between output and input")
            
            if message_dim < 0:
                raise ValueError("Output dimension must be >= input dimension")
            
            if self.args.message_dim_limit and message_dim > self.args.message_limit_dimension * self.args.n_agents:
                raise ValueError(f"Message dimension {message_dim} exceeds the limit {self.args.message_limit_dimension}")

        
        return message_dim
