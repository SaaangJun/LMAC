import importlib
import os
import sys
import torch as th


class MsgConverter:

    def __init__(self, args, comm_code_paths, phase_info=None):

        if isinstance(comm_code_paths, str):
            comm_code_paths = [comm_code_paths]
        
        self.comm_code_paths = comm_code_paths
        self.phase_info = phase_info or {}
        self.converters = []
        self.converter_names = []
        self.converter_types = []  
        print(f"[MsgConverter] Loading {len(comm_code_paths)} communication module(s)...")
        
        for i, comm_code_path in enumerate(comm_code_paths):
            module_name = os.path.splitext(os.path.basename(comm_code_path))[0] 
            unique_module_name = f"{module_name}_{i}" if i > 0 else module_name
            spec = importlib.util.spec_from_file_location(unique_module_name, comm_code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module
            spec.loader.exec_module(module)
            self.converters.append(module)
            self.converter_names.append(module_name)
            converter_type = "timestep" if 'timestep' in comm_code_path.lower() else "normal"
            self.converter_types.append(converter_type)
            print(f"[MsgConverter] Loaded module {i+1}/{len(comm_code_paths)}: {module_name} ({converter_type}) from {comm_code_path}")
        
        self.individual_dims = []
        self.obs_dim = None
        self._calculate_dimensions(args)
        self.obsmsg_dim = sum(self.individual_dims)
        print(f"[MsgConverter] Total combined message dimension: {self.obsmsg_dim}")

    def _calculate_dimensions(self, args):
        for i, converter in enumerate(self.converters):
            converter_type = self.converter_types[i]
            if converter_type == "timestep":
                time_seq = 10
                test_input = th.zeros(1, time_seq, args.n_agents, args.obs_ext_dim)
            else:
                test_input = th.zeros(1, args.n_agents, args.obs_ext_dim)
            full_output = converter.communication(test_input)
            if i == 0:
                self.obs_dim = args.obs_ext_dim
                if converter_type == "timestep":
                    full_output_reshaped = full_output.reshape(1, args.n_agents, -1)
                    msg_dim = full_output_reshaped.shape[-1] - self.obs_dim
                    total_dim = full_output_reshaped.shape[-1]
                else:
                    msg_dim = full_output.shape[-1] - self.obs_dim  
                    total_dim = full_output.shape[-1]  
                self.individual_dims.append(total_dim)
                print(f"[MsgConverter] Module {self.converter_names[i]} (first-{converter_type}): obs({self.obs_dim}) + msg({msg_dim}) = {total_dim}")
            else:
                if converter_type == "timestep":
                    full_output_reshaped = full_output.reshape(1, args.n_agents, -1)
                    msg_dim = full_output_reshaped.shape[-1] - self.obs_dim
                else:
                    msg_dim = full_output.shape[-1] - self.obs_dim
                self.individual_dims.append(msg_dim)
                print(f"[MsgConverter] Module {self.converter_names[i]} (additional-{converter_type}): msg_only({msg_dim})")

    def _build_time_inputs(self, obs, t):
        B, T, N, obs_ext_dim = obs.shape
        time_seq = 10  
        input_seq = []   
        for i in range(t - time_seq + 1, t + 1):
            if i < 0:
                obs_i = th.full_like(obs[:, 0], fill_value=-1.0) 
            else:
                obs_i = obs[:, i] 
            input_seq.append(obs_i)
        time_inputs = th.stack(input_seq, dim=1) 
        return time_inputs

    def __call__(self, obs):
        B, T, N, _ = obs.shape
        messages = []
        for t in range(T):
            timestep_messages = []
            for i, converter in enumerate(self.converters):
                if 'timestep' in self.comm_code_paths[i].lower():
                    time_input_4d = self._build_time_inputs(obs, t)
                    full_output = converter.communication(time_input_4d)
                else:
                    full_output = converter.communication(obs[:, t]) 
                
                if i == 0:
                    timestep_messages.append(full_output)
                else:
                    msg_only = full_output[:, :, self.obs_dim:]  
                    timestep_messages.append(msg_only)
            combined_msg = th.cat(timestep_messages, dim=-1) 
            messages.append(combined_msg)
        result = th.stack(messages, dim=1) 
        return result
    
    def get_converter_info(self):
        return {
            'num_converters': len(self.converters),
            'converter_names': self.converter_names,
            'individual_dims': self.individual_dims,
            'total_dim': self.obsmsg_dim,
            'paths': self.comm_code_paths
        }
    
    def get_individual_messages(self, obs):
        B, T, N, _ = obs.shape
        individual_results = []
        
        for converter_idx, converter in enumerate(self.converters):
            messages = []
            for t in range(T):
                if 'timestep' in self.comm_code_paths[converter_idx].lower():
                    time_input_4d = self._build_time_inputs(obs, t) 
                    full_output = converter.communication(time_input_4d)  
                else:
                    full_output = converter.communication(obs[:, t]) 
                if converter_idx == 0:
                    messages.append(full_output)
                else:
                    msg_only = full_output[:, :, self.obs_dim:]
                    messages.append(msg_only)
            result = th.stack(messages, dim=1) 
            individual_results.append(result)
        
        return individual_results
