"""
Environment-specific utilities for LLM communication
"""
import os
import re
import pandas as pd


class EnvUtils:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.n_agents = args.n_agents
        env_excel_name = "sc2.xlsx" if args.env in ["sc2", "sc2v2"] else f"{args.env}.xlsx"
        self.excel_path = os.path.join(args.dir, env_excel_name)

        if args.env == "grf":
            short_name = {
                "academy_run_pass_and_shoot_with_keeper": "2_vs_2",
                "academy_3_vs_1_with_keeper": "3_vs_2",
            }
            self.sheet_name = short_name[args.env_args['map_name']]
        elif args.env in ["sc2", "sc2v2"]:
            self.sheet_name = args.env_args['map_name']
        else:
            self.sheet_name = None

    def get_detail_content_and_task_desc(self, timewise=False):
        env = self.args.env

        if env in ["sc2", 'sc2v2', "grf"]:
            obs_file = pd.read_excel(self.excel_path, header=None, sheet_name=self.sheet_name)
            content = list(obs_file.iloc[:, 1])
            unit = list(obs_file.iloc[:, -2])
            detail_content = ''
            for ii in range(len(content)):
                if timewise:
                    prefix = f'o[batch][t][agent_id][{ii}]'
                else:
                    prefix = f'o[batch][agent_id][{ii}]'
                if 'unknown' in unit[ii].lower():
                    detail_content += f'- `{prefix}`: {content[ii]} — (unknown) This value is unobservable due to environment constraints.\n'
                else:
                    detail_content += f'- `{prefix}`: {content[ii]} — ({unit[ii]}) type value.\n'
            detail_content += 'Agents or enemy that are observable have valid values, while agents or enemy that are unobservable have all values set to zero.'
            task_description = obs_file.iloc[0, -1]
            total_dim = len(content)
            return detail_content, task_description, total_dim
        else:
            df, task_desc, _ = self.build_env_obs_df_and_desc(env)
            detail_content = ""
            for idx, row in df.iterrows():
                if timewise:
                    prefix = f'o[batch][t][agent_id][{row["Index"]}]'
                else:
                    prefix = f'o[batch][agent_id][{row["Index"]}]'
                detail_content += f"- {prefix}: {row['Feature Name']} — ({row['Type']})\n"
            total_dim = len(df)
            return detail_content, task_desc, total_dim

    def get_state_detail_content_and_task_desc(self):
        env = self.args.env
        
        if env in ["sc2", 'sc2v2', "grf"]:
            obs_file_state = pd.read_excel(self.excel_path, header=None, sheet_name=self.sheet_name + '(state)')
            content_state = list(obs_file_state.iloc[:, 1])
            unit_state = list(obs_file_state.iloc[:, -2])
            detail_content_state = ''
            for ii in range(len(obs_file_state)):
                detail_content_state += f'- `s[{ii}]`: {content_state[ii]}, {unit_state[ii]} value.\n'
            task_description = obs_file_state.iloc[0, -1]
            total_dim = len(obs_file_state)
            return detail_content_state, task_description, total_dim
        else:
            df_state, task_desc = self.build_env_state_df_and_desc(env)
            detail_content_state = ""
            for idx, row in df_state.iterrows():
                detail_content_state += f"- s[{row['Index']}]: {row['Feature Name']}, {row['Type']} value.\n"
            total_dim = len(df_state)
            return detail_content_state, task_desc, total_dim

    def build_env_obs_df_and_desc(self, env):
        if env in ['sc2', 'sc2v2', 'grf']:
            raise RuntimeError("Not use build_env_obs_df_and_desc for smac or grf")

        def add_last_action_and_agent_id(df, env_type, n_agents):
            rows = []
            idx_offset = len(df)

            if env_type == 'lbf':
                action_names = ["No-op", "Move north", "Move south", "Move west", "Move east", "Pick-up"]
            elif env_type == 'hallway':
                action_names = ["No-op", "Decrement state", "Increment state"]
            else:
                action_names = []

            for i, name in enumerate(action_names):
                rows.append([
                    str(idx_offset + i),
                    f"last_action_{i}: {name}",
                    "binary [0 or 1]"
                ])
            idx_offset += len(action_names)

            for i in range(n_agents):
                rows.append([
                    str(idx_offset + i),
                    f"agent_id_onehot_{i}: 1 if agent index == {i}, else 0",
                    "binary [0 or 1]"
                ])
            idx_offset += n_agents
            df2 = pd.DataFrame(rows, columns=["Index", "Feature Name", "Type"])
            df = pd.concat([df, df2], ignore_index=True)
            return df
        
        if env == 'lbf':
            N_AGENTS = int(self.args.env_args.get('players', 6))
            sight = int(self.args.env_args.get('sight', 1))
            grid_h = grid_w = 2 * sight + 1
            rows = []
            idx = 0

            for y in range(grid_h):
                for x in range(grid_w):
                    if y == sight and x == sight:
                        fname = f"agent_layer[{y},{x}]: self (always 1)"
                    else:
                        fname = f"agent_layer[{y},{x}]: 1 if other agent present, else 0"
                    rows.append([str(idx), fname, "binary [0 or 1]"])
                    idx += 1

            for y in range(grid_h):
                for x in range(grid_w):
                    fname = f"food_layer[{y},{x}]: 1 if food present, else 0"
                    rows.append([str(idx), fname, "binary [0 or 1]"])
                    idx += 1

            for y in range(grid_h):
                for x in range(grid_w):
                    fname = f"access_layer[{y},{x}]: 1 if accessible, else 0"
                    rows.append([str(idx), fname, "binary [0 or 1]"])
                    idx += 1

            task_desc = (
                f"LBF (grid obs): Each agent observes a local {grid_h}×{grid_w} grid centered on itself. "
                f"Observation is a 3-layer grid (agent/food/access), flatten order: agent layer → food layer → access layer (row-major). "
                f"agent_layer[{sight},{sight}] is always self (1), others are for nearby agents."
                f"The key is tracking food and agent positions in the actual grid, so coordination relies on where things truly are—not the padded edges."
                f"Since agents can only observe within their limited sight range, simply sharing local observations is insufficient to recover the full global state. Messages therefore need to be structured to go beyond raw local views by incorporating additional cues—such as positional context—so that other agents can complement their partial observations and infer a more coherent representation of the global situation." )

            df_base = pd.DataFrame(rows, columns=["Index", "Feature Name", "Type"])
            df = add_last_action_and_agent_id(df_base, 'lbf', N_AGENTS)
            return df, task_desc, df_base
        else:
            pass

        raise NotImplementedError(f"Observation table auto-generation for env='{env}' with env_args={self.args.env_args} is not implemented.")
    
    def get_imp_state_names_and_units(self, imp_state):
        env = self.args.env

        if env in ["sc2", 'sc2v2', "grf"]:
            state_file = pd.read_excel(self.excel_path, header=None, sheet_name=self.sheet_name + '(state)')
            all_state_names = list(state_file.iloc[:, 1])
            all_units = list(state_file.iloc[:, -2])
            task_desc = state_file.iloc[0, -1]
        elif env == 'lbf':
            df_obs, task_desc = self.build_env_state_df_and_desc_expanded(env)
            all_state_names = list(df_obs["Feature Name"])
            all_units = list(df_obs["Type"])
            
        else:
            df_obs, task_desc = self.build_env_state_df_and_desc(env)
            all_state_names = list(df_obs["Feature Name"])
            all_units = list(df_obs["Type"])
        filtered_names = [all_state_names[i] for i in imp_state]
        filtered_units = [all_units[i] for i in imp_state]
        return filtered_names, filtered_units, task_desc

    def build_env_state_df_and_desc(self, env):
        if env in ['grf', 'sc2', 'sc2v2']:
            raise RuntimeError("Not use build_env_state_df_and_desc for sc2/grf/sc2v2 ")

        if env == 'lbf':
            field_rows = int(self.args.env_args['field_size'])
            field_cols = int(self.args.env_args['field_size'])
            sight = int(self.args.env_args['sight'])
            
            pad_rows = field_rows + 2 * sight
            pad_cols = field_cols + 2 * sight
            P = pad_rows * pad_cols

            rows = []
            for l, layer in enumerate(['Agent layer', 'Food layer', 'Access layer']):
                base = l * P
                for y in range(pad_rows):
                    group_start = None
                    group_type = None
                    for x in range(pad_cols):
                        idx = base + y * pad_cols + x
                        if sight <= x < sight + field_cols and sight <= y < sight + field_rows:
                            cur_type = "field"
                        else:
                            cur_type = "padding"
                        if group_type is None:
                            group_start = idx
                            group_type = cur_type
                        last_cell = (x == pad_cols - 1)
                        if cur_type != group_type or last_cell:
                            end_idx = idx if last_cell and cur_type == group_type else idx - 1
                            if group_start == end_idx:
                                idx_range = f"{group_start}"
                            else:
                                idx_range = f"{group_start}-{end_idx}"
                            rows.append([
                                idx_range,
                                f"{layer} ({group_type})",
                                group_type
                            ])
                            group_start = idx
                            group_type = cur_type

            df_state = pd.DataFrame(rows, columns=["Index", "Feature Name", "Type"])
            
            task_desc = (
                f"LBF (state): Global 3-layer grid of shape ({pad_rows},{pad_cols}) "
                f"with sight={sight} padding. "
                f"Positions agent info layer[{sight}:{sight+field_rows-1}, {sight}:{sight+field_cols-1}] "
                f"are the actual game field; other positions are zero-padded boundary (padding). "
                f"Flattened order: agent layer → food layer → access layer (row-major). "
                f"Agents must collect food to gain rewards. "
                f"Episode ends when all food is collected or after a fixed time. "
                f"Each layer: agent layer = agent presence, food layer = food presence (reward source), access layer = accessible positions."
                )
            return df_state, task_desc
        else:
            df_obs, task_desc, df_base = self.build_env_obs_df_and_desc(env)
            N = int(getattr(self.args, 'n_agents', self.args.env_args.get('n_agents', 2)))
            state_rows = []
            for agent_id in range(N):
                for idx, row in df_base.iterrows():
                    state_idx = agent_id * len(df_base) + int(row["Index"])
                    feature_desc = f"agent{agent_id}_{row['Feature Name']}"
                    state_rows.append([
                        str(state_idx),
                        feature_desc,
                        row["Type"]
                    ])
            state_df = pd.DataFrame(
                state_rows,
                columns=["Index", "Feature Name", "Type"]
            )

            return state_df, task_desc

    def build_env_state_df_and_desc_expanded(self, env):
        if env == 'lbf':
            field_rows = int(self.args.env_args['field_size'])
            field_cols = int(self.args.env_args['field_size'])
            sight = int(self.args.env_args['sight'])
            pad_rows = field_rows + 2 * sight
            pad_cols = field_cols + 2 * sight
            P = pad_rows * pad_cols

            rows = []
            for l, layer in enumerate(['Agent layer', 'Food layer', 'Access layer']):
                base = l * P
                for y in range(pad_rows):
                    for x in range(pad_cols):
                        idx = base + y * pad_cols + x
                        if sight <= x < sight + field_cols and sight <= y < sight + field_rows:
                            cur_type = "field"
                        else:
                            cur_type = "padding"
                        rows.append([
                            idx,
                            f"{layer} ({cur_type})",
                            cur_type
                        ])

            df_state = pd.DataFrame(rows, columns=["Index", "Feature Name", "Type"])

            task_desc = (
                f"LBF (state): Global 3-layer grid of shape ({pad_rows},{pad_cols}) "
                f"with sight={sight} padding. "
                f"Positions agent info layer[{sight}:{sight+field_rows-1}, {sight}:{sight+field_cols-1}] "
                f"are the actual game field; other positions are zero-padded boundary (padding). "
                f"Flattened order: agent layer → food layer → access layer (row-major). "
                f"Agents must collect food to gain rewards. "
                f"Episode ends when all food is collected or after a fixed time. "
                f"Each layer: agent layer = agent presence, food layer = food presence (reward source), access layer = accessible positions."
            )
            return df_state, task_desc
