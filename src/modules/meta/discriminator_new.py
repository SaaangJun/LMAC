import os
import pickle
import time

import wandb
import torch as th
import torch.nn.functional as F
import numpy as np

from modules.meta import REGISTRY as meta_REGISTRY
from LLM.llm_core import Communication

class Discriminator:
    def __init__(self, args, msg_converter, discriminator_name="original"):
        self.msg_converter = msg_converter
        self.args = args
        self.discriminator_name = discriminator_name
        self.wandb_run = None
        self.args.obsmsg_dim = args.obsmsg_dim  
        wandb_project = 'MARL_Discriminator'
        if getattr(args, "use_wandb", False):
            map_name = args.env_args.get('map_name', args.env_args.get('key', 'unknown'))
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=f"{map_name}_{discriminator_name}_{self.args.mse_thres}_{args.model}",
                config=args.__dict__,
                reinit=True 
            )
            print(f"[WANDB] Initialized separate run for {discriminator_name}")
    
        self.model = meta_REGISTRY[args.discriminator](args.obsmsg_dim, args.imp_state_dim, args=args).to(args.device)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=args.lr)
        self.llm = Communication(args, None)
    
    def finish_wandb(self):
        if self.wandb_run is not None:
            wandb.finish()
            print(f"[WANDB] Finished run for {self.discriminator_name}")
            self.wandb_run = None

    def _has_timestep_module(self):
        if hasattr(self.msg_converter, 'comm_code_paths'):
            return any('timestep' in path.lower() for path in self.msg_converter.comm_code_paths)
        return False

    def preprocess(self, samples):
        obs = th.stack(samples['obs']).squeeze(1).to(self.args.device)
        state = th.stack(samples['state']).squeeze(1).to(self.args.device)
        mask = th.stack(samples['mask']).squeeze(1).to(self.args.device)
        actions_onehot = th.stack(samples['actions_onehot']).squeeze().to(self.args.device)
        
        B, T, N = obs.shape[:3]
        
        if self._has_timestep_module():
            obs_ext = th.zeros((obs.shape[0], obs.shape[1], obs.shape[2], self.args.obs_ext_dim), device=self.args.device)
            obs_ext[..., :self.args.obs_shape] = obs
            obs_ext[:, 1:, :, self.args.obs_shape:self.args.obs_shape + self.args.n_actions] = actions_onehot[:, :-1]
            obs_ext[..., self.args.obs_shape + self.args.n_actions:] = th.eye(self.args.n_agents, device=obs_ext.device).reshape(1, 1, self.args.n_agents, self.args.n_agents)
            messages = self.msg_converter(obs_ext)
            
        else:
            obs_ext = th.zeros((obs.shape[0], obs.shape[1], obs.shape[2], self.args.obs_ext_dim), device=self.args.device)
            obs_ext[..., :self.args.obs_shape] = obs
            obs_ext[:, 1:, :, self.args.obs_shape:self.args.obs_shape + self.args.n_actions] = actions_onehot[:, :-1]
            obs_ext[..., self.args.obs_shape + self.args.n_actions:] = th.eye(self.args.n_agents, device=obs_ext.device).reshape(1, 1, self.args.n_agents, self.args.n_agents)
            messages = self.msg_converter(obs_ext)
        
        return messages, state[...,self.args.imp_state], mask
    
    def preprocess_baseline(self, samples):

        obs = th.stack(samples['obs']).squeeze(1).to(self.args.device)
        state = th.stack(samples['state']).squeeze(1).to(self.args.device)
        mask = th.stack(samples['mask']).squeeze(1).to(self.args.device)
        actions_onehot = th.stack(samples['actions_onehot']).squeeze().to(self.args.device)
        obs_ext = th.zeros((obs.shape[0], obs.shape[1], obs.shape[2], self.args.obs_ext_dim), device=self.args.device)
        obs_ext[..., :self.args.obs_shape] = obs
        obs_ext[:, 1:, :, self.args.obs_shape:self.args.obs_shape + self.args.n_actions] = actions_onehot[:, :-1]
        obs_ext[..., self.args.obs_shape + self.args.n_actions:] = th.eye(self.args.n_agents, device=obs_ext.device).reshape(1, 1, self.args.n_agents, self.args.n_agents)

        return obs_ext, state[...,self.args.imp_state], mask

    def train(self, samples):
        self.model.train()
        inp, target, mask = self.preprocess(samples)
        B, T, N, _ = inp.shape
        mask = mask.unsqueeze(2)
        target = target.unsqueeze(2)
        losses = []
        mse_losses = []
        mse_per_agent_tot = []
        for epoch in range(self.args.n_epochs):
            hidden = self.model.init_hidden().unsqueeze(0).expand(B, self.args.n_agents, -1)
            preds = []
            for t in range(T):
                input_t = inp[:, t].to(self.args.device).reshape(-1, self.args.obsmsg_dim)
                out_t, _, hidden = self.model(input_t, hidden)
                preds.append(out_t.reshape(B, self.args.n_agents, -1))
            pred = th.stack(preds, dim=1)  

            mask_exp = mask.expand_as(pred)
            target_exp = target.expand_as(pred)
            state_pred_loss = (pred - target_exp) * mask_exp
            state_pred_loss = state_pred_loss ** 2

            mse_per_agent = (state_pred_loss.sum(dim=(0, 1, 3)) / mask_exp.sum(dim=(0, 1, 3)))
            mse_per_agent_tot.append(mse_per_agent.detach().cpu().numpy())

            mse_loss = state_pred_loss.sum() / mask_exp.sum()
            total_loss = self.args.recon_lambda * mse_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
            mse_losses.append(mse_loss.item())

        loss = sum(losses) / len(losses)
        mse_loss = sum(mse_losses) / len(mse_losses) if mse_losses else 0.0

        mse_per_agent_avg = np.stack(mse_per_agent_tot).mean(axis=0)
        mse_per_agent_dict = {f"agent_{i}_mse": float(val) for i, val in enumerate(mse_per_agent_avg)}

        if getattr(self.args, "use_wandb", False):
            wandb.log({
                "train_loss": loss,
                "mse_loss": mse_loss,
                "total_epochs": self.args.n_epochs,
                **{f"agent_{i}_mse": float(val) for i, val in enumerate(mse_per_agent_avg)},
            })
        return mse_loss

    def evaluate(self, samples, save_path):
        self.model.eval()
        os.makedirs(save_path, exist_ok=True)
        with th.no_grad():
            inp, target, mask = self.preprocess(samples)
            B, T, N, _ = inp.shape

            hidden = self.model.init_hidden().unsqueeze(0).expand(B, N, -1)
            preds = []
            for t in range(T):
                input_t = inp[:, t].to(self.args.device).reshape(-1, self.args.obsmsg_dim)
                out_t, _, hidden = self.model(input_t, hidden)
                preds.append(out_t.reshape(B, N, -1))
            pred = th.stack(preds, dim=1) 
            target_exp = target.unsqueeze(2).expand_as(pred)
            mask_exp = mask.unsqueeze(2).to(pred.device).expand_as(pred)
            squared_error = ((pred - target_exp) ** 2) * mask_exp

            mse_per_agent = []
            for i in range(N):
                denom = mask_exp[:, :, i, :].sum()
                mse = (squared_error[:, :, i, :].sum() / denom).item() if denom > 0 else 0.0
                mse_per_agent.append(mse)
            total_loss = sum(mse_per_agent) / N if N > 0 else 0.0
            pred_masked = pred.clone()
            squared_error_masked = squared_error.clone()
            target_masked = target_exp.clone()
            pred_masked[mask_exp == 0] = 0.0
            squared_error_masked[mask_exp == 0] = 0.0
            target_masked[mask_exp == 0] = 0.0

            print(f"[Test-{self.discriminator_name}] Final prediction loss: {total_loss:.5f}")
            print(f"[Test-{self.discriminator_name}] Per-agent MSE: {mse_per_agent}")

            if getattr(self.args, "use_wandb", False):
                wandb.log({
                    "test_loss": total_loss,
                    "avg_agent_mse": sum(mse_per_agent) / len(mse_per_agent) if mse_per_agent else 0,
                    **{f"agent_{i}_test_mse": mse for i, mse in enumerate(mse_per_agent)}
                })
                
                imp_state = self.args.imp_state
                names, units, _ = self.llm.get_imp_state_names_and_units(imp_state)
                se_np = squared_error_masked.cpu().numpy()
                mean_se = se_np.mean(axis=(0, 1))  # (N, D)
                N, D = mean_se.shape

                columns = ["state_dim", "feature_name"] + [f"agent_{i}_se" for i in range(N)]
                data = []
                for d in range(D):
                    row = [d, names[d] if d < len(names) else f"state_{d}"]
                    row.extend([float(mean_se[i, d]) for i in range(N)])
                    data.append(row)
                data = [[v if not isinstance(v, np.float32) else round(float(v), 4) for v in row] for row in data]
                table = wandb.Table(columns=columns, data=data)
                wandb.log({"state_prediction_analysis": table})

            if save_path is not None:
                feedback_data = {
                    'test_loss': total_loss,
                    'mse_per_agent': mse_per_agent,
                    'squared_error': squared_error_masked.cpu(),
                    'test_pred_state': pred_masked.cpu(),
                    'test_gt_state': target_masked.cpu(),
                    'mask': mask_exp.cpu(),
                    'discriminator_name': self.discriminator_name,
                    'timestamp': time.time(),
                    'args': {
                        'mse_thres': getattr(self.args, 'mse_thres', 0.002),
                        'obsmsg_dim': getattr(self.args, 'obsmsg_dim', 0),
                        'imp_state_dim': getattr(self.args, 'imp_state_dim', 0),
                        'phase_num': getattr(self.args, 'phase_num', None)
                    }
                }
                
                file_path = os.path.join(save_path, f"feedback_results_{self.discriminator_name}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump(feedback_data, f)
                print(f"[Test-{self.discriminator_name}] Saved feedback data to {file_path}")

        return total_loss, squared_error_masked

    def baseline_train(self, samples):
        """Train baseline model without communication using the same training loop as main model"""
        if not hasattr(self, 'baseline_model'):
            self.baseline_model = meta_REGISTRY[self.args.discriminator](self.args.obs_ext_dim, self.args.imp_state_dim, args=self.args).to(self.args.device)
            self.baseline_optimizer = th.optim.Adam(self.baseline_model.parameters(), lr=self.args.lr)
        
        self.baseline_model.train()
        inp_with_msg, target, mask = self.preprocess_baseline(samples)
        inp_baseline = inp_with_msg 
        
        B, T, N, _ = inp_baseline.shape
        mask = mask.unsqueeze(2)
        target = target.unsqueeze(2)
        losses = []
        mse_losses = []
        mse_per_agent_tot = []

        for epoch in range(self.args.n_epochs):
            hidden = self.baseline_model.init_hidden().unsqueeze(0).expand(B, self.args.n_agents, -1)
            preds = []
            for t in range(T):
                input_t = inp_baseline[:, t].to(self.args.device).reshape(-1, self.args.obs_ext_dim)
                out_t, _, hidden = self.baseline_model(input_t, hidden)
                preds.append(out_t.reshape(B, self.args.n_agents, -1))
            pred = th.stack(preds, dim=1)  
            mask_exp = mask.expand_as(pred)
            target_exp = target.expand_as(pred)
            state_pred_loss = (pred - target_exp) * mask_exp
            state_pred_loss = state_pred_loss ** 2

            mse_per_agent = (state_pred_loss.sum(dim=(0, 1, 3)) / mask_exp.sum(dim=(0, 1, 3)))
            mse_per_agent_tot.append(mse_per_agent.detach().cpu().numpy())

            mse_loss = state_pred_loss.sum() / mask_exp.sum()
            total_loss = self.args.recon_lambda * mse_loss

            self.baseline_optimizer.zero_grad()
            total_loss.backward()
            self.baseline_optimizer.step()
            losses.append(total_loss.item())
            mse_losses.append(mse_loss.item())

        loss = sum(losses) / len(losses)
        mse_loss = sum(mse_losses) / len(mse_losses) if mse_losses else 0.0

        mse_per_agent_avg = np.stack(mse_per_agent_tot).mean(axis=0)
        mse_per_agent_dict = {f"agent_{i}_baseline_mse": float(val) for i, val in enumerate(mse_per_agent_avg)}

        if getattr(self.args, "use_wandb", False):
            wandb.log({
                "baseline_train_loss": loss,
                "baseline_mse_loss": mse_loss,
                "total_epochs": self.args.n_epochs,
                **{f"baseline_agent_{i}_mse": float(val) for i, val in enumerate(mse_per_agent_avg)},
            })
        return mse_loss

    def baseline_evaluate(self, samples, save_path):
        """Evaluate baseline model without communication"""
        if not hasattr(self, 'baseline_model'):
            print(f"[Baseline-{self.discriminator_name}] No baseline model found. Training baseline first...")
            return None, None
        
        self.baseline_model.eval()
        os.makedirs(save_path, exist_ok=True)
        with th.no_grad():
            inp_baseline, target, mask = self.preprocess_baseline(samples)
            B, T, N, _ = inp_baseline.shape

            hidden = self.baseline_model.init_hidden().unsqueeze(0).expand(B, N, -1)
            preds = []
            for t in range(T):
                input_t = inp_baseline[:, t].to(self.args.device).reshape(-1, self.args.obs_ext_dim)
                out_t, _, hidden = self.baseline_model(input_t, hidden)
                preds.append(out_t.reshape(B, N, -1))
            pred = th.stack(preds, dim=1)  
            target_exp = target.unsqueeze(2).expand_as(pred)
            mask_exp = mask.unsqueeze(2).to(pred.device).expand_as(pred)
            squared_error = ((pred - target_exp) ** 2) * mask_exp

            mse_per_agent = []
            for i in range(N):
                denom = mask_exp[:, :, i, :].sum()
                mse = (squared_error[:, :, i, :].sum() / denom).item() if denom > 0 else 0.0
                mse_per_agent.append(mse)
            total_loss = sum(mse_per_agent) / N if N > 0 else 0.0

            pred_masked = pred.clone()
            squared_error_masked = squared_error.clone()
            target_masked = target_exp.clone()
            pred_masked[mask_exp == 0] = 0.0
            squared_error_masked[mask_exp == 0] = 0.0
            target_masked[mask_exp == 0] = 0.0

            print(f"[Test-Baseline-{self.discriminator_name}] Final baseline prediction loss: {total_loss:.5f}")
            print(f"[Test-Baseline-{self.discriminator_name}] Per-agent baseline MSE: {mse_per_agent}")

            if getattr(self.args, "use_wandb", False):
                wandb.log({
                    "baseline_test_loss": total_loss,
                    "baseline_avg_agent_mse": sum(mse_per_agent) / len(mse_per_agent) if mse_per_agent else 0,
                    **{f"baseline_agent_{i}_test_mse": mse for i, mse in enumerate(mse_per_agent)}
                })

                imp_state = self.args.imp_state
                names, units, _ = self.llm.get_imp_state_names_and_units(imp_state)
                se_np = squared_error_masked.cpu().numpy()
                mean_se = se_np.mean(axis=(0, 1))  
                N, D = mean_se.shape

                columns = ["state_dim", "feature_name"] + [f"agent_{i}_se" for i in range(N)]
                data = []
                for d in range(D):
                    row = [d, names[d] if d < len(names) else f"state_{d}"]
                    row.extend([float(mean_se[i, d]) for i in range(N)])
                    data.append(row)
                data = [[v if not isinstance(v, np.float32) else round(float(v), 4) for v in row] for row in data]
                table = wandb.Table(columns=columns, data=data)
                wandb.log({"baseline_state_prediction_analysis": table})

            if save_path is not None:
                file_path = os.path.join(save_path, f"feedback_results_{self.discriminator_name}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump({
                        "test_pred_state": pred_masked.cpu(),
                        "test_gt_state": target_masked.cpu(),
                        "squared_error": squared_error_masked.cpu(),
                        "per_agent_mse": mse_per_agent,
                        "test_loss": total_loss
                    }, f)
                print(f"[Test-Baseline-{self.discriminator_name}] Saved baseline prediction/test results to {save_path}")

        return total_loss, mse_per_agent
