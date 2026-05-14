import copy
import os
import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from controllers import REGISTRY as mac_REGISTRY
import torch.nn.functional as F
import torch.nn as nn

class LMAC_learner:
    def __init__(self, mac ,scheme, important_state,logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                assert args.common_reward, "VDN only supports common reward setting"
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                assert args.common_reward, "QMIX only supports common reward setting"
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer) 
            
        self.important_state = important_state
        self.imp_dim = len(important_state)
        
        if self.args.optimiser == "RMSprop":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        else:
            self.optimiser = Adam(params=self.params, lr=args.lr)
            
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if th.cuda.is_available() else "cpu" 
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)    
    

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        states = batch['state'][:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mac_out, pred_states, pred_metas, latent_zs, latent_recons = [], [], [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, pred_state, pred_meta, latent_z, latent_recon = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            pred_states.append(pred_state)
            pred_metas.append(pred_meta)
            latent_zs.append(latent_z)
            latent_recons.append(latent_recon)

        mac_out = th.stack(mac_out, dim=1)
        pred_states = th.stack(pred_states, dim=1)[:, :-1] 
        pred_metas = th.stack(pred_metas, dim=1)[:, :-1]
        latent_zs = th.stack(latent_zs, dim=1)[:, :-1]
        latent_recons = th.stack(latent_recons, dim=1)[:, :-1]

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, *_ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        target_mac_out = target_mac_out.masked_fill(avail_actions[:, 1:] == 0, -9999999)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = mac_out_detach.masked_fill(avail_actions == 0, -9999999)
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        targets = (
            rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()
        )

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)


        td_error = chosen_action_qvals - targets.detach()
        mask_td = mask.expand_as(td_error)
        masked_td_error = td_error * mask_td
        loss = (masked_td_error**2).sum() / mask_td.sum()


        state_info = states[:, :, self.important_state]
        state_info_exp = state_info.unsqueeze(2)
        mask_mse = mask.expand_as(state_info) 
        mask_mse_exp = mask_mse.unsqueeze(2)   

        state_pred_loss = (pred_states - state_info_exp) * mask_mse_exp
        state_pred_loss = state_pred_loss ** 2 

        mask_mse_exp = mask_mse_exp.expand_as(state_pred_loss)

        mse_per_agent = (state_pred_loss.sum(dim=(0, 1, 3)) / mask_mse_exp.sum(dim=(0, 1, 3)))
        mse_loss = state_pred_loss.sum() / mask_mse_exp.sum()

        mse_per_state_log = (state_pred_loss.sum(dim=(0, 1, 2)) / mask_mse_exp.sum(dim=(0, 1, 2)))
        
        mse_per_state = (state_pred_loss < self.args.mse_thres).float() 
        bce_elem = -(mse_per_state * pred_metas.clamp(1e-7, 1-1e-7).log() +
                (1 - mse_per_state) * (1 - pred_metas).clamp(1e-7, 1-1e-7).log())
        masked_bce = bce_elem * mask_mse_exp  # (bs, t, n_agents, output_dim)
        meta_loss = masked_bce.sum() / mask_mse_exp.sum() 
        
        consistency_loss = F.mse_loss(latent_zs, latent_recons)
        
        total_mse_loss = self.args.recon_lambda * mse_loss + self.args.meta_lambda * meta_loss + self.args.consistency_lambda * consistency_loss 

        tot_loss = loss + total_mse_loss 

        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
    
                    
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("Model_D_loss", total_mse_loss.item(), t_env)
            self.logger.log_stat("Mse_loss", mse_loss.item(), t_env)
            self.logger.log_stat("Meta_loss", meta_loss.item(), t_env)
            self.logger.log_stat("Consistency_loss", consistency_loss.item(), t_env)

            for i in range(self.n_agents):
                self.logger.log_stat(f"MSE_per_agent_{i}", mse_per_agent[i].item(), t_env)

            for i in range(self.imp_dim):
                self.logger.log_stat(f"Mse_per_state_{self.important_state[i]}", mse_per_state_log[i].item(), t_env)
                
        # Update target
        if self.args.optimiser == "RMSprop":
            if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = episode_num
        else:
            self.training_steps += 1
            if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_step = self.training_steps
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)

            
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask_td.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask_td).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask_td).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
        
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        
    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )