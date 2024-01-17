from itertools import chain
from dataclasses import dataclass

import torch
from torch import nn
from einops import repeat, rearrange
import numpy as np
import gin
import gymnasium as gym

from amago.loading import Batch, MAGIC_PAD_VAL
from amago.nets.tstep_encoders import *
from amago.nets.traj_encoders import *
from amago.nets import actor_critic


@gin.configurable
class Multigammas:
    def __init__(
        self,
        # fmt: off
        discrete = [.1, .9, .95, .97, .99, .995],
        continuous = [.1, .9, .95, .97, .99, .995],
        # fmt: on
    ):
        self.discrete = discrete
        self.continuous = continuous


@gin.configurable
class Agent(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        goal_space: gym.spaces.Box,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        horizon: int,
        tstep_encoder_Cls=FFTstepEncoder,
        traj_encoder_Cls=TformerTrajEncoder,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 1.0,
        offline_coeff: float = 0.1,
        gamma: float = 0.999,
        reward_multiplier: float = 10.0,
        tau: float = 0.003,
        fake_filter: bool = False,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.goal_space = goal_space
        self.rl2_space = rl2_space

        self.action_space = action_space
        self.multibinary = False
        self.discrete = False
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = self.action_space.n
            self.discrete = True
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            self.action_dim = self.action_space.n
            self.multibinary = True
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[-1]
        else:
            raise ValueError(f"Unsupported action space: `{type(self.action_space)}`")

        self.reward_multiplier = reward_multiplier
        self.pad_val = MAGIC_PAD_VAL
        self.fake_filter = fake_filter
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.tau = tau
        self.use_target_actor = use_target_actor
        assert num_critics_td <= num_critics
        self.num_critics_td = num_critics_td

        self.tstep_encoder = tstep_encoder_Cls(
            obs_space=obs_space,
            goal_space=goal_space,
            rl2_space=rl2_space,
        )
        self.traj_encoder = traj_encoder_Cls(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
            horizon=horizon,
        )
        self.emb_dim = self.traj_encoder.emb_dim

        if self.discrete:
            multigammas = Multigammas().discrete
        else:
            multigammas = Multigammas().continuous
        # provided hparam `gamma` will stay in the -1 index
        # of gammas, actor, and critic outputs.
        gammas = (multigammas if use_multigamma else []) + [gamma]
        self.gammas = torch.Tensor(gammas).float()

        self.popart = actor_critic.PopArtLayer(gammas=len(gammas), enabled=popart)

        ac_kwargs = {
            "state_dim": self.traj_encoder.emb_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.gammas,
        }
        self.critics = actor_critic.NCritics(**ac_kwargs, num_critics=num_critics)
        self.target_critics = actor_critic.NCritics(
            **ac_kwargs, num_critics=num_critics
        )
        if self.multibinary:
            ac_kwargs["cont_dist_kind"] = "multibinary"
        self.actor = actor_critic.Actor(**ac_kwargs)
        self.target_actor = actor_critic.Actor(**ac_kwargs)
        # full weight copy to targets
        self.hard_sync_targets()

    def get_current_timestep(
        self, sequences: torch.Tensor | dict[torch.Tensor], seq_lengths: torch.Tensor
    ):
        dict_based = isinstance(sequences, dict)
        if not dict_based:
            sequences = {"dummy": sequences}
        timesteps = {}
        for k, v in sequences.items():
            missing_dims = v.ndim - seq_lengths.ndim
            seq_lengths_ = seq_lengths.reshape(seq_lengths.shape + (1,) * missing_dims)
            timesteps[k] = torch.take_along_dim(v, seq_lengths_ - 1, dim=1)
        if not dict_based:
            timesteps = timesteps["dummy"]
        return timesteps

    @property
    def trainable_params(self):
        """
        Returns iterable of all trainable parameters that should be passed to an optimzer. (Everything but the target networks).
        """
        return chain(
            self.tstep_encoder.parameters(),
            self.traj_encoder.parameters(),
            self.critics.parameters(),
            self.actor.parameters(),
        )

    def hard_sync_targets(self):
        """
        Hard copy online actor/critics to target actor/critics
        """
        for target_param, param in zip(
            self.target_critics.parameters(), self.critics.parameters()
        ):
            target_param.data.copy_(param.data)
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(param.data)

    def _ema_copy(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def soft_sync_targets(self):
        """
        EMA copy online actor/critics to target actor/critics (DDPG-style)
        """
        self._ema_copy(self.target_critics, self.critics)
        self._ema_copy(self.target_actor, self.actor)

    def get_actions(
        self,
        obs,
        goals,
        rl2s,
        seq_lengths,
        time_idxs,
        hidden_state=None,
        sample: bool = True,
    ):
        """
        Get rollout actions from the current policy.
        """
        using_hidden = hidden_state is not None
        if using_hidden:
            obs = self.get_current_timestep(obs, seq_lengths)
            goals = self.get_current_timestep(goals, seq_lengths)
            rl2s = self.get_current_timestep(rl2s, seq_lengths)
            time_idxs = self.get_current_timestep(time_idxs, seq_lengths.squeeze(-1))
        tstep_emb = self.tstep_encoder(obs=obs, goals=goals, rl2s=rl2s)

        # sequence model embedding [batch, length, d_emb]
        traj_emb_t, hidden_state = self.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        if not using_hidden:
            traj_emb_t = self.get_current_timestep(traj_emb_t, seq_lengths)

        # generate action distribution [batch, len(self.gammas), d_action]
        action_dists = self.actor(traj_emb_t.squeeze(1))
        if sample:
            actions = action_dists.sample()
        else:
            if self.discrete:
                actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
            else:
                actions = action_dists.mean

        # get intended gamma distribution (always in -1 idx)
        actions = actions[..., -1, :].cpu().numpy()
        if self.discrete:
            actions = actions.astype(np.uint8)
        else:
            actions = actions.astype(np.float32)
        return actions, hidden_state

    def _td_stats(self, mask, q_s_a_g, r, td_target) -> dict:
        # messy data gathering for wandb console
        def masked_avg(x_, dim=0):
            return (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / mask[
                ..., dim, :
            ].sum()

        q_seq = self.popart(q_s_a_g.detach(), normalized=False)
        stats = {}
        for i, gamma in enumerate(self.gammas):
            stats[f"q_s_a_g gamma={gamma}"] = masked_avg(q_s_a_g, i)
            stats[f"q_s_a_g (rescaled) gamma={gamma}"] = masked_avg(
                q_seq.mean(2, keepdims=True), i
            )
            stats[f"q_seq_mean gamma={gamma}"] = q_seq[..., i, :].mean(2)
            stats[f"q_seq_std gamma={gamma}"] = q_seq[..., i, :].std(2)

        stats.update(
            {
                "q_s_a_g unmasked std": q_s_a_g.std(),
                "min_td_target": (mask * td_target).min(),
                "mean_r": masked_avg(r),
                "td_target (target gamma)": masked_avg(td_target, -1),
                "real_return": torch.flip(
                    torch.cumsum(torch.flip(mask.all(2, keepdims=True) * r, (1,)), 1),
                    (1,),
                ).squeeze(-1),
                "q_s_a_g popart (target gamma)": masked_avg(self.popart(q_s_a_g), -1),
            }
        )
        return stats

    def _policy_stats(self, mask, a_dist) -> dict:
        # messy data gathering for wandb console
        # mask shape is batch length gammas 1
        sum_ = mask.sum((0, 1))
        masked_avg = (
            lambda x_, dim: (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / sum_
        )

        if self.discrete:
            entropy = a_dist.entropy().unsqueeze(-1)
            low_prob = torch.min(a_dist.probs, dim=-1, keepdims=True).values
            high_prob = torch.max(a_dist.probs, dim=-1, keepdims=True).values
            return {
                "pi_entropy (target gamma)": masked_avg(entropy, -1),
                "pi_low_prob (target gamma)": masked_avg(low_prob, -1),
                "pi_high_prob (target gamma)": masked_avg(high_prob, -1),
                "pi_overall_high": (mask * a_dist.probs).max(),
            }
        else:
            entropy = -a_dist.log_prob(a_dist.sample()).sum(-1, keepdims=True)
            return {"pi_entropy (target_gamma)": masked_avg(entropy, -1)}

    def _filter_stats(self, mask, logp_a, filter_) -> dict:
        # messy data gathering for wandb console
        return {
            "filter": (mask[:, :-1, :] * filter_).sum() / mask[:, :-1, :].sum(),
            "min_logp_a": logp_a.min(),
            "max_logp_a": logp_a.max(),
        }

    def _popart_stats(self) -> dict:
        # messy data gathering for wandb console
        return {
            "popart_mu (mean over gamma)": self.popart.mu.data.mean().item(),
            "popart_nu (mean over gamma)": self.popart.nu.data.mean().item(),
            "popart_w (mean over gamma)": self.popart.w.data.mean().item(),
            "popart_b (mean over gamma)": self.popart.b.data.mean().item(),
            "popart_sigma (mean over gamma)": self.popart.sigma.mean().item(),
        }

    def forward(self, batch: Batch, log_step: bool):
        """
        Main step of training loop. Generate actor and critic loss
        terms in a minimum number of forward passes with a compact
        sequence format. See comments for detailed explanation.
        """
        self.update_info = {}  # holds wandb stats

        ##########################
        ## Step 0: Timestep Emb ##
        ##########################
        o = self.tstep_encoder(obs=batch.obs, goals=batch.goals, rl2s=batch.rl2s)

        ###########################
        ## Step 1: Get Organized ##
        ###########################
        B, L, D_o = o.shape
        # padded actions are `self.pad_val` which will be invalid;
        # clip to valid range now and mask the loss later
        if self.discrete:
            a = batch.actions.clamp(0, 1.0)
        else:
            a = batch.actions.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        # note that the last timestep does not have an action.
        # we give it a fake one to make shape math work.
        a_buffer = torch.cat((a, a[:, -1, ...].clone().unsqueeze(1)), axis=1)
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        # arrays used by critic update end up in a (B, L, C, G, dim) format
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat(
            (self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r"
        )
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        D_emb = self.traj_encoder.emb_dim
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()
        actor_mask = repeat(state_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask[:, 1:, ...], f"b l 1 -> b l {C} {G} 1")

        ################################
        ## Step 2: Sequence Embedding ##
        ################################
        # one trajectory encoder forward pass
        s_rep, hidden_state = self.traj_encoder(
            seq=o, time_idxs=batch.time_idxs, hidden_state=None
        )
        assert s_rep.shape == (B, L, D_emb)

        #########################################
        ## Step 3: a' ~ \pi, Q(s, a'), Q(s, a) ##
        #########################################
        critic_loss = None
        # one actor forward pass
        a_dist = self.actor(s_rep)

        if self.discrete:
            a_agent = a_dist.probs
        elif self.actor.actions_differentiable:
            a_agent = a_dist.rsample()
        else:
            a_agent = a_dist.sample()

        if not self.fake_filter or self.online_coeff > 0:
            # in practice two critic passes is same speed on forward, faster on backward
            s_a_agent_g = (s_rep.detach(), a_agent)
            # detach() above b/c these grads flow to traj_encoder through the policy
            s_a_g = (s_rep[:, :-1, ...], a_buffer[:, :-1, ...])
            # all the `phi` terms are only here b/c we used to implement DR3
            q_s_a_agent_g, phi_s_a_agent_g = self.critics(*s_a_agent_g)
            assert q_s_a_agent_g.shape == (B, L, C, G, 1)
            q_s_a_g, phi_s_a_g = self.critics(*s_a_g)

            ########################
            ## Step 4: TD Targets ##
            ########################
            # \mathcal{B}\bar{Q}(s, a, g)
            with torch.no_grad():
                a_prime_dist = (
                    self.target_actor(s_rep) if self.use_target_actor else a_dist
                )
                ap = a_prime_dist.probs if self.discrete else a_prime_dist.sample()
                assert ap.shape == (B, L, G, D_action)
                sp_ap_gp = (s_rep[:, 1:, ...].detach(), ap[:, 1:, ...].detach())
                q_targ_sp_ap_gp, _ = self.target_critics(*sp_ap_gp)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                q_targ_sp_ap_gp = self.popart(q_targ_sp_ap_gp, normalized=False)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                gamma = self.gammas.to(r.device).unsqueeze(-1)
                ensemble_td_target = r + gamma * (1.0 - d) * q_targ_sp_ap_gp
                assert ensemble_td_target.shape == (B, L - 1, C, G, 1)
                # redq random subset of critic ensemble. multigamma format
                # makes this even more random.
                random_subset = torch.randint(
                    low=0,
                    high=C,
                    size=(B, L - 1, self.num_critics_td, G, 1),
                    device=r.device,
                )
                td_target_rand = torch.take_along_dim(
                    ensemble_td_target, random_subset, dim=2
                )
                if self.online_coeff > 0:
                    # clipped double q
                    td_target = td_target_rand.min(2, keepdims=True).values
                else:
                    # without DPG updates the usual min creates strong underestimation. take mean instead
                    td_target = td_target_rand.mean(2, keepdims=True)
                assert td_target.shape == (B, L - 1, 1, G, 1)
                self.popart.update_stats(
                    td_target, mask=critic_mask.all(2, keepdim=True)
                )
                td_target_norm = self.popart.normalize_values(td_target)
                assert td_target_norm.shape == (B, L - 1, 1, G, 1)

            #########################
            ## Step 5: Critic Loss ##
            #########################
            assert q_s_a_g.shape == (B, L - 1, C, G, 1)
            critic_loss = (self.popart(q_s_a_g) - td_target_norm.detach()).pow(2)
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            if log_step:
                td_stats = self._td_stats(critic_mask, q_s_a_g, r, td_target)
                popart_stats = self._popart_stats()
                self.update_info.update(td_stats | popart_stats)

        ######################
        ## Step 6: DPG Loss ##
        ######################
        actor_loss = torch.zeros((B, L - 1, G, 1), device=a.device)
        if self.online_coeff > 0:
            assert (
                self.actor.actions_differentiable
            ), "online-style actor loss is not compatible with action distribution"
            actor_loss += self.online_coeff * -(
                self.popart(q_s_a_agent_g[:, :-1, ...].min(2).values)
            )
        if log_step:
            self.update_info.update(self._policy_stats(actor_mask, a_dist))

        ######################
        ## Step 7: FBC Loss ##
        ######################
        if self.offline_coeff > 0:
            if not self.fake_filter:
                # Critic Regularized Regression (but noisy w/ k = 1 to save a forward pass)
                with torch.no_grad():
                    val_s_g = q_s_a_agent_g[:, :-1, ...].mean(2).detach()
                    assert val_s_g.shape == (B, L - 1, G, 1)
                    advantage_a_s_g = q_s_a_g.mean(2) - val_s_g
                    assert advantage_a_s_g.shape == (B, L - 1, G, 1)
                    filter_ = (advantage_a_s_g > 1e-3).float()
            else:
                # Behavior Cloning
                filter_ = torch.ones((1, 1, 1, 1), device=a.device)
            if self.discrete:
                # buffer actions are one-hot encoded
                logp_a = a_dist.log_prob(a_buffer.argmax(-1)).unsqueeze(-1)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                # action probs at the [-1, 1] border can be very unstable with some
                # distribution implementations
                logp_a = a_dist.log_prob(a_buffer.clamp(-0.995, 0.995)).sum(
                    -1, keepdim=True
                )
            # clamp for stability and throw away last action that was a duplicate
            logp_a = logp_a[:, :-1, ...].clamp(-1e3, 1e3)
            # filtered nll
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                filter_stats = self._filter_stats(actor_mask, logp_a, filter_)
                self.update_info.update(filter_stats)

        return critic_loss, actor_loss
