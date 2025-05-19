from itertools import chain
from typing import Type, Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
import wandb
import gin
import gymnasium as gym

from amago.loading import Batch, MAGIC_PAD_VAL
from amago.nets.tstep_encoders import TstepEncoder
from amago.nets.traj_encoders import TrajEncoder
from amago.nets import actor_critic
from amago import utils


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
def binary_filter(adv: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    return adv > threshold


@gin.configurable
def exp_filter(
    adv: torch.Tensor,
    beta: float = 1.0,
    clip_adv_low: Optional[float] = None,
    clip_adv_high: Optional[float] = None,
    clip_weights_low: Optional[float] = 1e-7,
    clip_weights_high: Optional[float] = None,
) -> torch.Tensor:
    """
    Weights policy regression data according to `exp(beta * adv)`.

    NOTE that some papers define the beta hparam according to
    exp( 1/beta * adv ), so check whether you need to invert the value
    to match their setting.

    Clip the raw advantages and/or the resulting weights for stability.
    """
    if clip_adv_low is not None or clip_adv_high is not None:
        adv = torch.clamp(adv, min=clip_adv_low, max=clip_adv_high)
    weights = torch.exp(beta * adv)
    if clip_weights_low is not None or clip_weights_high is not None:
        weights = torch.clamp(weights, min=clip_weights_low, max=clip_weights_high)
    return weights


@gin.configurable
class Agent(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        max_seq_len: int,
        tstep_encoder_type: Type[TstepEncoder],
        traj_encoder_type: Type[TrajEncoder],
        # number of critics in the ensemble
        num_critics: int = 4,
        # number of critics used to compute TD targets (REDQ)
        num_critics_td: int = 2,
        # weight of the "online" aka DPG actor loss (DDPG)
        online_coeff: float = 1.0,
        # weight of the "offline" aka advantage weighted/"filtered" regression term (CRR/AWAC)
        offline_coeff: float = 0.1,
        # the discount factor *of the policy we sample during rollouts/evals*
        gamma: float = 0.999,
        # scale every reward by a constant (for loss function only). This can only help to
        # bring Q-vals into a range that is numerically stable for PopArt.
        # If you know the return scale, edit this to stay somewhere like |Q| in [100, 10000].
        reward_multiplier: float = 10.0,
        # standard polyak update for the target critic(s) (DDPG)
        tau: float = 0.003,
        # skip the computation of the "offline" policy regression weights/"filter".
        # Useful when you're doing imitation learning (all weights become 1.0)
        fake_filter: bool = False,
        # num actions to use anytime we're going to estimate E_[Q(s, a ~ pi)] by sampling continuous action dists
        num_actions_for_cont_value: int = 12,
        # function that takes seq of advantage estimates and outputs the regression weights.
        # offline_loss = -fbc_filter_func(advantages) * log pi(actions | states)
        # defaults to binary_filter, which masks negative advantage values (CRR).
        fbc_filter_func: callable = binary_filter,
        # PopArt adaptive normalization for value network outputs
        popart: bool = True,
        # use polyak averaged actor for TD targets
        use_target_actor: bool = True,
        # train on multiple discount horizons in parallel.
        # values set according to configurable `Multigammas` object above.
        use_multigamma: bool = True,
    ):
        super().__init__()
        self.obs_space = obs_space
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
        self.num_actions_for_cont_value = num_actions_for_cont_value
        self.fbc_filter_func = fbc_filter_func
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.tau = tau
        self.use_target_actor = use_target_actor
        self.max_seq_len = max_seq_len

        self.tstep_encoder = tstep_encoder_type(
            obs_space=obs_space,
            rl2_space=rl2_space,
        )
        self.traj_encoder = traj_encoder_type(
            tstep_dim=self.tstep_encoder.emb_dim,
            max_seq_len=max_seq_len,
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
        assert num_critics_td <= num_critics
        self.num_critics = num_critics
        self.num_critics_td = num_critics_td

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
        self.maximized_critics = actor_critic.NCritics(
            **ac_kwargs, num_critics=num_critics
        )
        if self.multibinary:
            ac_kwargs["cont_dist_kind"] = "multibinary"
        self.actor = actor_critic.Actor(**ac_kwargs)
        self.target_actor = actor_critic.Actor(**ac_kwargs)
        # full weight copy to targets
        self.hard_sync_targets()

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

    def _full_copy(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(param.data)

    def _ema_copy(self, target, online):
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_sync_targets(self):
        """
        Hard copy online actor/critics to target actor/critics
        """
        self._full_copy(self.target_critics, self.critics)
        self._full_copy(self.target_actor, self.actor)
        self._full_copy(self.maximized_critics, self.critics)

    def soft_sync_targets(self):
        """
        EMA copy online actor/critics to target actor/critics (DDPG-style)
        """
        self._ema_copy(self.target_critics, self.critics)
        self._ema_copy(self.target_actor, self.actor)
        # full copy duplicate critic
        self._full_copy(self.maximized_critics, self.critics)

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ):
        """
        Get rollout actions from the current policy.
        """
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        # sequence model embedding [batch, length, d_emb]
        traj_emb_t, hidden_state = self.traj_encoder(
            tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
        )
        # generate action distribution [batch, length, len(self.gammas), d_action]
        action_dists = self.actor(traj_emb_t)
        if sample:
            actions = action_dists.sample()
        else:
            if self.discrete:
                actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
            else:
                actions = action_dists.mean
        # get intended gamma distribution (always in -1 idx)
        actions = actions[..., -1, :]
        dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
        return actions.to(dtype=dtype), hidden_state

    def _sample_k_actions(self, dist, k: int):
        if self.discrete:
            assert k == 1, "There is no need to sample multiple discrete actions"
            a = dist.probs.unsqueeze(0)
        elif self.actor.actions_differentiable:
            a = torch.stack([dist.rsample() for _ in range(k)], dim=0)
        else:
            a = dist.sample((k,))
        return a

    def forward(self, batch: Batch, log_step: bool):
        """
        Main step of training loop. Generate actor and critic loss
        terms in a minimum number of forward passes with a compact
        sequence format. See comments for detailed explanation.
        """
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##########################
        ## Step 0: Timestep Emb ##
        ##########################
        active_log_dict = self.update_info if log_step else None
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)

        ###########################
        ## Step 1: Get Organized ##
        ###########################
        B, L, D_o = o.shape
        # padded actions are `self.pad_val` which will be invalid;
        # clip to valid range now and mask the loss later
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1., 1.)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K = self.num_actions_for_cont_value if not self.discrete else 1
        # note that the last timestep does not have an action.
        # we give it a fake one to make shape math work.
        a_buffer = torch.cat((a, a[:, -1, ...].clone().unsqueeze(1)), axis=1)
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        # arrays used by critic update end up in a (B, L, C, G, dim) format
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat((self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        D_emb = self.traj_encoder.emb_dim
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()
        actor_mask = repeat(state_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask[:, 1:, ...], f"b l 1 -> b l {C} {G} 1")

        ################################
        ## Step 2: Sequence Embedding ##
        ################################
        # one trajectory encoder forward pass
        s_rep, hidden_state = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        assert s_rep.shape == (B, L, D_emb)

        #########################################
        ## Step 3: a' ~ \pi, Q(s, a'), Q(s, a) ##
        #########################################
        critic_loss = None
        a_dist = self.actor(s_rep)
        a_agent = self._sample_k_actions(a_dist, k=K)
        assert a_agent.shape == (K, B, L, G, D_action)

        if not self.fake_filter or self.online_coeff > 0:
            # (s, a ~ pi), Q(s, a ~ pi)
            s_a_agent_g = (s_rep.detach(), a_agent)
            q_s_a_agent_g = self.maximized_critics(*s_a_agent_g).mean(0)
            # (s, a), Q(s, a)
            s_a_g = (s_rep[:, :-1, ...], a_buffer[:, :-1, ...].unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g).mean(0)
            assert q_s_a_agent_g.shape == (B, L, C, G, 1)
            assert q_s_a_g.shape == (B, L-1, C, G, 1)

            ########################
            ## Step 4: TD Targets ##
            ########################
            # td_target = r + gamma * (1 - d) * Q_target(s', a' ~ pi(s'))
            with torch.no_grad():
                # (a ~ pi_target)
                a_prime_dist = self.target_actor(s_rep) if self.use_target_actor else a_dist
                ap = self._sample_k_actions(a_prime_dist, k=K).detach()
                assert ap.shape == (K, B, L, G, D_action)
                # (s', a' ~ pi(s'))
                sp_ap_gp = (s_rep[:, 1:, ...], ap[:, :, 1:, ...])
                # Q_target(s', a' ~ pi(s'))
                q_targ_sp_ap_gp = self.popart(self.target_critics(*sp_ap_gp).mean(0), normalized=False)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                # y = r + gamma * (1 - d) * Q_target(s', a' ~ pi(s'))
                gamma = self.gammas.to(r.device).unsqueeze(-1)
                ensemble_td_target = r + gamma * (1.0 - d) * q_targ_sp_ap_gp
                assert ensemble_td_target.shape == (B, L - 1, C, G, 1)
                # random subset of critic ensemble
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
            critic_loss = (self.popart(q_s_a_g) - td_target_norm.detach()).pow(2)
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            if log_step:
                td_stats = self._td_stats(
                    critic_mask,
                    q_s_a_g,
                    self.popart(q_s_a_g, normalized=False),
                    r=r / self.reward_multiplier,
                    d=d,
                    td_target=td_target,
                )
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
                with torch.no_grad():
                    # V(s) = E_a[Q(s, a ~ pi)] (computed with sample of num_actions_for_cont_value 
                    # in continuous case, or explicitly in discrete case). Already taken mean
                    # over actions, now mean over critic ensemble.
                    val_s_g = q_s_a_agent_g[:, :-1, ...].mean(2).detach()
                    assert val_s_g.shape == (B, L - 1, G, 1)
                    # A(s, a) = Q(s, a) - V(s)
                    advantage_a_s_g = q_s_a_g.mean(2) - val_s_g
                    assert advantage_a_s_g.shape == (B, L - 1, G, 1)
                    filter_ = self.fbc_filter_func(advantage_a_s_g).float()
            else:
                # Behavior Cloning
                filter_ = torch.ones((1, 1, 1, 1), device=a.device)
            if self.discrete:
                # buffer actions are one-hot encoded
                logp_a = a_dist.log_prob(a_buffer.argmax(-1)).unsqueeze(-1)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                logp_a = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
            # clamp for stability and throw away last action that was a duplicate
            logp_a = logp_a[:, :-1, ...].clamp(-1e3, 1e3)
            # filtered nll
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                filter_stats = self._filter_stats(actor_mask, logp_a, filter_)
                self.update_info.update(filter_stats)

        # fmt: on
        return critic_loss, actor_loss

    def _td_stats(self, mask, raw_q_s_a_g, q_s_a_g, r, d, td_target) -> dict:
        # messy data gathering for wandb console
        def masked_avg(x_, dim=0):
            return (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / mask[
                ..., dim, :
            ].sum()

        where_mask = torch.where(mask.all(2, keepdims=True) > 0)

        stats = {}
        for i, gamma in enumerate(self.gammas):
            stats[f"Q(s, a) (global mean, rescaled) gamma={gamma:.3f}"] = masked_avg(
                q_s_a_g, i
            )
            stats[f"Q(s,a) (global mean, raw scale) gamma={gamma:.3f}"] = masked_avg(
                raw_q_s_a_g, i
            )
            stats[
                f"Q(s, a) Ensemble Stdev. (raw scale, ignoring padding) gamma={gamma:.3f}"
            ] = (raw_q_s_a_g[..., i, :].std(2).mean())

        stats.update(
            {
                "Q(s, a) (global std, rescaled, ignoring padding)": q_s_a_g.std(),
                "Min TD Target": td_target[where_mask].min(),
                "Max TD Target": td_target[where_mask].max(),
                "TD Target (test-time gamma)": masked_avg(td_target, -1),
                "Mean Reward (in training sequences)": masked_avg(r),
                "Sequences Containing Done": d[:, :, 0, 0, 0].any(1).sum(),
                "Min Reward (in training sequences)": r[where_mask].min(),
                "Max Reward (in training sequences)": r[where_mask].max(),
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
                "Policy Per-timestep Entropy (test-time gamma)": masked_avg(
                    entropy, -1
                ),
                "Policy Per-timstep Low Prob. (test-time gamma)": masked_avg(
                    low_prob, -1
                ),
                "Policy Per-timestep High Prob. (test-time gamma)": masked_avg(
                    high_prob, -1
                ),
                "Policy Overall Highest Prob.": (mask * a_dist.probs).max(),
            }
        else:
            entropy = -a_dist.log_prob(a_dist.sample()).sum(-1, keepdims=True)
            return {"Policy Entropy (test-time gamma)": masked_avg(entropy, -1)}

    def _filter_stats(self, mask, logp_a, filter_) -> dict:
        mask = mask[:, :-1, ...]

        # messy data gathering for wandb console
        def masked_avg(x_, dim=0):
            return (mask[..., dim, :] * x_[..., dim, :]).sum().detach() / mask[
                ..., dim, :
            ].sum()

        binary_filter = filter_ > 0

        # messy data gathering for wandb console
        stats = {
            "Minimum Action Logprob": logp_a.min(),
            "Maximum Action Logprob": logp_a.max(),
            "Filter Max": filter_.max(),
            "Filter Min": filter_.min(),
            "Filter Mean": filter_.mean(),
            "Pct. of Actions Approved by Binary FBC Filter (All Gammas)": utils.masked_avg(
                binary_filter, mask
            )
            * 100.0,
        }

        if filter_.shape[-2] == len(self.gammas):
            for i, gamma in enumerate(self.gammas):
                stats[
                    f"Pct. of Actions Approved by Binary FBC (gamma = {gamma : .3f})"
                ] = (masked_avg(binary_filter, dim=i) * 100.0)
        return stats

    def _popart_stats(self) -> dict:
        # messy data gathering for wandb console
        return {
            "PopArt mu (mean over gamma)": self.popart.mu.data.mean().item(),
            "PopArt nu (mean over gamma)": self.popart.nu.data.mean().item(),
            "PopArt w (mean over gamma)": self.popart.w.data.mean().item(),
            "PopArt b (mean over gamma)": self.popart.b.data.mean().item(),
            "PopArt sigma (mean over gamma)": self.popart.sigma.mean().item(),
        }


@gin.configurable
class MultiTaskAgent(Agent):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        tstep_encoder_type: Type[TstepEncoder],
        traj_encoder_type: Type[TrajEncoder],
        max_seq_len: int,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 0.0,
        offline_coeff: float = 1.0,
        gamma: float = 0.999,
        reward_multiplier: float = 10.0,
        tau: float = 0.003,
        fake_filter: bool = False,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 3,
        fbc_filter_func: callable = binary_filter,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
    ):
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            action_space=action_space,
            max_seq_len=max_seq_len,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            num_critics=num_critics,
            num_critics_td=num_critics_td,
            gamma=gamma,
            reward_multiplier=reward_multiplier,
            tau=tau,
            fake_filter=fake_filter,
            online_coeff=online_coeff,
            offline_coeff=offline_coeff,
            use_target_actor=use_target_actor,
            use_multigamma=use_multigamma,
            fbc_filter_func=fbc_filter_func,
            popart=popart,
        )
        self.num_actions_for_value_in_critic_loss = num_actions_for_value_in_critic_loss
        self.num_actions_for_value_in_actor_loss = num_actions_for_value_in_actor_loss
        critic_kwargs = {
            "state_dim": self.traj_encoder.emb_dim,
            "action_dim": self.action_dim,
            "gammas": self.gammas,
            "num_critics": self.num_critics,
        }
        self.critics = actor_critic.NCriticsTwoHot(**critic_kwargs)
        self.target_critics = actor_critic.NCriticsTwoHot(**critic_kwargs)
        self.maximized_critics = actor_critic.NCriticsTwoHot(**critic_kwargs)
        self.hard_sync_targets()

    def forward(self, batch: Batch, log_step: bool):
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##########################
        ## Step 0: Timestep Emb ##
        ##########################
        active_log_dict = self.update_info if log_step else None
        o = self.tstep_encoder(
            obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict
        )

        ###########################
        ## Step 1: Get Organized ##
        ###########################
        B, L, D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K_c = self.num_actions_for_value_in_critic_loss
        a_buffer = torch.cat((a, a[:, -1, ...].clone().unsqueeze(1)), axis=1)
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat(
            (self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r"
        )
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        D_emb = self.traj_encoder.emb_dim
        Bins = self.critics.num_bins
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()
        actor_mask = repeat(state_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask[:, 1:, ...], f"b l 1 -> b l {C} {G} 1")

        ################################
        ## Step 2: Sequence Embedding ##
        ################################
        s_rep, hidden_state = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, hidden_state=None)
        assert s_rep.shape == (B, L, D_emb)

        #########################################
        ## Step 3: a' ~ \pi, Q(s, a'), Q(s, a) ##
        #########################################
        a_dist = self.actor(s_rep)
        if self.discrete:
            a_dist = actor_critic.DiscreteLikeContinuous(a_dist)

        critic_loss = None
        if not self.fake_filter or self.online_coeff > 0:
            ########################
            ## Step 4: TD Targets ##
            ########################
            with torch.no_grad():
                if self.use_target_actor:
                    a_prime_dist = self.target_actor(s_rep)
                    if self.discrete:
                        a_prime_dist = actor_critic.DiscreteLikeContinuous(a_prime_dist)
                else:
                    a_prime_dist = a_dist
                ap = self._sample_k_actions(a_prime_dist, k=K_c)
                assert ap.shape == (K_c, B, L, G, D_action)
                sp_ap_gp = (s_rep[:, 1:, ...].detach(), ap[:, :, 1:, ...].detach())
                q_targ_sp_ap_gp = self.target_critics(*sp_ap_gp)
                assert q_targ_sp_ap_gp.probs.shape == (K_c, B, L - 1, C, G, Bins)
                q_targ_sp_ap_gp = self.target_critics.bin_dist_to_raw_vals(q_targ_sp_ap_gp).mean(0)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                gamma = self.gammas.to(r.device).unsqueeze(-1)
                ensemble_td_target = r + gamma * (1.0 - d) * q_targ_sp_ap_gp
                assert ensemble_td_target.shape == (B, L - 1, C, G, 1)
                random_subset = torch.randint(
                    low=0,
                    high=C,
                    size=(B, L - 1, self.num_critics_td, G, 1),
                    device=r.device,
                )
                td_target_rand = torch.take_along_dim(
                    ensemble_td_target, random_subset, dim=2
                )
                td_target = td_target_rand.mean(2, keepdims=True)
                assert td_target.shape == (B, L - 1, 1, G, 1)
                # we are only using popart to track stats for online actor update,
                # since scale intentionally does not impact critic loss
                self.popart.update_stats(
                    td_target, mask=critic_mask.all(2, keepdim=True)
                )
                assert td_target.shape == (B, L - 1, 1, G, 1)
                td_target_labels = self.target_critics.raw_vals_to_labels(td_target)
                td_target_labels = repeat(
                    td_target_labels, f"b l 1 g bins -> b l {C} g bins"
                )
                assert td_target_labels.shape == (B, L - 1, C, G, Bins)

            #########################
            ## Step 5: Critic Loss ##
            #########################
            s_a_g = (s_rep, a_buffer.unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g)
            assert q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
            critic_loss = F.cross_entropy(
                rearrange(q_s_a_g.logits[0, :, :-1, ...], "b l c g u -> (b l c g) u"),
                rearrange(td_target_labels, "b l c g u -> (b l c g) u"),
                reduction="none",
            )
            critic_loss = rearrange(
                critic_loss, "(b l c g) -> b l c g 1", b=B, l=L - 1, c=C, g=G
            )
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            scalar_q_s_a_g = self.critics.bin_dist_to_raw_vals(q_s_a_g).squeeze(0)
            if log_step:
                td_stats = self._td_stats(
                    critic_mask,
                    scalar_q_s_a_g[:, :-1, ...],
                    self.popart.normalize_values(scalar_q_s_a_g)[:, :-1, ...],
                    r=r,
                    d=d,
                    td_target=td_target,
                    raw_q_bins=q_s_a_g.probs[0, :, :-1],
                )
                popart_stats = self._popart_stats()
                self.update_info.update(td_stats | popart_stats)

        ######################
        ## Step 6: FBC Loss ##
        ######################
        actor_loss = 0.0
        K_a = self.num_actions_for_value_in_actor_loss
        a_agent = self._sample_k_actions(a_dist, k=K_a)
        if not self.fake_filter and self.offline_coeff > 0:
            with torch.no_grad():
                q_s_a_agent = self.critics(s_rep.detach(), a_agent)
                assert q_s_a_agent.probs.shape == (K_a, B, L, C, G, Bins)
                # mean over actions and critic ensemble
                val_s = self.critics.bin_dist_to_raw_vals(q_s_a_agent).mean((0, 3))
                assert val_s.shape == (B, L, G, 1)
                # A(s, a) = Q(s, a) - V(s) = mean_over_critics(Q(s, a)) - mean_over_critics(mean_over_actions(Q(s, a ~ pi)))
                advantage_s_a = scalar_q_s_a_g.mean(2) - val_s
                assert advantage_s_a.shape == (B, L, G, 1)
                filter_ = self.fbc_filter_func(advantage_s_a)[:, :-1, ...].float()
                binary_filter_ = binary_filter(advantage_s_a)[:, :-1, ...].float()
        else:
            filter_ = binary_filter_ = torch.ones(
                (B, L - 1, G, 1), dtype=torch.float32, device=s_rep.device
            )

        if self.offline_coeff > 0:
            if self.discrete:
                logp_a = a_dist.log_prob(a_buffer)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                logp_a = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
            logp_a = logp_a[:, :-1, ...].clamp(-1e3, 1e3)
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                policy_stats = self._policy_stats(actor_mask, a_dist)
                filter_stats = self._filter_stats(actor_mask, logp_a, binary_filter_)
                self.update_info.update(filter_stats | policy_stats)

        ######################
        ## Step 7: DPG Loss ##
        ######################
        if self.online_coeff > 0:
            assert (
                self.actor.actions_differentiable and not self.discrete
            ), "this ablation only supports continuous actions with rsample()"
            q_s_a_agent = self.maximized_critics(s_rep.detach(), a_agent)
            q_s_a_agent = self.popart.normalize_values(
                # mean over K actions, min over critic ensemble
                self.maximized_critics.bin_dist_to_raw_vals(q_s_a_agent).mean(0).min(2).values
            )
            actor_loss += self.online_coeff * -(q_s_a_agent[:, :-1, ...])
        

        return critic_loss, actor_loss

    def _td_stats(
        self, mask, raw_q_s_a_g, q_s_a_g, r, d, td_target, raw_q_bins
    ) -> dict:
        stats = super()._td_stats(
            mask=mask,
            raw_q_s_a_g=raw_q_s_a_g,
            q_s_a_g=q_s_a_g,
            r=r,
            d=d,
            td_target=td_target,
        )
        *_, Bins = raw_q_bins.shape
        max_bin_all_gammas_histogram = raw_q_bins.argmax(-1)[torch.where(mask.all(-1))]
        max_bin_target_gamma_histogram = raw_q_bins[..., -1, :].argmax(-1)[
            torch.where(mask[..., -1, :].all(-1))
        ]
        stats.update(
            {
                "Maximum Bin (All Gammas)": wandb.Histogram(
                    max_bin_all_gammas_histogram.cpu().numpy(), num_bins=Bins
                ),
                "Maximum Bin (Target Gamma)": wandb.Histogram(
                    max_bin_target_gamma_histogram.cpu().numpy(), num_bins=Bins
                ),
            }
        )
        return stats
