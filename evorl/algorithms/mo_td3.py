from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from omegaconf import DictConfig

from evorl.agent import AgentState
from evorl.envs import AutoresetMode, Box, Space, create_env
from evorl.evaluators import Evaluator
from evorl.networks import MLP
from evorl.replay_buffers import ReplayBuffer
from evorl.replay_buffers.her import add_her_transitions
from evorl.sample_batch import SampleBatch
from evorl.types import (
    Action,
    LossDict,
    PolicyExtraInfo,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_get
from evorl.utils.morl_math import scalarize

from .td3 import TD3Agent, TD3NetworkParams, TD3Workflow


def select_pessimistic_q_vector(
    twin_q_values: chex.Array, preference: chex.Array
) -> chex.Array:
    """Select one complete critic vector using its scalarized value."""
    critic_index = jnp.argmin(scalarize(twin_q_values, preference), axis=-1)
    return jnp.take_along_axis(
        twin_q_values, critic_index[..., None, None], axis=-2
    )[..., 0, :]


def vector_bellman_target(
    rewards: chex.Array,
    done: chex.Array,
    next_q_values: chex.Array,
    discount: float,
) -> chex.Array:
    return rewards + discount * (1 - done)[..., None] * next_q_values


def twin_smooth_l1_loss(
    twin_q_values: chex.Array, target_q_values: chex.Array
) -> chex.Array:
    """Sum the mean Smooth-L1 loss of both critics."""
    losses = optax.huber_loss(twin_q_values, target_q_values[..., None, :])
    return losses.mean(axis=(0, 2)).sum()


def add_target_policy_smoothing(
    actions: chex.Array,
    key: chex.PRNGKey,
    policy_noise: float,
    clip_policy_noise: float,
) -> chex.Array:
    noise = jnp.clip(
        jax.random.normal(key, actions.shape) * policy_noise,
        -clip_policy_noise,
        clip_policy_noise,
    )
    return jnp.clip(actions + noise, -1.0, 1.0)


def parallel_actor_update_mask(
    round_index: chex.Array, process_count: int, policy_freq: int
) -> chex.Array:
    """Actor-update positions for K source-faithful learner calls."""
    update_ids = round_index * process_count + jnp.arange(1, process_count + 1)
    return update_ids % policy_freq == 0


def select_warmup_actions(
    policy_actions: chex.Array,
    random_actions: chex.Array,
    worker_steps: chex.Array,
    start_timesteps: int,
) -> chex.Array:
    """Select random actions for logical workers still in source warm-up."""
    worker_ids = jnp.arange(policy_actions.shape[0]) % worker_steps.shape[0]
    random_mask = worker_steps[worker_ids] < start_timesteps
    return jnp.where(random_mask[..., None], random_actions, policy_actions)


class PreferenceActor(nn.Module):
    action_size: int
    hidden_layer_sizes: tuple[int, ...] = (400, 400)
    max_action: float = 1.0

    @nn.compact
    def __call__(self, obs: chex.Array, preference: chex.Array) -> chex.Array:
        inputs = jnp.concatenate((obs, preference), axis=-1)
        return self.max_action * MLP(
            layer_sizes=(*self.hidden_layer_sizes, self.action_size),
            activation=nn.relu,
            activation_final=nn.tanh,
            kernel_init=jax.nn.initializers.xavier_normal(),
            name="actor",
        )(inputs)


class TwinVectorCritic(nn.Module):
    reward_size: int = 2
    hidden_layer_sizes: tuple[int, ...] = (400, 400)

    @nn.compact
    def __call__(
        self,
        obs: chex.Array,
        preference: chex.Array,
        actions: chex.Array,
    ) -> chex.Array:
        inputs = jnp.concatenate((obs, preference, actions), axis=-1)
        q_values = [
            MLP(
                layer_sizes=(*self.hidden_layer_sizes, self.reward_size),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.xavier_normal(),
                name=f"critic_{critic_id}",
            )(inputs)
            for critic_id in range(2)
        ]
        return jnp.stack(q_values, axis=-2)


class MOTD3Agent(TD3Agent):
    """Preference-conditioned vector-Q extension of EvoRL TD3."""

    obs_preprocessor: Any = pytree_field(default=None, static=True)
    reward_size: int = pytree_field(default=2, static=True)
    process_count: int = pytree_field(default=1, static=True)
    start_timesteps: int = pytree_field(default=10000, static=True)
    action_low: chex.Array | None = None
    action_high: chex.Array | None = None

    def init(
        self, obs_space: Space, action_space: Space, key: chex.PRNGKey
    ) -> AgentState:
        key, critic_key, actor_key = jax.random.split(key, num=3)
        dummy_obs = jtu.tree_map(lambda x: x[None, ...], obs_space.sample(key))
        dummy_action = action_space.sample(key)[None, ...]
        dummy_preference = jnp.full((1, self.reward_size), 1 / self.reward_size)

        critic_params = self.critic_network.init(
            critic_key, dummy_obs, dummy_preference, dummy_action
        )
        actor_params = self.actor_network.init(
            actor_key, dummy_obs, dummy_preference
        )
        params_state = TD3NetworkParams(
            critic_params=critic_params,
            actor_params=actor_params,
            target_critic_params=critic_params,
            target_actor_params=actor_params,
        )

        obs_preprocessor_state = (
            running_statistics.init_state(tree_get(dummy_obs, 0))
            if self.normalize_obs
            else None
        )
        return AgentState(
            params=params_state,
            obs_preprocessor_state=obs_preprocessor_state,
            extra_state=PyTreeDict(
                worker_steps=jnp.zeros((self.process_count,), dtype=jnp.uint32)
            ),
        )

    @staticmethod
    def _preference(sample_batch: SampleBatch) -> chex.Array:
        return sample_batch.extras.policy_extras.preference

    def compute_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        policy_actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        noise_key, random_key = jax.random.split(key)
        policy_actions += (
            jax.random.normal(noise_key, policy_actions.shape)
            * self.exploration_epsilon
        )
        policy_actions = jnp.clip(
            policy_actions, self.action_low, self.action_high
        )
        random_actions = jax.random.uniform(
            random_key,
            policy_actions.shape,
            minval=self.action_low,
            maxval=self.action_high,
        )
        actions = select_warmup_actions(
            policy_actions,
            random_actions,
            agent_state.extra_state.worker_steps,
            self.start_timesteps,
        )
        return actions, PyTreeDict(preference=preference)

    def evaluate_actions(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> tuple[Action, PolicyExtraInfo]:
        del key
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        return actions, PyTreeDict(preference=preference)

    def target_actions(
        self,
        agent_state: AgentState,
        next_obs: chex.Array,
        preference: chex.Array,
        key: chex.PRNGKey,
    ) -> chex.Array:
        actions = self.actor_network.apply(
            agent_state.params.target_actor_params, next_obs, preference
        )
        return add_target_policy_smoothing(
            actions, key, self.policy_noise, self.clip_policy_noise
        )

    def critic_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        obs = sample_batch.obs
        next_obs = sample_batch.extras.env_extras.ori_obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)
            next_obs = self.obs_preprocessor(
                next_obs, agent_state.obs_preprocessor_state
            )

        next_actions = self.target_actions(
            agent_state, next_obs, preference, key
        )
        next_twin_q = self.critic_network.apply(
            agent_state.params.target_critic_params,
            next_obs,
            preference,
            next_actions,
        )
        next_q = select_pessimistic_q_vector(next_twin_q, preference)
        env_extras = sample_batch.extras.env_extras
        done = jnp.maximum(env_extras.termination, env_extras.truncation)
        q_target = vector_bellman_target(
            sample_batch.rewards, done, next_q, self.discount
        )
        q_target = jax.lax.stop_gradient(q_target)

        q_values = self.critic_network.apply(
            agent_state.params.critic_params,
            obs,
            preference,
            sample_batch.actions,
        )
        critic_loss = twin_smooth_l1_loss(q_values, q_target)
        return PyTreeDict(
            critic_loss=critic_loss,
            q_value=scalarize(q_values, preference).mean(),
            q_target=q_target,
        )

    def actor_loss(
        self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
    ) -> LossDict:
        del key
        obs = sample_batch.obs
        preference = self._preference(sample_batch)
        if self.normalize_obs:
            obs = self.obs_preprocessor(obs, agent_state.obs_preprocessor_state)

        actions = self.actor_network.apply(
            agent_state.params.actor_params, obs, preference
        )
        q_values = self.critic_network.apply(
            agent_state.params.critic_params, obs, preference, actions
        )
        actor_loss = -scalarize(q_values[..., 0, :], preference).mean()
        return PyTreeDict(actor_loss=actor_loss)


def make_mo_td3_agent(
    action_space: Space,
    actor_hidden_layer_sizes: tuple[int, ...] = (400, 400),
    critic_hidden_layer_sizes: tuple[int, ...] = (400, 400),
    reward_size: int = 2,
    discount: float = 0.995,
    exploration_epsilon: float = 0.1,
    policy_noise: float = 0.2,
    clip_policy_noise: float = 0.5,
    normalize_obs: bool = False,
    process_count: int = 1,
    start_timesteps: int = 10000,
) -> MOTD3Agent:
    assert isinstance(action_space, Box), "Only continuous action spaces are supported."
    max_action = float(jnp.max(action_space.high))
    actor_network = PreferenceActor(
        action_size=action_space.shape[0],
        hidden_layer_sizes=tuple(actor_hidden_layer_sizes),
        max_action=max_action,
    )
    critic_network = TwinVectorCritic(
        reward_size=reward_size,
        hidden_layer_sizes=tuple(critic_hidden_layer_sizes),
    )
    return MOTD3Agent(
        critic_network=critic_network,
        actor_network=actor_network,
        obs_preprocessor=(running_statistics.normalize if normalize_obs else None),
        reward_size=reward_size,
        process_count=process_count,
        start_timesteps=start_timesteps,
        action_low=action_space.low,
        action_high=action_space.high,
        discount=discount,
        exploration_epsilon=exploration_epsilon,
        policy_noise=policy_noise,
        clip_policy_noise=clip_policy_noise,
        critics_in_actor_loss="first",
    )


class MOTD3Workflow(TD3Workflow):
    env_extra_fields = ("ori_obs", "termination", "truncation")

    @classmethod
    def name(cls):
        return "MO-TD3"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env_kwargs = dict(
            episode_length=config.env.max_episode_steps,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
            vector_reward=True,
            episode_preference=True,
            process_count=config.process_count,
        )
        env = create_env(config.env, parallel=config.num_envs, **env_kwargs)
        agent = make_mo_td3_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            reward_size=config.reward_size,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            normalize_obs=config.normalize_obs,
            process_count=config.process_count,
            start_timesteps=config.start_timesteps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
            optax.adam(config.optimizer.lr),
        )
        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=max(
                config.batch_size, config.learner_start_replay_entries
            ),
            sample_batch_size=config.batch_size,
        )
        eval_env = create_env(
            config.env,
            parallel=config.num_eval_envs,
            episode_length=config.env.max_episode_steps,
            autoreset_mode=AutoresetMode.DISABLED,
            vector_reward=True,
            episode_preference=True,
            process_count=config.process_count,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )
        return cls(env, agent, optimizer, evaluator, replay_buffer, config)

    def _setup_replaybuffer(self, key: chex.PRNGKey):
        dummy_obs = self.env.obs_space.sample(key)
        dummy_done = jnp.zeros(())
        return self.replay_buffer.init(
            SampleBatch(
                obs=dummy_obs,
                actions=jnp.zeros(self.env.action_space.shape),
                rewards=jnp.zeros((self.config.reward_size,)),
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(
                        preference=jnp.zeros((self.config.reward_size,))
                    ),
                    env_extras=PyTreeDict(
                        ori_obs=dummy_obs,
                        termination=dummy_done,
                        truncation=dummy_done,
                    ),
                ),
            )
        )

    def _add_to_replay_buffer(self, replay_buffer_state, trajectory, key):
        return add_her_transitions(
            self.replay_buffer,
            replay_buffer_state,
            trajectory,
            jax.random.fold_in(key, 0x484552),
            self.config.num_relabel_preferences,
            self.config.her_start_timesteps,
            self.config.process_count,
        )

    def _postsetup_replaybuffer(self, state: State) -> State:
        state = super()._postsetup_replaybuffer(state)
        prefill_steps = (
            self.config.learning_start_timesteps + self.config.num_envs - 1
        ) // self.config.num_envs
        return state.replace(
            agent_state=state.agent_state.replace(
                extra_state=state.agent_state.extra_state.replace(
                    worker_steps=jnp.full(
                        (self.config.process_count,),
                        prefill_steps,
                        dtype=jnp.uint32,
                    )
                )
            )
        )

    def step(self, state: State):
        train_metrics, state = super().step(state)
        agent_state = state.agent_state.replace(
            extra_state=state.agent_state.extra_state.replace(
                worker_steps=state.agent_state.extra_state.worker_steps
                + jnp.uint32(self.config.rollout_length)
            )
        )
        return train_metrics, state.replace(agent_state=agent_state)

    def _parallel_actor_update_mask(self, state):
        return parallel_actor_update_mask(
            state.metrics.iterations,
            self.config.process_count,
            self.config.actor_update_interval,
        )
