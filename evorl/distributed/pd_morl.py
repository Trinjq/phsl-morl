import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class PDMORLDeviceLayout:
    """Physical placement for the frozen K=10, batch=256 algorithm."""

    num_devices: int
    logical_workers: int = 10
    global_batch_size: int = 256

    def __post_init__(self):
        if self.num_devices < 1:
            raise ValueError("num_devices must be positive")

    @property
    def rollout_slots_per_device(self) -> int:
        return math.ceil(self.logical_workers / self.num_devices)

    @property
    def physical_rollout_slots(self) -> int:
        return self.rollout_slots_per_device * self.num_devices

    @property
    def learner_rows_per_device(self) -> int:
        return math.ceil(self.global_batch_size / self.num_devices)

    @property
    def physical_learner_batch(self) -> int:
        return self.learner_rows_per_device * self.num_devices

    @property
    def logical_worker_ids(self) -> jax.Array:
        padding = self.physical_rollout_slots - self.logical_workers
        return jnp.pad(
            jnp.arange(self.logical_workers), (0, padding), constant_values=-1
        )

    @property
    def valid_rollout_mask(self) -> jax.Array:
        return self.logical_worker_ids >= 0

    @property
    def valid_learner_mask(self) -> jax.Array:
        return jnp.arange(self.physical_learner_batch) < self.global_batch_size


def pad_batch(tree: chex.ArrayTree, physical_size: int) -> chex.ArrayTree:
    """Pad a batch by repeating its last row; masks keep those rows inert."""
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return tree
    size = leaves[0].shape[0]
    if size < 1 or physical_size < size:
        raise ValueError(f"cannot pad batch of size {size} to {physical_size}")
    chex.assert_tree_shape_prefix(tree, (size,))
    padding = physical_size - size
    if padding == 0:
        return tree
    return jtu.tree_map(
        lambda x: jnp.concatenate((x, jnp.repeat(x[-1:], padding, axis=0))), tree
    )


def logical_worker_keys(seed: chex.PRNGKey, worker_ids: chex.Array) -> chex.Array:
    """Derive RNG identity from logical worker IDs, never device ordinals."""
    return jax.vmap(lambda worker_id: jax.random.fold_in(seed, worker_id))(worker_ids)


def masked_mean(values: chex.Array, valid_mask: chex.Array) -> chex.Array:
    """Mean over valid rows and every trailing element."""
    mask = valid_mask.reshape((valid_mask.shape[0],) + (1,) * (values.ndim - 1))
    denominator = valid_mask.sum() * math.prod(values.shape[1:])
    return jnp.where(mask, values, 0).sum() / denominator


def replica_max_diff(tree: chex.ArrayTree) -> jax.Array:
    """Maximum difference across a leading replica axis."""
    differences = jtu.tree_leaves(
        jtu.tree_map(lambda x: jnp.max(jnp.abs(x - x[0])), tree)
    )
    return jnp.max(jnp.stack(differences)) if differences else jnp.asarray(0.0)


def make_distributed_gradient_step(
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    devices: Sequence[jax.Device],
    axis_name: str = "DP",
):
    """Build an all-reduced update; optimizer clipping runs after ``psum``.

    ``loss_fn(params, local_batch, local_mask)`` must return additive local
    contributions ``(loss, aux)`` already divided by the global denominator.
    """
    mesh = Mesh(tuple(devices), (axis_name,))

    def local_step(opt_state, params, batch, valid_mask):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, valid_mask
        )
        loss, aux, grads = jtu.tree_map(
            lambda x: jax.lax.psum(x, axis_name), (loss, aux, grads)
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, aux, grads, params, opt_state

    return jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(P(), P(), P(axis_name), P(axis_name)),
        out_specs=(P(), P(), P(), P(), P()),
        check_vma=False,
    )
