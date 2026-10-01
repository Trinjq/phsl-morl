"""Train one fixed-preference Brax key policy and persist its best return."""

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from evorl.envs.brax import create_wrapped_brax_env
from evorl.envs.wrappers.training_wrapper import AutoresetMode
from evorl.algorithms.mo_td3 import PreferenceActor, TwinVectorCritic
from evorl.utils.pd_morl_interpolator import key_preferences
from evorl.replay_buffers import ReplayBuffer
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.rollout import rollout, fast_eval_rollout_episode

# 参数设置
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
seed=42
num_envs=64
rollout_len=4
eval_episodes=16
hidden_sizes=(400,400)
lr=3e-4
gamma=0.99
tau=0.005
expl_noise=0.1
policy_noise=0.2
noise_clip=0.5
policy_freq=2
warmup_steps=25088
total_env_steps=2000128
eval_interval=16000
batch_size=100
replay_capacity=500000
horizon=500
reward_size=2
updates_per_chunk=num_envs*rollout_len


def replay_next_obs_and_done(trajectory, obs_dim):
    """Return true successor observations and combined episode boundaries."""
    extras = trajectory.extras.env_extras
    required = ("ori_obs", "termination", "truncation")
    missing = [name for name in required if name not in extras]
    if missing:
        raise ValueError(f"rollout is missing required env extras: {missing}")
    next_obs = extras.ori_obs.reshape(-1, obs_dim)
    done = jnp.maximum(extras.termination, extras.truncation).reshape(-1)
    return next_obs, done

def train_single_key(preference,seed_val):
    w_np=np.asarray(preference,dtype=np.float32)
    w=jnp.asarray(w_np)
    rng=jax.random.PRNGKey(seed_val)
    print(f"\n=======================================================",flush=True)
    print(f"Training Brax Key Preference: {w_np.tolist()}",flush=True)
    print(f"Budget: {total_env_steps} steps (Warmup: {warmup_steps}), Envs: {num_envs}, Seed: {seed_val}",flush=True)
    print(f"=======================================================",flush=True)

    train_env=create_wrapped_brax_env(
        "walker2d",
        episode_length=horizon,
        parallel=num_envs,
        autoreset_mode=AutoresetMode.NORMAL,
        record_ori_obs=True,
        vector_reward=True,
    )
    eval_env=create_wrapped_brax_env(
        "walker2d",
        episode_length=horizon,
        parallel=eval_episodes,
        autoreset_mode=AutoresetMode.DISABLED,
        vector_reward=True,
    )

    act_dim=train_env.action_space.shape[0]
    obs_dim=train_env.obs_space.shape[0]

    actor=PreferenceActor(action_size=act_dim,hidden_layer_sizes=hidden_sizes)
    twin_critic=TwinVectorCritic(reward_size=reward_size,hidden_layer_sizes=hidden_sizes)

    rng,ka,kc=jax.random.split(rng,3)
    dummy_obs=jnp.zeros((obs_dim,))
    dummy_act=jnp.zeros((act_dim,))
    actor_params=actor.init(ka,dummy_obs,w)
    critic_params=twin_critic.init(kc,dummy_obs,w,dummy_act)
    target_actor_params=actor_params
    target_critic_params=critic_params

    actor_opt=optax.chain(optax.clip_by_global_norm(100.0),optax.adam(lr))
    critic_opt=optax.chain(optax.clip_by_global_norm(100.0),optax.adam(lr))
    actor_opt_state=actor_opt.init(actor_params)
    critic_opt_state=critic_opt.init(critic_params)

    rb=ReplayBuffer(capacity=replay_capacity,sample_batch_size=batch_size)
    dummy_batch=SampleBatch(
        obs=dummy_obs,
        actions=dummy_act,
        rewards=jnp.zeros((reward_size,)),
        dones=jnp.zeros(()),
        next_obs=dummy_obs,
    )
    rb_state=rb.init(dummy_batch)

    # 动作函数
    def warmup_action_fn(_params,state_batch,k):
        obs=state_batch.obs
        return jax.random.uniform(k,(*obs.shape[:-1],act_dim),minval=-1.0,maxval=1.0),PyTreeDict()

    def train_action_fn(params,state_batch,k):
        obs=state_batch.obs
        w_b=jnp.broadcast_to(w,(*obs.shape[:-1],reward_size))
        a=actor.apply(params,obs,w_b)
        noise=jax.random.normal(k,a.shape)*expl_noise
        return jnp.clip(a+noise,-1.0,1.0),PyTreeDict()

    def eval_action_fn(params,state_batch,k):
        obs=state_batch.obs
        w_b=jnp.broadcast_to(w,(*obs.shape[:-1],reward_size))
        return actor.apply(params,obs,w_b),PyTreeDict()

    # 训练 Chunk（支持区分 warmup 与 policy update）
    def make_train_chunk(is_warmup):
        @jax.jit
        def train_chunk(carry,unused):
            (k,env_s,ap,cp,tap,tcp,ap_opt,cp_opt,rb_s,it)=carry
            k,kr,ku=jax.random.split(k,3)

            action_fn=warmup_action_fn if is_warmup else lambda _s,sb,rk:train_action_fn(ap,sb,rk)

            traj,env_s=rollout(
                env_fn=train_env.step,
                action_fn=action_fn,
                env_state=env_s,
                agent_state=None,
                key=kr,
                rollout_length=rollout_len,
                env_extra_fields=("ori_obs", "termination", "truncation"),
            )

            flat_obs=traj.obs.reshape(-1,obs_dim)
            flat_act=traj.actions.reshape(-1,act_dim)
            flat_rew=traj.rewards.reshape(-1,reward_size)
            flat_next_obs,flat_done=replay_next_obs_and_done(traj,obs_dim)

            batch_to_add=SampleBatch(
                obs=flat_obs,
                actions=flat_act,
                rewards=flat_rew,
                dones=flat_done,
                next_obs=flat_next_obs,
            )
            rb_s=rb.add(rb_s,batch_to_add)

            if is_warmup:
                return (k,env_s,ap,cp,tap,tcp,ap_opt,cp_opt,rb_s,it),None

            def update_step(inner_carry,ukey):
                cur_cp,cur_cp_opt,cur_ap,cur_ap_opt,cur_tap,cur_tcp,cur_it=inner_carry
                ukey,ks,kn=jax.random.split(ukey,3)
                sample=rb.sample(rb_s,ks)
                w_b=jnp.broadcast_to(w,(batch_size,reward_size))

                next_act=actor.apply(cur_tap,sample.next_obs,w_b)
                act_noise=jnp.clip(jax.random.normal(kn,next_act.shape)*policy_noise,-noise_clip,noise_clip)
                next_act=jnp.clip(next_act+act_noise,-1.0,1.0)

                target_qs=twin_critic.apply(cur_tcp,sample.next_obs,w_b,next_act)
                q1_sc=jnp.sum(target_qs[...,0,:]*w_b,axis=-1)
                q2_sc=jnp.sum(target_qs[...,1,:]*w_b,axis=-1)
                target_q_chosen=jnp.where(q1_sc[...,None]<=q2_sc[...,None],target_qs[...,0,:],target_qs[...,1,:])
                target_vec_q=sample.rewards+gamma*(1.0-sample.dones[...,None])*target_q_chosen
                target_vec_q=jax.lax.stop_gradient(target_vec_q)

                def critic_loss_fn(p):
                    qs=twin_critic.apply(p,sample.obs,w_b,sample.actions)
                    diff1=jnp.abs(qs[...,0,:]-target_vec_q)
                    diff2=jnp.abs(qs[...,1,:]-target_vec_q)
                    h1=jnp.where(diff1<1.0,0.5*jnp.square(diff1),diff1-0.5)
                    h2=jnp.where(diff2<1.0,0.5*jnp.square(diff2),diff2-0.5)
                    return jnp.mean(h1)+jnp.mean(h2)

                c_loss,c_grad=jax.value_and_grad(critic_loss_fn)(cur_cp)
                c_upd,cur_cp_opt=critic_opt.update(c_grad,cur_cp_opt)
                cur_cp=optax.apply_updates(cur_cp,c_upd)

                cur_it=cur_it+1

                def update_actor_fn(ap_in,ap_opt_in,tap_in,tcp_in):
                    def actor_loss_fn(p):
                        acts=actor.apply(p,sample.obs,w_b)
                        q1=twin_critic.apply(cur_cp,sample.obs,w_b,acts)[...,0,:]
                        return -jnp.mean(jnp.sum(q1*w_b,axis=-1))

                    a_loss,a_grad=jax.value_and_grad(actor_loss_fn)(ap_in)
                    a_upd,new_ap_opt=actor_opt.update(a_grad,ap_opt_in)
                    new_ap=optax.apply_updates(ap_in,a_upd)
                    new_tap=jax.tree_util.tree_map(lambda p,tp:tau*p+(1.0-tau)*tp,new_ap,tap_in)
                    new_tcp=jax.tree_util.tree_map(lambda p,tp:tau*p+(1.0-tau)*tp,cur_cp,tcp_in)
                    return new_ap,new_ap_opt,new_tap,new_tcp

                def skip_actor_fn(ap_in,ap_opt_in,tap_in,tcp_in):
                    return ap_in,ap_opt_in,tap_in,tcp_in

                cur_ap,cur_ap_opt,cur_tap,cur_tcp=jax.lax.cond(
                    cur_it%policy_freq==0,
                    update_actor_fn,
                    skip_actor_fn,
                    cur_ap,cur_ap_opt,cur_tap,cur_tcp,
                )
                return (cur_cp,cur_cp_opt,cur_ap,cur_ap_opt,cur_tap,cur_tcp,cur_it),None

            init_inner=(cp,cp_opt,ap,ap_opt,tap,tcp,it)
            update_keys=jax.random.split(ku,updates_per_chunk)
            (cp,cp_opt,ap,ap_opt,tap,tcp,it),_=jax.lax.scan(update_step,init_inner,update_keys)
            return (k,env_s,ap,cp,tap,tcp,ap_opt,cp_opt,rb_s,it),None

        return train_chunk

    warmup_chunk=make_train_chunk(is_warmup=True)
    learn_chunk=make_train_chunk(is_warmup=False)

    # 评估函数
    @jax.jit
    def evaluate(curr_actor_params,eval_rng):
        eval_rng,ke=jax.random.split(eval_rng)
        init_s=eval_env.reset(ke)
        metrics,_=fast_eval_rollout_episode(
            env_fn=eval_env.step,
            action_fn=lambda _s,sb,rk:eval_action_fn(curr_actor_params,sb,rk),
            env_state=init_s,
            agent_state=None,
            key=eval_rng,
            rollout_length=horizon,
        )
        episode_vec_returns=metrics.episode_returns
        mean_ret=jnp.mean(episode_vec_returns,axis=0)
        std_ret=jnp.std(episode_vec_returns,axis=0)
        return mean_ret,std_ret,jnp.dot(mean_ret,w)

    rng,ke=jax.random.split(rng)
    env_state=train_env.reset(ke)

    step_chunk_trans=num_envs*rollout_len
    warmup_chunks=warmup_steps//step_chunk_trans
    total_chunks=total_env_steps//step_chunk_trans
    eval_chunk_freq=max(1,eval_interval//step_chunk_trans)

    carry=(rng,env_state,actor_params,critic_params,target_actor_params,target_critic_params,
           actor_opt_state,critic_opt_state,rb_state,jnp.int32(0))

    best_scalar=-1e9
    best_vec_return=np.zeros((reward_size,),dtype=np.float64)
    start_t=time.time()

    # 1. Warm-up 阶段
    print(f"Running Warm-up ({warmup_steps} steps)...",flush=True)
    carry,_=jax.lax.scan(warmup_chunk,carry,None,length=warmup_chunks)
    print("Warm-up complete. Starting policy learning...",flush=True)

    # 2. 学习与评估阶段
    learning_chunks=total_chunks-warmup_chunks
    chunks_done=0
    round_index=0

    while chunks_done<learning_chunks:
        chunk_count=min(eval_chunk_freq,learning_chunks-chunks_done)
        carry,_=jax.lax.scan(learn_chunk,carry,None,length=chunk_count)
        chunks_done+=chunk_count
        curr_steps=warmup_steps+chunks_done*step_chunk_trans

        # 评估
        k_eval=jax.random.fold_in(rng,round_index+100)
        mean_r,std_r,sc_r=evaluate(carry[2],k_eval)
        mean_r_np=np.asarray(mean_r)
        std_r_np=np.asarray(std_r)
        sc_r_val=float(sc_r)

        if sc_r_val>best_scalar:
            best_scalar=sc_r_val
            best_vec_return=mean_r_np

        print(f"Steps {curr_steps:6d}/{total_env_steps}: Return={np.round(mean_r_np,2)}, Std={np.round(std_r_np,2)}, Scalar={sc_r_val:6.2f}, Best={best_scalar:6.2f}",flush=True)
        round_index+=1

    cost_m=(time.time()-start_t)/60.0
    print(f"Key {w_np.tolist()} finished in {cost_m:.2f} mins. Optimal Vector Return: {best_vec_return.tolist()}",flush=True)
    return best_vec_return

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--key-index",type=int,choices=(0,1,2),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    keys=key_preferences(2) # [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    preference=np.asarray(keys[args.key_index],dtype=np.float32)
    seed_val=seed+args.key_index*17
    result=train_single_key(preference,seed_val=seed_val)
    payload={
        "key_index":args.key_index,
        "preference":preference.tolist(),
        "seed":seed_val,
        "vector_return":np.asarray(result,dtype=np.float64).tolist(),
        "scalarized_return":float(preference@result),
        "total_env_steps":total_env_steps,
        "warmup_steps":warmup_steps,
        "num_envs":num_envs,
        "rollout_length":rollout_len,
        "updates_per_chunk":updates_per_chunk,
        "critic_optimizer_step_count":(
            total_env_steps-warmup_steps
        )//(num_envs*rollout_len)*updates_per_chunk,
        "actor_optimizer_step_count":(
            (total_env_steps-warmup_steps)//(num_envs*rollout_len)*updates_per_chunk
        )//policy_freq,
        "batch_size":batch_size,
        "policy_freq":policy_freq,
        "gamma":gamma,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(f"Saved key result to {args.output}",flush=True)

if __name__=="__main__":
    main()
