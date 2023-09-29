import argparse
import os
import time

import gym
import numpy as np
import torch

import TD


def train_online(RL_agent, env, eval_env, args):
	times = []
	steps = []
	evals = []

	start_time = time.time()
	allow_train = False

	state, ep_finished = env.reset(), False
	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

	for t in range(int(args.max_timesteps + 1)):
		maybe_evaluate_and_print(RL_agent, eval_env, times, steps, evals, t, start_time, args)
		
		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, reward, ep_finished, _ = env.step(action) 
		
		ep_total_reward += reward
		ep_timesteps += 1

		done = float(ep_finished) if ep_timesteps < env._max_episode_steps else 0
		RL_agent.replay_buffer.add(state, action, next_state, reward, done)

		state = next_state

		if allow_train:
			RL_agent.train()

		if ep_finished:

			if allow_train and args.use_checkpoints and "TD7":
				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

			if t >= args.timesteps_before_training:
				allow_train = True

			state, done = env.reset(), False
			ep_total_reward, ep_timesteps = 0, 0
			ep_num += 1

	with open(f"./results/{args.env}/{args.file_name}", "w") as file:
		file.write(f"{times}\n{steps}\n{evals}")


def train_offline(RL_agent, env, eval_env, args):
	RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))

	evals = []
	steps = []
	times = []

	start_time = time.time()

	for t in range(int(args.max_timesteps + 1)):
		maybe_evaluate_and_print(RL_agent, eval_env, times, steps, evals, t, start_time, args)
		RL_agent.train()


def maybe_evaluate_and_print(RL_agent, eval_env, times, steps, evals, t, start_time, args):
	if t % args.eval_freq == 0:
		total_reward = np.zeros(args.eval_eps)

		for ep in range(args.eval_eps):
			state, done = eval_env.reset(), False

			while not done:
				action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
				state, reward, done, _ = eval_env.step(action)
				total_reward[ep] += reward

		time_total = (time.time() - start_time) / 60
		score = eval_env.get_normalized_score(total_reward.mean()) * 100 if args.offline else total_reward.mean()

		print(f"Timesteps: {(t + 1):,.1f}\tMinutes {time_total:.1f}\tRewards: {score:,.1f}")

		times.append(time_total)
		steps.append(t + 1)
		evals.append(score)

		with open(f"./results/{args.env}/{args.file_name}", "w") as file:
			file.write(f"{times}\n{steps}\n{evals}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# RL
	parser.add_argument("--policy", default="TD7", type=str)
	parser.add_argument("--env", default="HalfCheetah-v4", type=str)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--offline", default=0, type=int)
	parser.add_argument('--use_checkpoints', default=True)

	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25_000, type=int)
	parser.add_argument("--eval_freq", default=5_000, type=int)
	parser.add_argument("--eval_eps", default=10, type=int)
	parser.add_argument("--max_timesteps", default=1_000_000, type=int)
	parser.add_argument("--min_known_training_steps", default=100_000, type=int)
	parser.add_argument("--alpha_unknown", default=1., type=float)
	parser.add_argument("--redq_sample_size", default=2, type=int)

	# File
	parser.add_argument('--file_name', default=None)
	parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
	args = parser.parse_args()

	args.cmps = args.policy.split("+")
	args.cmps += ["SALE"] if "TD7" in args.cmps else []
	args.total_q_networks = 1 if "DDPG" in args.cmps else 3 if "MaxMin" in args.cmps or "REDQ" in args.cmps else 2

	if args.offline == 1:
		import d4rl
		d4rl.set_dataset_path(args.d4rl_path)
		args.use_checkpoints = False

	if args.file_name is None:
		args.file_name = f"{args.policy}_{args.seed}"

	if not os.path.exists(f"./results/{args.env}"):
		os.makedirs(f"./results/{args.env}")

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	print("---------------------------------------")
	print(f"Algorithm: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env.seed(args.seed)
	env.action_space.seed(args.seed)
	eval_env.seed(args.seed+100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	RL_agent = TD.Agent(state_dim, action_dim, max_action, args)

	if args.offline == 1:
		train_offline(RL_agent, env, eval_env, args)
	else:
		train_online(RL_agent, env, eval_env, args)