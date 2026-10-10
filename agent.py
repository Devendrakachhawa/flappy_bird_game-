import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.use_lidar = params["use_lidar"]
        self.eval_every = params["eval_every"]
        self.eval_episodes = params["eval_episodes"]

        self.loss_fn = nn.SmoothL1Loss()  # huber - MSE turns big TD errors into huge gradients
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):

        # use_lidar=False -> the 12-feature obs. lidar mode gives 180 raw rays AND
        # adds a -0.5/step "private zone" penalty in the env, which drowns out the
        # +0.1 alive / +1.0 pipe rewards.
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=self.use_lidar,
        )

        num_states = env.observation_space.shape[0] # input dim
        num_actions = env.action_space.n # output dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt & bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_eval = float("-inf")

            # separate env for greedy evaluation
            eval_env = gym.make("FlappyBird-v0", use_lidar=self.use_lidar)

            # resume if a checkpoint exists, so a restart can't overwrite a better model
            if os.path.isfile(self.MODEL_FILE):
                ckpt = torch.load(self.MODEL_FILE, map_location=device)
                policy_dqn.load_state_dict(ckpt["model"])
                target_dqn.load_state_dict(ckpt["model"])
                best_eval = ckpt.get("best_eval", float("-inf"))
                epsilon = ckpt.get("epsilon", self.epsilon_init)
                print(f"resumed {self.MODEL_FILE}: best_eval={best_eval} epsilon={epsilon}", flush=True)

        else:
            # best policy load
            ckpt = torch.load(self.MODEL_FILE, map_location=device)
            policy_dqn.load_state_dict(ckpt["model"])
            policy_dqn.eval()
            epsilon = 0  # no exploration when evaluating

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                next_state, reward, terminated, _, _ = env.step(action.item())
                
                episode_reward += reward

                # create tensors
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state

                # one gradient step per env step (was once per episode)
                if is_training and len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # sync the network
                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

            print(f"episode={episode+1} with total reward={episode_reward} & epsilon={epsilon}")

            if is_training:
                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # checkpoint on a greedy eval, not on one training episode. with epsilon
                # exploration the random flaps score well by themselves, so training
                # reward keeps climbing even while the policy itself collapses.
                if (episode + 1) % self.eval_every == 0:
                    eval_reward, eval_pipes = self.evaluate(policy_dqn, eval_env)

                    log_msg = (
                        f"eval at episode={episode+1}: mean_reward={eval_reward:.2f} "
                        f"mean_pipes={eval_pipes:.2f} best={best_eval:.2f}"
                    )
                    print(log_msg, flush=True)
                    with open(self.LOG_FILE, "a") as f:
                        print(log_msg, file=f)

                    if eval_reward > best_eval:
                        best_eval = eval_reward

                        with open(self.LOG_FILE, "a") as f:
                            print(f"SAVED new best eval={best_eval:.2f} pipes={eval_pipes:.2f} at episode={episode+1}", file=f)

                        torch.save(
                            {
                                "model": policy_dqn.state_dict(),
                                "best_eval": best_eval,
                                "eval_pipes": eval_pipes,
                                "epsilon": epsilon,
                            },
                            self.MODEL_FILE,
                        )

            # env.close() - manually stop

    
    def evaluate(self, policy_dqn, env):
        """greedy rollouts - the honest measure of the policy alone, no exploration"""
        policy_dqn.eval()

        total_reward = 0.0
        total_pipes = 0

        for i in range(self.eval_episodes):
            state, _ = env.reset(seed=1000 + i)  # fixed seeds -> comparable across evals
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0
            info = {}

            while (not terminated and episode_reward < self.reward_threshold):
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward
                state = torch.tensor(next_state, dtype=torch.float, device=device)

            total_reward += episode_reward
            total_pipes += info.get("score", 0)

        policy_dqn.train()
        return total_reward / self.eval_episodes, total_pipes / self.eval_episodes


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

            
        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_dqn.parameters(), 10.0)
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--render', help='Show the game window', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, render=args.render)
    else:
        dql.run(is_training=False, render=True)
