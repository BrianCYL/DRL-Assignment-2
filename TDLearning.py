import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from student_agent import Game2048Env
import os
from tqdm import tqdm
import pickle
from utils import NTupleApproximator

def select_action(env, approximator, legal_moves, prev_score):
    max_value = -float("inf")
    for action in legal_moves:
        sim_env = copy.deepcopy(env)
        next_state, score, done, _, after_state = sim_env.step(action)
        value = (score - prev_score) + approximator.value(after_state)
        if value > max_value:
            max_value = value
            best_action = action
    return best_action

def td_learning(env, approximator, start_eps, num_episodes=50000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            best_action = select_action(env, approximator, legal_moves, previous_score)
            state_copy = copy.deepcopy(state)
            next_state, new_score, done, _, after_state = env.step(best_action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            if done:
                after_state = None
                incremental_reward = 0
            trajectory.append((copy.deepcopy(state_copy), best_action, incremental_reward, copy.deepcopy(after_state)))           

            state = after_state

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        # for s, a, r, ns in trajectory:
        #     if not ns is None:
        #         delta = r + gamma * approximator.value(ns) - approximator.value(s)
        #         approximator.update(s, delta, alpha)
        #     else:
        #         delta = - approximator.value(s)
        #         approximator.update(s, delta, alpha)

        for t in range(len(trajectory)):
            G = 0.0
            for k in range(3):
                if t + k < len(trajectory):
                    _, _, r, _ = trajectory[t + k]
                    G += (gamma ** k) * r
                else:
                    break
            if t + 3 < len(trajectory):
                s_next, _, _, _ = trajectory[t + 3]
                if s_next is None:
                    G += 0
                else:
                    G += (gamma ** 3) * approximator.value(s_next)  # bootstrap from n-th step

            s, _, _, _ = trajectory[t]
            # TD update
            delta = G - approximator.value(s)
            approximator.update(s, delta, alpha)


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    if not os.path.exists('model/'):
        os.makedirs('model/')
    with open(f"model/value_4tuple_approximator_{start_eps+num_episodes}.pkl", 'wb') as f:
        pickle.dump(approximator, f)

    return final_scores

def plot_scores(scores):
    weights = np.ones(100) / 100
    sma = np.convolve(scores, weights, mode='valid')
    plt.plot(sma)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.show()

def main():
    # patterns = [[(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
    #             [(1,0), (1,1), (1,2), (1,3), (2,0), (2,1)],
    #             [(2,0), (2,1), (2,2), (2,3), (3,0), (3,1)],
    #             [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
    #             [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]] 
    # patterns = [[(1,0), (2,0), (3,0), (1,1), (2,1), (3,1)],
    #             [(1,1), (2,1), (3,1), (1,2), (2,2), (3,2)],
    #             [(0,0), (1,0), (2,0), (3,0), (2,1), (3,1)],
    #             [(1,0), (1,1), (1,2), (1,3), (2,2), (3,2)]]

    # patterns = [[(0, 0), (0, 1), (0, 2), (0, 3)],
    #             [(1, 0), (1, 1), (1, 2), (1, 3)],
    #             [(2, 0), (2, 1), (2, 2), (2, 3)],
    #             [(3, 0), (3, 1), (3, 2), (3, 3)],

    #             # vertical 4-tuples
    #             [(0, 0), (1, 0), (2, 0), (3, 0)],
    #             [(0, 1), (1, 1), (2, 1), (3, 1)],
    #             [(0, 2), (1, 2), (2, 2), (3, 2)],
    #             [(0, 3), (1, 3), (2, 3), (3, 3)],

    #             # all 4-tile squares
    #             [(0, 0), (0, 1), (1, 0), (1, 1)],
    #             [(1, 0), (1, 1), (2, 0), (2, 1)],
    #             [(2, 0), (2, 1), (3, 0), (3, 1)],
    #             [(0, 1), (0, 2), (1, 1), (1, 2)],
    #             [(1, 1), (1, 2), (2, 1), (2, 2)],
    #             [(2, 1), (2, 2), (3, 1), (3, 2)],
    #             [(0, 2), (0, 3), (1, 2), (1, 3)],
    #             [(1, 2), (1, 3), (2, 2), (2, 3)],
    #             [(2, 2), (2, 3), (3, 2), (3, 3)],
    #             ]

    approximator = NTupleApproximator(4, patterns)
    start_eps = 10_000
    if os.path.exists(f"model/value_4tuple_approximator_{start_eps}.pkl"):
        print("Loading existing model...")
        with open(f"model/value_4tuple_approximator_{start_eps}.pkl", 'rb') as f:
            approximator = pickle.load(f)

    env = Game2048Env()
    scores = td_learning(env, approximator, start_eps, num_episodes=10_000, alpha=0.1, gamma=0.99, epsilon=0.1)
    plot_scores(scores)

if __name__ == "__main__":
    main()