import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from student_agent import Game2048Env
import os
from tqdm import tqdm
import pickle
from utils import NTupleApproximator

def select_action(env, approximator, legal_moves):
    max_value = -float("inf")
    for action in legal_moves:
        sim_env = copy.deepcopy(env)
        prev_score = sim_env.score
        next_state, score, done, _ = sim_env.step(action, spawn_tile=False)
        value = (score - prev_score) + approximator.value(next_state)
        if value > max_value:
            max_value = value
            best_action = action
            best_reward = score - prev_score
            best_next_state = next_state
    return best_next_state, best_action, best_reward

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
        done = False
        max_tile = np.max(state)
        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.

            next_state, best_action, incremental_reward = select_action(env, approximator, legal_moves)
            _, _, done, _ = env.step(best_action)
            # next_after_state = copy.deepcopy(env.after_state)
            # incremental_reward = new_score - previous_score
            # previous_score = new_score
            max_tile = max(max_tile, np.max(env.board))

            # TODO: Store trajectory or just update depending on the implementation
            # if done:
            #     next_state = None
            #     incremental_reward = 0
            trajectory.append((copy.deepcopy(next_state), incremental_reward))           



            # state = next_state

        # for t in reversed(range(len(trajectory)-1)):
        #     G = 0.0
        #     for k in range(3):
        #         if t + k < len(trajectory):
        #             s, r = trajectory[t + k]
        #             G += (gamma ** k) * r
        #         else:
        #             break
        #     if t + 3 < len(trajectory):
        #         s_next, r = trajectory[t + 3]
        #         if s_next is None:
        #             G += 0
        #         else:
        #             G += (gamma ** 3) * approximator.value(s_next)  # bootstrap from n-th step

        #     s, r = trajectory[t]
        #     # TD update
        #     delta = G - approximator.value(s)
        #     approximator.update(s, delta, alpha)
        next_s_value = 0
        for afterstate, reward in reversed(trajectory):
            delta = reward + gamma * next_s_value - approximator.value(afterstate)
            approximator.update(afterstate, delta, alpha)
            next_s_value = approximator.value(afterstate)



        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

        if (episode + 1) % 1000 == 0:
            if not os.path.exists('model/'):
                os.makedirs('model/')
            with open(f"model/value_revised_approximator_{start_eps+episode+1}.pkl", 'wb') as f:
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

    patterns = [
                [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)],
                [(0,1), (1,1), (2,1), (0,2), (1,2), (2,2)],
                [(0,0), (1,0), (2,0), (3,0), (2,1), (3,1)],
                [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2)],
                [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1)],
                [(0,1), (1,1), (2,1), (3,1), (1,2), (2,2)],
                [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2)],
                [(0,0), (1,0), (2,0), (3,0), (2,1), (2,2)],
            ]   

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
    start_eps = 20_000
    if os.path.exists(f"model/value_revised_approximator_{start_eps}.pkl"):
        print("Loading existing model...")
        with open(f"model/value_revised_approximator_{start_eps}.pkl", 'rb') as f:
            approximator = pickle.load(f)

    env = Game2048Env()
    scores = td_learning(env, approximator, start_eps, num_episodes=10_000, alpha=0.01, gamma=1.0, epsilon=0.1)
    plot_scores(scores)

if __name__ == "__main__":
    main()