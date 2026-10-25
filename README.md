# SARSA on CliffWalking (Reinforcement Learning)

A from-scratch implementation of the **SARSA** algorithm that teaches an agent to cross the Cliff Walking grid using [Gymnasium](https://gymnasium.farama.org/). The agent learns to reach the goal while avoiding the cliff.

## The Problem

The environment is a 4 x 12 grid world (`CliffWalking-v1`):

- **Start:** bottom-left cell
- **Goal:** bottom-right cell
- **Cliff:** the cells between start and goal in the bottom row
- **Rewards:** -1 for every step, -100 for stepping onto the cliff (the agent is sent back to the start)
- **Actions:** up, right, down, left
- **States:** 48 (one per grid cell)

## How SARSA Works

SARSA is an **on-policy** temporal-difference algorithm. After every step, it updates its estimate of how good an action is using the action it will actually take next:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
```

The agent chooses actions with an **epsilon-greedy** policy: it explores a random action with probability `epsilon`, and otherwise picks the action with the highest Q value.

## Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.5 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 0.1 | Exploration rate |
| `episodes` | 500 | Training episodes |

## Results

After training, the agent's total reward per episode settles at roughly **-17 to -20**. A reward of -17 matches the safe path: up 3 rows, right 11 columns, down 3 rows. Because SARSA accounts for its own random exploration, it prefers this safer route over walking right next to the cliff edge.

## Run It Yourself

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/sarsa-cliff-walking.git
   cd sarsa-cliff-walking
   ```
2. Install the dependencies:
   ```bash
   pip install "gymnasium[toy-text]" numpy matplotlib jupyter
   ```
3. Open the notebook and run all cells:
   ```bash
   jupyter notebook sarsa_cliff_walking.ipynb
   ```

The last cell renders the trained agent walking to the goal in a game window.

## Project Files

| File | Description |
|---|---|
| `sarsa_cliff_walking.ipynb` | Training and evaluation code |

## What I Learned

- How the exploration vs. exploitation trade-off works with epsilon-greedy
- How the SARSA update rule uses the next action that is actually taken
- Why on-policy learning (SARSA) leads to a safer path than off-policy learning
- How to debug an RL loop, such as forgetting to update the state after each step

## Next Steps

- Implement **Q-learning** on the same environment and compare the paths
- Try `is_slippery=True` to see how a stochastic environment changes the result
- Add epsilon decay

## Author

**Dnyaneshwari** | GitHub:
