---
title: "Writing a Simple Reinforcement Learner from Scratch"
excerpt_separator: "<!--more-->"
tags:
  - Reinforcement Learning
  - Python
---

# Overview

Here is a proof-of-concept Python implementation of a reinforcement learner that learns to play Tic-Tac-Toe. It is written without using any machine learning packages.

I personally found this exercise helpful because it emphasizes the essential elements of reinforcement learning (RL), while naturally leading to more complex RL concepts. By using only basic Python packages, it forces me to understand what's happening at every step and not take any shortcuts.

<details>

<summary><b>What is a reinforcement learner?</b></summary>

A reinforcement learner is a program that "learns" while it runs. Its performance should improve over time.<br>
<br>
It is distinct from supervized learning because it does not have access to labeled data before it runs. But it is also not considered unsupervised learning; its outputs are rewarded or penalized over time, based on its performance.
</details>

## Reinforcement learning and Tic-Tac-Toe

In the following tic-tac-toe board, supposing it's <span style="color:blue;">**R**</span>'s turn, <span style="color:blue;">**R**</span> must play its next move in the top left in order to block <span style="color:red;">**O**</span>'s win.

![tictactoe1](/assets/images/tictactoe1.gif){:height="25%" width="25%"}

An untrained reinforcement learner would randomly choose a move from the 5 empty spots on the board. A trained reinforcement learner (<span style="color:blue;">**R**</span>) should be able to correctly block <span style="color:red;">**O**</span>.  

## Training a Reinforcement Learner

By definition, a reinforcement learner does not know the goal of tic-tac-toe. The only thing it can do is try different moves and see if they lead to a reward (a win). If a move leads to a reward, it will update the probability of making that move in future games so that it will be more likely to play the move again.

This leads to the following software considerations:
1. The reinforcement learner will need to play a lot of games in order to train.
2. The reinforcement learner needs some way to store its learnings (e.g. updated probabilities) across games. The technical terminology for this is a `policy`, which we will define here as a dictionary that maps each board state to a set of moves and their evolving probabilities.
3. The reinforcement learner needs an opponent to play against.

Taking these requirements at face value, it's quite simple to write a program that fulfills the basic definition of a reinforcement learner.

# Code

## The reinforcement learner

<!-- <details markdown="1"><summary>Click to expand code</summary> -->

{% highlight python linenos %}
class ReinforcementLearner(Player):
    def __init__(self, name=None):
        super().__init__(name)
        self.policy = {}
    def move(self, board_state):
            board_state_std = get_board_state_std(board_state)
            self.initialize_policy_entry_if_missing(board_state_std, board_state)
            player_idx = len(board_state_std) % 2
            possible_moves = list(self.policy[board_state_std].keys())
            # Weights must be non-negative and can't all be 0
            weights = softmax([self.policy[board_state_std][key] for key in possible_moves])
            next_move = random.choices(possible_moves,
                           weights=weights,
                           k=1)[0]
            board_state[player_idx].append(next_move) 
    def update_policy(self, game_state, board_state, player_idx):
        if game_state != "Tie":
            # If the player lost, penalize each move the player took.
            # If the player won, reward each move the player took.
            reward_sign = -1 if ((game_state == "Player 1 Win") & (player_idx == 1)) | ((game_state == "Player 2 Win") & (player_idx == 0)) else 1
            # How many moves did the player take before winning/losing?
            num_moves = len(board_state[player_idx])
            for i in range(num_moves):
                board_state_before_ith_move = [board_state[0][0:i+1],board_state[1][0:i]] if player_idx == 1 else [board_state[0][0:i],board_state[1][0:i]]
                ith_move = board_state[player_idx][i]
                policy_key = get_board_state_std(board_state_before_ith_move)
                # Apply a weaker reward/penalty for earlier moves
                reward_multiplier = (0.5**(num_moves-1-i))*0.1
                current_reward = reward_sign*reward_multiplier
                self.initialize_policy_entry_if_missing(policy_key, board_state_before_ith_move)
                self.policy[policy_key][ith_move] += current_reward
    def initialize_policy_entry_if_missing(self, policy_key, current_board_state):
        if not policy_key in self.policy:
            possible_moves = set(get_available_moves(current_board_state))
            self.policy[policy_key] = dict.fromkeys(possible_moves, 0.5)

# Turn board_state into a simpler "standardized" value, an integer that can be used as a dictionary key
def get_board_state_std(board_state):
    # [[3,5,2],[9,4]] would become '23549'
    # Each player's turns are sorted and then concatenated
    return "".join([str(num) for num in sorted(board_state[0]) + sorted(board_state[1])])

def softmax(weights):
    weights_softmax = [math.e**(x/temp) for x in weights]
    divisor = sum(weights_softmax)
    weights_softmax = [x / divisor for x in weight  s_softmax]
    return weights_softmax

class Player:
    def __init__(self, name=None):
        self.name = name
{% endhighlight %}

<!-- </details> -->