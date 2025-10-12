---
title: "Writing a Simple Reinforcement Learner from Scratch"
excerpt_separator: "<!--more-->"
tags:
  - Reinforcement Learning
  - Python
---
<br>
# Introduction

This post guides the reader through a proof-of-concept Python implementation of a reinforcement learner that learns to play Tic-Tac-Toe. It is intended for people new to reinforcement learning.

## What is a Reinforcement Learner?

A reinforcement learner is a program that "learns" while it runs, using trial-and-error. Let's contrast this to how a human behaves:

**How a typical human behaves**

When a human plays tic-tac-toe, they enter the game with an understanding of the objective of the game (aim to get 3-in-a-row) and a general strategy. When a human sees this board:

![tictactoe5](/assets/images/tictactoe5.svg){:height="15%" width="15%"}  

they will recognize that <span style="color:blue;">**X**</span> has to block <span style="color:red;">**O**</span> on the next turn by playing the bottom-left, or <span style="color:blue;">**X**</span> will lose.

**How a Reinforcement Learner learns**

When an untrained reinforcement learner is shown the same board, it does not know the objective of tic-tac-toe. The only thing it can do is place an <span style="color:blue;">**X**</span> in one of the 6 empty spaces on the board. It has no knowledge about whether one move is better than another, so it will randomly choose a move.

Suppose it randomly chooses the following space as its next move:  
![tictactoe6](/assets/images/tictactoe6.gif){:height="15%" width="15%"}  
<span style="color:red;">**O**</span> then wins on the next move:  
![tictactoe7](/assets/images/tictactoe7.gif){:height="15%" width="15%"} 

This will trigger a "game over" state, and the reinforcement learner loses. When this happens, the reinforcement learner will update its future behavior to account for the fact that it lost to <span style="color:red;">**O**</span>. After this game (or "episode"), it will be less likely to make the same move in the future. In RL terminology, the agent receives a negative reward for its behavior. This is how it "learns". 

**RL Terminology:** Episodic tasks in reinforcement learning consist of a sequence of steps that end in a terminal state, after which the agent is rewarded for its performance. 
{: .notice--info}

To recap, a reinforcement learner does not know the goal of tic-tac-toe. The only thing it can do is try different moves and see if they lead to a win (a reward). After a move leads to a reward, it will update the probability of making that move in future games so that it will be more likely to play the move again.

# Let's Train a Reinforcement Learner

Let's look at how such a reinforcement learner would perform over time. This section will provide context for and motivate the [coding section]() later in the post.

## Performance improves over time

Let's have the reinforcement learner play 100,000 games against a player called `OptimalPlayer` which always plays optimal moves. Every 1,000 games, we'll calculate what proportion of the last 1,000 games the reinforcement learner won, and plot the performance over time.  
![tictactoe10](/assets/images/tictactoe10.svg)  
The performance starts out pretty bad, but after 100,000 games, the score is around 0.4.  
This is not bad, considering that if two optimal players play against each other, every game will end in a tie:  
![tictactoe12](/assets/images/tictactoe11.svg)  

Let's see if we can get that score even higher by training even longer! Say, 1,500,000 games.  
After 1,500,000 games of tic-tac-toe, the average score is very close to 0.5:  
![tictactoe15](/assets/images/tictactoe15.svg)  

That's pretty good! 

## Testing the reinforcement learner

To test the reinforcement learner's performance after 1,500,000 games, let's have the reinforcement learner compete in tournaments against `OptimalPlayer`, `DumbPlayer`, and `MischeviousPlayer`. `DumbPlayer` just plays random moves on every turn, and `MischeviousPlayer` is mischevious!

Let's have the trained reinforcement learner play 20,000 games against each player (note that the reinforcement learner does not learn from games played during tournaments) and plot the percent of games the reinforcement learner won, tied, or lost against each opponent:

---------

Almost all of the games against `OptimalPlayer` end in a tie, as expected. But the tournament against `MischeviousPlayer` has a lot of losses, and it even loses against `DumbPlayer` _% of the time. What's going on?

During training, the reinforcement learner played games against the **Optimal** Player. This optimal player always plays optimal moves which pursue the most win conditions while blocking the opponent's win. This means the reinforcement learner does not have any data about how to respond to suboptimal moves.

Here's one game in which the reinforcement learner lost against `MischeviousPlayer`. `MischeviousPlayer` intentionally plays suboptimal moves. In this game, you can see move _ is suboptimal:

------------

This is a very common pitfall in reinforcement learning; the training process does not expose the reinforcement learner to a wide enough variety of situations, leading to poor performance later on.

How can we train the reinforcement learner so that it can perform better against more opponents?  
1. We can have the reinforcement learner train against different opponents.
2. We can also have the reinforcement learner do more intentional "exploring" (next section)...

## Exploration vs Exploitation

The **reinforcement learner**, by its own design, can get stuck playing the same moves over and over again.  
If the reinforcement learner plays <span style="color:blue;">5</span> on its first turn and it leads to a win, it will be more likely to play <span style="color:blue;">5</span> in the second game, which, if it again leads to a win, will further reinforce the probability of playing space <span style="color:blue;">5</span>...in fact it might never practice playing space <span style="color:blue;">1</span>.  
It's like walking over the same set of footprints in a snowy field over and over again.  
![snowyfootsteps](/assets/images/snowy-footsteps.jpg){:height="30%" width="30%"}  
This behavior is "exploitative", meaning that the reinforcement learner will always be more likely to choose a move that historically led to a higher score. We can instead have it train "exploratively" by increasing its probability of choosing different moves.

**RL Terminology:** The [exploration-exploitation tradeoff](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma) is fundamental in reinforcement learning. Exploration involves seeking new, uncertain opportunities to discover potential future rewards, while exploitation involves leveraging existing knowledge and resources to secure immediate, known benefits.
{: .notice--info}

To accomplish a higher level of exploration, we provide an epsilon parameter in the reinforcement learner which controls the rate at which it explores. We'll keep the epsilon parameter high during training to encourage exploration.

## Training and Testing Round 2

Let's have the reinforcement learner continue training, this time against `DumbPlayer`, using the epsilon parameter to accomplish greater exploration.    
After 1,500,000 training games, let's test the reinforcement learner again:  

-------------

Not perfect, but much better!

# Code

This section talks about the technical details of coding the reinforcement learner and the training and testing functions. The code is written without using machine learning packages.

In order to code and train a reinforcement learner, there are a few software considerations:
1. The reinforcement learner needs to train by playing a lot of games.
2. The reinforcement learner needs some way to store its learnings (e.g. updated probabilities) across games. The technical terminology for this is a `policy`, which we will define here as a dictionary that maps each board state to a set of possible moves and their evolving probabilities/weights.
3. The reinforcement learner also needs an opponent.

**RL Terminology:** A policy is the strategy an agent uses to decide actions. A policy maps states to possible actions and their probabilities.
{: .notice--info}

Taking these requirements at face value, we can write a program that fulfills the basic definition of a reinforcement learner.  

## Representing the board

There needs to be a way to represent the tic-tac-toe board using code.  
One approach is to consider the board as a grid with spots numbered 1 through 9:  
![tictactoe3](/assets/images/tictactoe3.svg){:height="15%" width="15%"}  
Now suppose we want to represent the following game state:  
![tictactoe2](/assets/images/tictactoe2.svg){:height="15%" width="15%"}  
Let's say <span style="color:blue;">**X**</span> played first in space <span style="color:blue;">`7`</span>, followed by <span style="color:red;">**O**</span> in <span style="color:red;">`5`</span>, and so on, like this:  
![tictactoe8](/assets/images/tictactoe8.gif){:height="20%" width="20%"}  

**Representation 1 (move order does matter):**  

We can use the following list representation of the board:  
```
board = [[7,6],[5,9]]
```
Here it is taken to mean that player 1 (<span style="color:blue;">**X**</span>) played <span style="color:blue;">`7`</span> on their first move, followed by <span style="color:blue;">`6`</span> on their next turn.  
We will use this representation to track the in-game state of the board. We use a list because it's easy to append to.

**Representation 2 (when move order does not matter):**  

Sometimes we want to ignore move order, in which case we'll use the following string representation of the board:  
```
board_str = '6759'
``` 
By "ignoring move order", it means that the following two games:
```
[[7,6],[5,9]]
[[6,7],[5,9]]
```
are represented by the single string above.  
We will use this representation as the key to our policy dictionary (discussed later). We purposely ignore the move order in the policy dictionary because the historical order of the moves does not affect what the "best" move will be on the player's next move. The list has been converted to a string because a list cannot be a dictionary key.  

Now if you see `board` and `board_str` in the code, you'll know what they are referring to.  

## The Reinforcement Learner

<!-- <details markdown="1"><summary>Click to expand code</summary> -->
Now let's write the reinforcement learner.

The `ReinforcementLearner` class has two main functions: `move` and `update_policy`.  
- The `move` function accepts the current `board` and looks up its entry in `self.policy`. It will then choose randomly from the moves available for the current board, weighted by the values in `self.policy`.  
An example of a possible `policy` entry for the following board is 
```
{'6759': {1: 0.525,
          2: 0.475,
          3: 0.15,
          4: 0.45,
          8: 0.425}}
```
![tictactoe2](/assets/images/tictactoe2.svg){:height="15%" width="15%"}  
Here we see the dictionary entry for `'6759'` contains another dictionary in which a weight is assigned for each available move (1, 2, 3, 4, and 8).
- The `update_policy` function is run after each game of tic-tac-toe. If the reinforcement learner wins a game, it will be "rewarded" by positively incrementing the weight of the moves it made (in `self.policy`).   
Note that not all moves are rewarded equally. For the following board where <span style="color:blue;">**X**</span> won (represented by `[[6,7,1,4],[5,9,2]]`), the winning move <span style="color:blue;">`4`</span> is awarded a full 0.1 points, the second-to-last move is awarded 0.05 points, all the way down to 0.0125 points for the first move (<span style="color:blue;">`6`</span>).  
![tictactoe4](/assets/images/tictactoe4.svg){:height="15%" width="15%"}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Replay:
![tictactoe9](/assets/images/tictactoe9.gif){:height="20%" width="20%"}  
- By the way, the policy entry does not actually contain probabilities. It's just the sum of all the rewards that have been applied. In fact it's possible for the policy entry to contain negative values:
```
{'6759': {1: 4.17,
          2: -5.37,
          3: -8.64,
          4: -3.95,
          8: -5.52}}
```
This doesn't translate directly to appropriate probabilities for choosing each move. So we will convert these values to actual probabilities using the `softmax` function, which is commonly used in RL. Here you can see how the policy values turn into appropriate probabilities after applying the softmax function:

| Move | Policy Value | Softmax |
|---:|---:|---:|
| 1 | 4.17 | 0.999 |
| 2 | -5.37 | 7.11e-05 |
| 3 | -8.64 | 2.69e-06 |
| 4 | -3.95 | 0.0002 |
| 8 | -5.52 | 6.12e-05 |

**RL Terminology:** The [softmax function](https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning) can be used to convert values into action probabilities.
{: .notice--info}

Putting it all together, the `ReinforcementLearner` class looks like this:  
([link]() to the corresponding code in a Jupyter Notebook, slightly modified for additional functionality)

{% highlight python linenos %}
class ReinforcementLearner(Player):
    def __init__(self, name=None):
        super().__init__(name)
        self.policy = {}
    def move(self, board):
            board_str = board_to_str(board)
            self.initialize_policy_entry_if_missing(board_str, board)
            player_idx = len(board_str) % 2
            possible_moves = list(self.policy[board_str].keys())
            # Weights must be non-negative and can't all be 0
            weights = softmax([self.policy[board_str][key] for key in possible_moves])
            # The player chooses randomly from the available moves, weighted
            # by the softmax of the associated policy entry
            next_move = random.choices(possible_moves,
                           weights=weights,
                           k=1)[0]
            # The player appends its next move to the board
            board[player_idx].append(next_move) 
    def update_policy(self, game_status, board, player_idx):
        if game_status != "Tie":
            # If the player lost, penalize each move the player took.
            # If the player won, reward each move the player took.
            reward_sign = -1 if ((game_status == "Player 1 Win") & (player_idx == 1)) | ((game_status == "Player 2 Win") & (player_idx == 0)) else 1
            # How many moves did the player take before winning/losing?
            num_moves = len(board[player_idx])
            for i in range(num_moves):
                board_before_ith_move = [board[0][0:i+1],board[1][0:i]] if player_idx == 1 else [board[0][0:i],board[1][0:i]]
                ith_move = board[player_idx][i]
                policy_key = board_to_str(board_before_ith_move)
                # Apply a weaker reward/penalty for earlier moves
                reward_multiplier = (0.5**(num_moves-1-i))*0.1
                current_reward = reward_sign*reward_multiplier
                self.initialize_policy_entry_if_missing(policy_key, board_before_ith_move)
                self.policy[policy_key][ith_move] += current_reward
    def initialize_policy_entry_if_missing(self, policy_key, current_board):
        if not policy_key in self.policy:
            possible_moves = set(get_available_moves(current_board))
            self.policy[policy_key] = dict.fromkeys(possible_moves, 0.5)

class Player:
    def __init__(self, name=None):
        self.name = name

def board_to_str(board):
    # [[3,5,2],[9,4]] would become '23549'
    # Each player's turns are individually sorted, and then concatenated
    return "".join([str(num) for num in sorted(board[0]) + sorted(board[1])])

def softmax(weights):
    weights_softmax = [math.e**(x) for x in weights]
    divisor = sum(weights_softmax)
    weights_softmax = [x / divisor for x in weights_softmax]
    return weights_softmax

{% endhighlight %}


<!-- </details> -->

## The Opponent

In addition to the reinforcement learner, we will provide three opponents:  
1. `OptimalPlayer` will always choose an optimal move.
2. `DumbPlayer` will always choose a random move.
3. `MischeviousPlayer` is mischevious! 😈

Link to the code for these players: [link]()

## Gameplay

Finally, we have a couple functions that take care of game play: `play_game()` plays a single game, and `train_rl()` trains a reinforcement learner against an opponent for a specified number of games.

We also have a `tournament()` function which we will use to test how well our reinforcement learner performs after being trained.

Link to this code: [link]()

# Review and Conclusion

In this post we saw how it is possible to write a basic reinforcement learner from scratch in Python. We saw how a reinforcement learner's performance improves over time, and we also observed that it is important to train a reinforcement learner carefully so that it gains experience across a wide range of situations.

RL Terminology Covered:  
- Episodic tasks
- Policy
- Softmax
- Exploration-Exploitation Tradeoff

Reinforcement learning is a vast topic and an active research area. Famous reinforcement learners include AlphaGo and even ChatGPT. Hopefully this post can begin to demystify some of the technology behind such software. Thanks for reading!