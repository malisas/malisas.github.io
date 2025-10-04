---
title: "Writing a Simple Reinforcement Learner from Scratch"
excerpt_separator: "<!--more-->"
tags:
  - Reinforcement Learning
  - Python
---

Here is a proof-of-concept Python implementation of a reinforcement learner that learns to play Tic-Tac-Toe. It is written without using any machine learning packages.

This exercise emphasizes the essential elements of reinforcement learning (RL), while naturally leading to more complex RL concepts.

<details>

<summary><b>What is a reinforcement learner?</b></summary>

A reinforcement learner is a program that "learns" while it runs. Its performance should improve over time.<br>
<br>
It is distinct from supervized learning because it does not have access to labeled data before it runs. But it is also not considered unsupervised learning; its outputs are rewarded or penalized over time, based on its performance.
</details>
<br>
**Reinforcement learning and Tic-Tac-Toe**

In the following tic-tac-toe board, supposing it's <span style="color:blue;">**R**</span>'s turn, <span style="color:blue;">**R**</span> must play its next move on the top left in order to block <span style="color:red;">**O**</span>'s win.

<!-- ![tictactoe1](/assets/images/tictactoe1.svg){:height="25%" width="25%"} -->

![tictactoe1](/assets/images/tictactoe1.gif){:height="25%" width="25%"}

<!-- {% svg /assets/images/tictactoe1.svg width=24 %} -->

An untrained reinforcement learner will randomly choose a move from the 5 empty spots on the board. A trained reinforcement learner (<span style="color:blue;">**R**</span>) should be able to correctly block <span style="color:red;">**O**</span>.  

**Training the Reinforcement Learner**

By definition, a reinforcement learner does not know what the goal of tic-tac-toe is. The only thing it can do is try different moves and see if they lead to a reward (or a penalty). If a move leads to a reward (i.e. a win), it will update the probability of making that move in future games so that it will be more likely to play the move again.

This leads to the following software considerations:
1. The reinforcement learner will need to play a lot of games.
2. The reinforcement learner needs some way to store its learnings (e.g. updated probabilities) across games. The technical terminology for this is a `policy`, which we will define here as a dictionary that maps each board state to a set of moves and their probabilities.
3. The reinforcement learner needs an opponent to play against.