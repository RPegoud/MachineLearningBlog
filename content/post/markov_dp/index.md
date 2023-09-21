---
title: "Markov Decision Processes"
description: Framing Reinforcement Learning problems efficiently
slug: markov_decision_processes
date: 2023-04-03 00:00:00+0000
image: cover.jpg
categories:
    - Reinforcement Learning
tags: 
    - Theory
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
math: true
---

`Markov Decision Processes are meant to be a straightforward framing of the problem of learning from interaction to achieve a goal.`

You might have already wondered how Reinforcement Learning (RL) agents manage to reach super-human levels in games such as chess.
Just like in chess, where a player has to ***make decisions based on the current state of the board*** and the available moves, RL agents also face similar decision-making challenges in dynamic environments.

Markov Decision Processes (MDP) **provide a mathematical framework** to **model such sequential decision-making problems**, making them a fundamental tool in Reinforcement Learning.

## ***Markov State***

To understand an RL problem, we need to define its key components. The main character in this scenario, responsible for learning and making decisions, is referred to as the **agent**. On the other side of the equation, we have the **environment**, which encompasses everything the agent interacts with.

These two, the agent and the environment, engage in an ongoing interaction. The agent takes actions, and the environment responds to these actions, presenting new situations to the agent. After each action of the agent, the environment returns a **state**, a snapshot of the environment at a given time, providing essential information for decision-making.

The totality of the state is **not always revealed to the agent**. In games like poker for example, the state could be represented as the cards held by all the players, the cards on the table and the order of cards in the deck. However, **each player only has partial knowledge of the state**. While their action modify the state of the environment, they only get an **observation** back from the environment. We call
these settings **partially observable**.

Inversely, in chess, each player can observe the state of the board fully.

Finally, the environment offers **rewards**, which are specific numerical values that the agent aims to maximize over time by making thoughtful choices in its actions.

<center><img src="RL Loop.png"></center>

In a *finite* MDP, the sets of states, actions and rewards $(\mathcal{S, A} \text{ and } \mathcal{R})$ have a finite number of elements.
In this case, the random variables $R_t$ and $S_t$ have discrete probability distributions dependent only on the preceding state and action.
We then note the probability of transitioning to a new state $s'$ while receiving a reward $r$ based on the current state $s$ and the action $a$ taken by the agent:

$$p(s',r|s,a) = Pr(S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a)$$

For every value of these random variables, $s' \in \mathcal S$ and $r \in \mathcal R$, there is a probability of those values occuring at time $t$, given particular values of the preceding state and action. The function $p$ defines the **dynamics** of the MDP. The dynamics function $p : \mathcal S ~X~ \mathcal R ~X~ \mathcal S ~X~ \mathcal A → [0,1]$ is a **probability distribution** (as indicated by the '$|$'), therefore the sum of probabilites of all states $s'$ and rewards $r$ given a state $s$ and an action $a$ equals 1. $$\sum_{s' \in \mathcal S}\sum_{r \in \mathcal R}p(s',r|s,a)=1$$ In an MDP, the probability given by $p$ **completely** characterizes the environment dynamics.

Indeed, an environment having the **Markov property** means that the probability of each possible value for $\mathcal S_t$ and $\mathcal R_t$ depends **only on the immediately preceding state and action**, $\mathcal S_{t-1}$ and $\mathcal A_{t-1}$, and not at all on earlier states and actions.

This is best viewed as a restriction not on the decision process, but on the state. The **state must include information about all aspects of the past agent–environment interactions** that make a difference for the future.
This is a crucial assumption in Reinforcement Learning as it simplifies the learning problem by allowing the agent to ignore the history of past states and actions and focus solely on the current state.

A chess game is a good example of an MDP. We can define the environment's state as the position of all the pieces on the board. In this setting, the **game history** (sequence of moves played to get to the current position) **is not useful to predict the best possible move**. Therefore the environment is said to be Markov.

<center><img src="Markov State.png"></center>

## ***Reward and reward hypothesis:***

At each timestep, the agent receives a ***reward*** $R_t \in \mathbb{R}$ defining its purpose. The goal of the agent is to maximize the total amount of rewards i.e. the cumulative reward. This idea is illustrated by the ***reward hypothesis***:

> ***"All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward)"***

To complete the chess example, we could define the reward as being **positive** when the agent **wins** the game and **negative** when it **loses** or **draws**. One could argue that capturing pieces should generate positive rewards, however it is possible to lose or draw the game after capturing almost all the opponent's pieces.

It is important to understand that rewards are used to **set the goal to achieve and not the way to achieve it** which is for the agent to figure out.

## ***Return and episodes***

Now that we have defined the notion of reward, we are interested in maximizing this reward over time. For this we need to define a new variable, the **return**.

The return $G_t$ is the **sum of future rewards** at timestep $t$.
$$G_t = R_{t+1}+R_{t+2}+R_{t+3}+...=\sum_{k=0}^\infty R_{t+k+1}$$
Importantly, the return is a **random variable**, as the dynamics of the MDP can be **stochastic** (i.e. involve randomness). In other words, the same actions can lead to different rewards if the environment dynamics are random.
Therefore we want to consider and maximize the **expected return**, the expected sum of future rewards:

$$\mathbb{E}[G_t] = \mathbb{E}[\sum_{k=0}^\infty R_{t+k+1}]$$

For this definition to make sense, the sum of rewards has to be **finite**. What happens if we want the agent to perform a task continuously, i.e. if the number of timesteps is infinite ?

We must distinguish two cases:

* **Episodic MDPs**: the MDP can be naturally decomposed in **episodes**, a finite sequence of actions that end on a **terminal state**. Each  episode starts in the **same configuration** and is **independent from previous episodes**.
  * A game of chess is again a good example, **each game starts in the same setting regardless of previous games**. A game ends by a draw or by checkmate (terminal states). When the game ends we can reset the board and start anew.
* **Continuous MDPs**: here the number of timesteps in **infinite**, the MDP goes on continually
  * An example of continuous MDP could be controlling a dam to optimize the energy production depending on the predicted demand. The dam being permanently active, there is no terminal state (if we leave out special events such as maintenance).

## ***Continuing tasks and discounting rewards***

One might wonder, how can the expected return be finite when the MDP goes on forever ?

For continuous MDPs, the return is defined as the **discounted** sum of future rewards. The discount factor $\gamma \in [0,1)$ makes sure that ***rewards far in the future receive a lower weight***:
 $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

This acknowledges that rewards obtained in the future are generally considered less valuable than immediate rewards. This concept is crucial when dealing with tasks that have uncertain, long-term consequences.

 A high discount factor values immediate rewards more, while a low factor assigns similar importance to rewards regardless of when they are received.

 <center><img src="Discounted Reward.png"></center>

We can prove that the discounted return is finite by defining $R_{max}$ as the highest reward the agent can receive, therefore:
$$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1} \leq \sum_{k=0}^\infty \gamma^k R_{max}$$
As $R_{max}$ is constant, we can write:
$$\sum_{k=0}^\infty \gamma^k R_{max} = R_{max} \sum_{k=0}^\infty \gamma^k$$
As $\sum_{k=0}^\infty \gamma^k$ is a geometric series converging for $\gamma \lt 1$ we can conclude that:
$$R_{max} \sum_{k=0}^\infty \gamma^k = R_{max} \times { {1}\over{1-\gamma} } $$
In conclusion, $G_t$ has a finite upper bound and is therefore finite:
$$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1} \leq R_{max} \times { {1}\over{1-\gamma} }$$

## ***Wrapping up***

There we have it ! In this article, we learned about ***MDPs***, ***rewards*** and how to define the ***return for episodic and continuing tasks***. In the next article, we'll see how we can ***solve*** an MDP to find the best possible action in each state by introducing the notions of ***policy***, ***value functions*** and last but not least, the <a href="../bellman_equation/">Bellman Equation<a/>
