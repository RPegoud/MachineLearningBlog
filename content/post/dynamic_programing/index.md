---
title: "Dynamic programming"
description: Solving environments with known dynamics pig-pong style
slug: dynamic_programming
date: 2023-05-09 00:00:00+0000
image: cover.jpg
categories:
    - Reinforcement Learning
tags: 
    - Theory
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
math: true
---

*<center>Because GPI is some sort of ping-pong 🏓</center>*

In the previous article, we discussed the [[notes/theory/Bellman Equation|Bellman equation]], a key concept in RL that enables us to calculate the optimal value function of an agent in a Markov Decision Process. The Bellman equation laid the foundation for the development of dynamic programming methods in RL.

Dynamic Programming (DP) is a ***class of algorithms that utilizes the Bellman equation to solve problems*** in a systematic and efficient manner. Dynamic Programming methods are especially useful for solving large and complex decision-making problems.

In this article, we will dive deeper into the world of Dynamic Programming methods for RL. We will explore various algorithms such as ***Value Iteration*** and ***Policy Iteration***, and see how they can be used to solve larger and more complex decision-making problems. We will discuss the strengths and weaknesses of each algorithm and when to use them depending on the problem we are trying to solve.

> An implementation of the principles described in this article is available here: [[notes/implementations/Policy Iteration on Frozen Lake| Policy Iteration on Frozen Lake]]

## ***1) Introduction***

Dynamic programming methods are a collection of algorithms that can ***compute an optimal policy*** $\pi_\star$ ***given a perfect model of an environment as an MDP***. In other words, dynamic programming is particularly useful if we know $p$, the environment dynamics.
However most of the recent RL methods try to replicate the results obtained by dynamic programming in a more efficient manner (requiring less computation) and without a perfect model of the environment (which is almost never available in practice).

The core idea of DP methods is to ***use value functions to structure the search for good policies***, the best illustration of this idea is ***Generalized Policy Iteration***.

Generalized Policy Iteration (GPI) is a reinforcement learning algorithm that ***iteratively improves a policy and the corresponding value function in an MDP***. The algorithm consists of two main steps: ***policy evaluation*** and ***policy improvement***.

GPI combines these two steps in an iterative process until the optimal policy and value function are found. The algorithm continues to ***alternate between policy evaluation and improvement until the policy converges to the optimal policy***. This process is guaranteed to converge to an optimal policy after a large number of iterations for a ***finite MDP***.

## ***2) Policy Evaluation***

Policy evaluation (also called prediction) is the task of determining the state value function of a given policy:
$$\pi \rightarrow v_\pi$$
In theory, if the environment's dynamics are known, the Bellman equation provides a system of $|\mathcal S|$ (the total number of states) linear equations in $|\mathcal S|$ unknowns ($v_\pi(s), s \in \mathcal S$), this system can be solved to find the optimal value function.

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P[s',r|s,a](r + \gamma v_\pi(s'))$$

$$\rightarrow \text{Linear solver}$$
$$\rightarrow v_\star = \max_a \sum_{s',r}p[s',r|s,a](r +\gamma v_\star(s'))$$

In practice, we prefer iterative approches. Consider a sequence of approximations of the value function $v_0, v_1, v_2, ...$ mapping $\mathcal S^+$ (all states that are non terminal) to $\mathbb{R}$ (real numbers). *
We can compute the optimal policy by ***repeatedly applying the Bellman equation as an update rule***:

* $v_0$  is chosen arbitrarily, $v$(terminal) = 0
* $v_{k+1} = \mathop{\mathbb{E}}[R_{t+1} + \gamma v_k(S_{t+1}) | S_t=s]$
 $\qquad = \sum_a \pi(a|s) \sum_{s',r}p[s',r|s,a](r + \gamma v_k(s'))$

The sequence {$v_k$} is shown to converge to $v_k$ = $v_\pi$ as $k \rightarrow \infty$.
For each successive approximation, the state values will be replaced with new values computed from the current ones and the expected immediate reward from all the one-step transitions possible under policy $\pi$.

To understand how to implement the policy improvement algorithm in python, head over the

## ***3) Policy Improvement***

Using policy evaluation, we have now determined $v_\pi$ for an arbitrary deterministic policy $\pi$. Now we want to improve this policy (ultimately this is the goal of computing the value function). ***Improving*** a policy means ***finding states for which we can pick actions yielding a higher expected reward than the current policy's choices***. In other words, we know $v_\pi(s)$ or "How good is it to follow the current policy from state $s$" and ***want to determine whether it would be better or not to change the policy to deterministically choose an action $a \neq \pi(s)$.***

Remember that in state $s$ choosing action $a$ has a value of:
$$q_\pi(s,a) = \sum_{s',r}p[s',r|s,a](r + \gamma v_\pi(s'))$$
The important question here is whether $q_\pi(s,a)$ is superior to $v_\pi(s)$. If yes, then it is better to pick action $a$ and then follow $\pi$ than to always follow $\pi$. In this case, why not pick $a$ every time we are in state $s$ ? Indeed, this would lead to a better policy, this result is a general result of the *policy improvement theorem*.

## ***3.1) Theorem***

Consider two deterministic policies $\pi$ and $\pi'$ such that:
$$q_\pi(s, \pi'(s)) \geq v_\pi(s) \quad\forall s\in\mathcal S$$
In other words, the action value of $\pi'$ for state $s$ is superior the state value of $\pi$ for state $s$, for any state. This implies that the $\pi'$ must be equal or better than $\pi$, therefore $\pi'$ must obtain greater or equal expected return from all states:
$$v_\pi'(s) \geq v_\pi(s)$$
Now let's go back to the policies we were considering in the previous section, $\pi$ and $\pi'$ that are identical except for an action $a$ where $\pi'(s) = a \neq \pi(s)$.
In this case if the specific action we changed is better than the one picked by the current policy, i.e. $q_{\pi}(s,a) > v_\pi(s)$ then the policy $\pi'$ is superior.

One explanation goes as follows, let's assume:

$$v_\pi(s) \leq q_{\pi}(s, \pi'(s))$$
We can rewrite the right side as the value of state $s$ from which we follow $\pi'$: $$ = \mathbb E [R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s, A_t = \pi'(s)]$$
Adding $\pi'$ in the expectation, we get:  $$ = \mathbb E_{\pi'} [R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s]$$
Using the previous relation $v_\pi(s) \leq q_\pi(s, \pi'(s))$, we can write:
$$ = \mathbb E_{\pi'} [R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s]\leq  \mathbb E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1}))|S_t=s]$$
From there, we can keep repeating the previous steps:
$$\mathbb E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1}))|S_t=s]$$
$$= \mathbb E_{\pi'} [R_{t+1} + \gamma \mathbb E_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) | S_{t+1}, A_{t+1}=\pi'(S_{t+1})] | S_t=s]$$
$$= \mathbb E_{\pi'} [R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2})|S_t=s]$$
$$\leq \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + \gamma^3v_\pi(S_{t+3})|S_t=s]$$
And so on until we reach a terminal state:
$$\leq \mathbb E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + \gamma^3R_{t+4} + ... | S_t=s]$$
$$= v_{\pi'}(s)$$
So far, we've seen how to evaluate a change in the policy for a single state-action pair. We can also extend this to all states and all actions, selecting the best action according to $q_\pi(s,a)$.
This is synonymous with being greedy with respect to $q_\pi$, which leads to the new policy:
$$\pi'(s) = arg\max_a q_\pi(s,a)$$
$$= arg\max_a \mathbb E[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t =s, A_t=a]$$
$$= arg\max_a \sum_{s',r} p[s',r |s,a](r + \gamma v_\pi(s'))$$
Here $arg\max$ refers to the action for which the following expression is maximized. An important detail is that if several action maximize this expression, the chosen action is picked at random among them.

This new policy meets the condition of the policy improvement theorem, therefore it is as good as, or better than the original policy.

>***The process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy, is called policy improvement.***

## ***4) Policy Iteration***

*Coming soon*
