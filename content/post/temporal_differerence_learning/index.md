---
title: Comparing Temporal Difference Learning algorithms
description: Welcome to Hugo Theme Stack
slug: td_learning_comparison
date: 2022-03-06 00:00:00+0000
image: cover.jpg
categories:
    - Reinforcement Learning
tags:
    - Example Tag
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

*The full code for this experiment is available in the following [GitHub repo](https://github.com/RPegoud/Temporal-Difference-learning/tree/main)*

| Algorithm  | Type        | Runtime (400 episodes, 10 runs) | Discovered second optimal strategy |     | 
| ---------- | ----------- | ---------------------- | ------------------------ | --- |
| Q-learning | Model-free  | 4 min 03 sec           | No                       |     |
| Dyna-Q     | Model-based | 59 min 03 sec          | No                       |     |
| Dyna-Q+    | Model-based | 48 min 12 sec          | Yes                      |     |

## ***Introduction***

***Coming soon***

## ***The environment***

<img src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Environment.svg?raw=true">

The environment we'll use in this experiment is a grid world with the following features:
* The grid is 12 by 8 cells, meaning there are 96 states in total
* The agent starts in the bottom left corner of the grid
* The objective is to reach the treasure located in the top right corner
* There are different kind of portals:
	* The blue portals are connected, going through the portal located on the cell (10, 6) leads to the cell  (11, 0). The agent cannot take the portal again after its first transition.
	* The purple portal only appears after 100 episodes but allows to reach the treasure faster
	* The red portal are traps (terminal states) and end the episode
	* The agent starts the episode surrounded by walls, bumping into one of them will result in the agent remaining in the same state

<img src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Movements.jpg?raw=true">

The aim of this experiment is to compare the behavior of the Q-learning, Dyna-Q and Dyna-Q+ agents in a changing environment. Indeed, after 100 episodes, the optimal policy is bound to change and the optimal number of steps during a successful episode will decrease from 17 to 12.

<img src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/GridWorld.svg?raw=true">

## ***Q-learning:***

<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/q_learning_state_values.gif?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/q_learning_101.png?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/q_learning_251.png?raw=true">
<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/images/q_learning_performance_report.png?raw=true">

## ***Dyna-Q:***

<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_state_values.gif?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/dyna_q_101.png?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/dyna_q_251.png?raw=true">

<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/images/dyna_q_performance_report.png?raw=true">

## ***Dyna-Q+***


<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/videos/dyna_q_plus_state_values.gif?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/dyna_q_plus_101.png?raw=true">
<src="https://github.com/RPegoud/ML_Blog/blob/hugo/content/notes/images/Portal%20gridworld%20TD%20learning/dyna_q_plus_251.png?raw=true">

<img src="https://github.com/RPegoud/Temporal-Difference-learning/blob/main/images/dyna_q_plus_performance_report.png?raw=true">
