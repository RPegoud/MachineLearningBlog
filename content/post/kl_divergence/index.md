---
title: "KL divergence"
description: Measuring distances between probability distribution
slug: kl_divergence
date: 2023-05-01 00:00:00+0000
image: cover.jpg
categories:
    - Statistics
tags: 
    - Theory
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
math: true
---

***Credits: [Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM), [Wikipedia](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)***

In recent years, reinforcement learning (RL) has emerged as a powerful paradigm for solving complex decision-making problems. One of the fundamental challenges in RL is to accurately estimate the value of different actions in a given state.
KL divergence, ***a measure of how one probability distribution differs from another***, has become an essential tool for addressing this challenge. KL divergence is widely used in machine learning, particularly in deep learning and probabilistic modeling, to compare the difference between two probability distributions.
In the context of RL, ***KL divergence plays a crucial role in many algorithms, such as policy optimization and value function approximation***. In this article, we will explore the importance of KL divergence in RL and machine learning in general and discuss its applications in solving a wide range of problems.

## ***Definition***

The ***Kullback-Leibler divergence*** is a statistical distance measuring ***how different a probability distribution is from a reference distribution.***
We usually consider two distributions:

* $P$ which serves as reference, it represents the data or observations
* $Q$ which represents a model or an approximation of $P$

The KL divergence is defined for both ***continuous*** and ***discrete probability distributions***.

### ***Discrete case***

For $P$ and $Q$ defined on the same sample space $\mathcal{X}$, the KL divergence from $Q$ to $P$ is defined as:
$$D_{KL}(P||Q) = \sum_{x \in \mathcal{X}}P(x)log({P(x)\over{Q(x)}})$$
One way to interpret this formula is to consider it as the ***expectation of the logarithmic difference between the probabilities $P$ and $Q$***, where the expectation is taken using $P$ as reference.

### ***Continuous case***

For distributions $P$ and $Q$ of a continuous random variable, the KL divergence from $Q$ to $P$ is defined as the integral:
$$D_{KL}(P||Q) = \int_{-\infty}^{\infty}P(x)log({P(x)\over{Q(x)}})dx$$

## ***Intuitive explanation***

As mentioned previously, the KL divergence is intended to measure the difference between probability distributions. In other words it measures how likely it is for the distribution $Q$ to generate samples from distribution $P$.

Let's take the simplest example of a Binomial distribution, the coin toss:
Consider two coins, the first one is a fair coin, therefore $P(heads) = P(tails) = 0.5$

However the second coin is biased, $P(heads)=p$ and $P(tails)=1-p$

How can we measure the distance between these two distributions ? Certainly if $p$ is close to 0.55, the distributions would be much more similar than if $p=0.95$.
Indeed if $p=0.55$, then it would be easy to ***confuse*** the two distributions. We could measure this by ***comparing the probability of a specific sequence under both distributions***.

Let's say we toss the first coin 10 times and obtain the following sequence:
$$H,T,T,H,T,H,H,H,T,H$$
Comparing the likelihood of this sequence happening for the fair coin and the biased coin could boil down to computing:
$${P(sequence | \text{fair coin})\over P(sequence | \text{biased coin})}$$
To compute the likelihood of this sequence happening for both coins, lets define:

* $p_1 = P(head|\text{fair coin})$ and  $p_2 = P(tails|\text{fair coin})$
* $q_1 = P(head|\text{biased coin})$ and  $q_2 = P(tails|\text{biased coin})$

Intuitively, the probability of observing the previous sequence would be:
$$P(sequence|\text{fair coin}) =  p_1\times p_2\times p_2\times p_1\times p_2\times p_1\times p_1\times p_1\times p_2\times p_1$$
A more elegant way to write this expression is obtained by raising both probabilities to the power $N_H$ and $N_T$ where $N_H$ is the number of heads and $N_T$ the number of tails.
$$P(sequence|\text{fair coin}) = p_1^{N_H} \text{ } p_2^{N_T}$$
$$P(sequence|\text{biased coin}) = q_1^{N_H} \text{ } q_2^{N_T}$$

Therefore, the ratio defined previously becomes:
$${P(sequence | \text{fair coin})\over P(sequence | \text{biased coin})} = {p_1^{N_H} \text{ } p_2^{N_T}\over q_1^{N_H} \text{ } q_2^{N_T}}$$
Believe it or not, the KL divergence is just around the corner ! Let's normalize for sample size by raising the ratio to the power of $1/N$ and then take the log of the expression, we obtain:
$$log({p_1^{N_H} \text{ } p_2^{N_T}\over q_1^{N_H} \text{ } q_2^{N_T}})^{1\over N}$$
Using the log properties, we obtain the following equivalences:
$${1\over N}log({p_1^{N_H} \text{ } p_2^{N_T}\over q_1^{N_H} \text{ } q_2^{N_T}})$$ By breaking down multiplications and divisions:
$${1\over N}log\text{ }p_1^{N_H} + {1\over N}log\text{ }p_2^{N_T} - {1\over N}log\text{ }q_1^{N_H} - {1\over N}log\text{ }q_2^{N_T}$$
We can once again drop down the powers:
$${N_H\over N}log\text{ }p_1+ {{N_T}\over N}log\text{ }p_2 - {{N_H}\over N}log\text{ }q_1 - {{N_T}\over N}log\text{ }q_2$$

Now, if the observations are generated by the fair coin (which we use as reference), then, as $N$ tends to infinity, the proportion of observed heads becomes $p_1$ and the proportion of observed tails becomes $p_2$.

Therefore, in the limit we can say that ${N_H\over N} = p_1$ and ${N_T\over N} = p_2$

From there, we can simplify the equation and finally get to the discrete definition of the KL divergence:
$$p_1\text{ }log\text{ }p_1+ p_2\text{ }log\text{ }p_2 - p_1\text{ }log\text{ }q_1 - p_2\text{ }log\text{ }q_2$$
$$= p_1\text{ }log\text{ }{p_1\over q_1} + p_2\text{ }log{p_2 \over q_2}$$
$$= \sum p(x)log\text{ }{p(x)\over q(x)}$$
Please note that this proof also holds for more than two classes.

## ***To remember***

* While the KL divergence is a distance it is ***not a metric*** for the following reasons:
  * It is not symmetric
  * It doesn't satisfy the triangle inequality

 Indeed the KL divergence is ***not symmetric***, meaning that the KL divergence between two probability distributions P and Q is not necessarily equal to the KL divergence between Q and P. This is because the KL divergence measures the difference between two probability distributions in terms of ***how much information is lost when using Q to approximate P***, and ***the amount of information lost depends on which distribution is used as the reference***.

 In other words, the KL divergence measures the extent to which one probability distribution differs from another, and this difference may be asymmetric depending on the specific context and the reference distribution used. For example, in some cases, one distribution may be a much better approximation of another than vice versa, resulting in different KL divergences when comparing the two distributions.

* The KL Loss is ***equivalent to the Cross-Entropy Loss*** since both aim to minize distances between distributions
