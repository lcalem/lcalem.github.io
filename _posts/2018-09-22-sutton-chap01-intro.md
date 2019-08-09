---
layout: post
title:  "Sutton & Barto summary chap 01 - Introduction"
date:   2018-09-22 02:23:34 +0200
categories: rl sutton
math: true
---

This is the first post of the Sutton & Barto summary [series][ref-series].

* TOC
{:toc}

## 1.1. Elements of Reinforcement learning

What RL is about:
- trial and error search, delayed reward
- exploration vs exploitation tradeoff

Main elements : agent and environment

<div class="img-block" style="width: 400px;">
    <img src="/imgs/sutton/rl_basics.png"/>
    <span><strong>Fig 2.1.</strong> Agent and Environment interactions</span>
</div>

Subelements:
- __policy__ $\pi$, mapping from states to actions. May be stochastic
- __reward__ signal: given by the environment at each time step
- __value__ function of a state $V(s)$: total amount of reward an agent can expect to accumulate over the future, starting from that state
- [optional] a __model__ of the environment: allows inferences to be made about how the environment will behave

## 1.2. Limitations and scope

- state signal design is overlooked
- focus on value function estimation (no evolutionary methods)

## 1.3. Example

Temporal-difference learning method illustrated with the game of tic-tac-toe:

We want to build a player who will find imperfections in its opponent's play and learn to maximize its chances of winning.

- We will consider every possible configuration of the board as a __state__
- For each state keep a __value__ representing our estimate of our probability of winning from this state. We initialize each value in some way (perhaps 1 for all the winning states, 0 for all losing states and 0.5 otherwise)
- Everytime we have to play, we examine the states that would result from our possible moves, and pick the action leading to the state with the highest value, most of the time. That is, we play _greedily_. Sometimes we may _not_ play the greedy action and pick a random one instead. We would then make an __exploratory__ move.
- While we are playing, we want to make our values more accurate. The current value of the earlier state $V(S_t)$ is updated to be closer to the value of the later state $V(S_{t+1})$:

$$V(S_t) \gets V(S_t) + \alpha [ V(S_{t+1}) - V(S_t) ]$$

where $\alpha$ is the __step-size paramter__ controlling the rate of learning. This update rule is an example of __temporal-difference learning__, which name comes from the fact that its changes are based on the difference $V(S_{t+1}) - V(S_t)$ between estimates at two different times.

## 1.4. History of reinforcement learning

Major topics in chronological order

- At the beginning there was studies about : trial and error / animal learning / law of effect
- __optimal control__: designing a controller to minimize a measure of a dynamical system’s behavior over time - value functions and dynamic programming
- __temporal difference (TD) methods__: driven by the difference between temporally successive estimates of the same quantity

Something interesting
- RL + game theory : Szita 2012


[ref-series]: /blog/2018/09/22/sutton-index
