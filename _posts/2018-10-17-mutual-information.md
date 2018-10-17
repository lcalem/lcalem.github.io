---
layout: post
title:  "Mutual Information"
date:   2018-10-17 14:00:00 +0200
categories: information_theory
math: true
---

* TOC
{:toc}



## Introduction

While reading [this](https://openreview.net/forum?id=HyEtjoCqFX) paper lately, I stumbled on the notion of __mutual information__ from information theory, so as always I took a trip on the internet and summarized my findings in a markdown file. Here they are:

We can see mutual information as a __reduction in uncertainty__ for predicting parts of the outcome of a system of two random variables.
Given a system with two random variables $X$ and $Y$, when we observe the outcome of $X$, there is a corresponding reduction in uncertainty for $Y$. Mutual information measures this reduction.

## The formula

$$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)} \tag{1}\label{eq:1}$$

In plain English:

The __mutual information__ $I(X, Y)$ bewteen two random variables $X$ and $Y$ is equal to the __sum__ over all possible values $x$ of $X$ and $y$ of $Y$ of the __joint probability__ $p(x, y)$ of the two events $x$ and $y$, times the $\log$ of the __joint probability__ divided by the product of the __marginal probabilities__ $p(x)$ and $p(y)$.

Note: This is the formula for discrete processes (for continuous, sums become integrals)

Let's quickly explain those two notions:
- The __marginal probability__ $P(x)$ gives the probability of an event occurring. It is not conditioned on another event. It is more formally denoted $P_{X}(x)$.
- The __joint probability__ $P(x, y)$ gives the probability of event $X = x$ __and__ event $Y = y$ occurring. It is the probability of the intersection of the two elements.

The __distribution__ part is when you consider these probabilities over all possible values of the random variables, that is the _distribution_ over the probabilities. The __marginal distribution__ becomes $P(X)$ and the __joint distribution__ becomes $P(X, Y)$.

When the <strong style="color: #1E72E7">joint distribution</strong> of $X$ and $Y$ is known, the <strong style="color: #ED412D">marginal distribution</strong> of $X$ is the probability distribution of $X$ averaging over information about $Y$. To calculate $P(X)$, we sum the joint probabilities over Y.

<div class="table-wrap">
    <table class="prob-table">
        <tr>
            <td></td>
            <td></td>
            <td colspan="3"><strong>Y</strong></td>
            <td><strong>P(Y)</strong></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>y<sub>1</sub></td>
            <td>y<sub>2</sub></td>
            <td>y<sub>3</sub></td>
            <td></td>
        </tr>
        <tr>
            <td rowspan="3"><strong>X</strong></td>
            <td>x<sub>1</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>2</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>1</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>2</sup>&frasl;<sub>16</sub></td>
            <td style="color: #ED412D;"><sup>5</sup>&frasl;<sub>16</sub></td>
        </tr>
        <tr>
            <td>x<sub>2</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>1</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>1</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>4</sup>&frasl;<sub>16</sub></td>
            <td style="color: #ED412D;"><sup>6</sup>&frasl;<sub>16</sub></td>
        </tr>
        <tr>
            <td>x<sub>3</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>1</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>3</sup>&frasl;<sub>16</sub></td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8"><sup>1</sup>&frasl;<sub>16</sub></td>
            <td style="color: #ED412D;"><sup>5</sup>&frasl;<sub>16</sub></td>
        </tr>
        <tr>
            <td><strong>P(X)</strong></td>
            <td></td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8"><sup>4</sup>&frasl;<sub>16</sub></td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8"><sup>5</sup>&frasl;<sub>16</sub></td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8"><sup>7</sup>&frasl;<sub>16</sub></td>
            <td></td>
        </tr>
    </table>
</div>

## What does it mean?

To grasp the meaning of the mutual information, we will consider the two extremes. First the case where our two random variables are completely independent, and then the case where they are completely dependent.

### Independent random variables

$X$ and $Y$ have two values, $0$ or $1$, each with probability $\frac{1}{2}$.

<div class="table-wrap">
    <table class="prob-table">
        <tr>
            <td></td>
            <td></td>
            <td colspan="2"><strong>Y</strong></td>
            <td><strong>P(Y)</strong></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>0</td>
            <td>1</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>X</strong></td>
            <td>0</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.25</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.25</td>
            <td style="color: #ED412D;">0.5</td>
        </tr>
        <tr>
            <td>1</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.25</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.25</td>
            <td style="color: #ED412D;">0.5</td>
        </tr>
        <tr>
            <td><strong>P(X)</strong></td>
            <td></td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8">0.5</td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8">0.5</td>
            <td></td>
        </tr>
    </table>
</div>

The <strong style="color: #1E72E7">blue</strong> values are the __joint probabilities__ $P(x, y)$.
The <strong style="color: #ED412D">red</strong> values are the __marginal probabilities__ $P(x)$, $P(y)$.

Here we see that $P(x, y) = 0.25$ for all possible values of $x$ and $y$, and that the marginal probability is always $0.5$. This matches the definition of independence where the joint probability is equal to the product of the marginal probabilities:

$$P(x, y) = P(x) P(y)$$

With this observation, calculating the mututal information is fairly easy, we can change the numerator $P(x, y)$ in the mutual information formula \ref{eq:1} by $P(x) P(y)$:

$$
\begin{align}
I(X; Y) & = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)} \tag{1}\\
 & = \sum_{x, y} P(x, y) \log \frac{P(x) P(y)}{P(x) P(y)}\\
 & = \sum_{x, y} P(x, y) \log 1\\
 & = 0
\end{align}
$$

The values in the log cancel out and we're left with $\log(1) = 0$, that is, the mutual information of two independent variables is zero. Since the two variables are independent, the mutual information, that is the reduction in uncertainty for the second variable given the first variable is observed, is zero. If the two variables are independent, knowing the value of $X$ doesn't reduce the uncertainty of $Y$ at all.

Let's consider the opposite case where we have very strong dependence between the two variables.


### Dependent random variables

Here, if we know the outcome of $X$, we also perfectly know the outcome of $Y$.

<div class="table-wrap">
    <table class="prob-table">
        <tr>
            <td></td>
            <td></td>
            <td colspan="2"><strong>Y</strong></td>
            <td><strong>P(Y)</strong></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td>0</td>
            <td>1</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>X</strong></td>
            <td>0</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.5</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.0</td>
            <td style="color: #ED412D;">0.5</td>
        </tr>
        <tr>
            <td>1</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.0</td>
            <td style="color: #1E72E7; border-top: 1px solid #e8e8e8">0.5</td>
            <td style="color: #ED412D;">0.5</td>
        </tr>
        <tr>
            <td><strong>P(X)</strong></td>
            <td></td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8">0.5</td>
            <td style="color: #ED412D; border-top: 1px solid #e8e8e8">0.5</td>
            <td></td>
        </tr>
    </table>
</div>

We should expect that the reduction in uncertainty in one of the variables to be equal to $1$, because one value perfectly determines the other.

If we go ahead with the formula, we need to sum over all the possible values of $x$ and $y$.
We should only consider the diagonal values because in the anti-diagonal the joint probability $P(x, y)$ is zero and so the corresponding term is zero.

For the diagonal terms where $P(x, y) = 0.5$ we have:

$$\begin{equation} \label{eq2}
\begin{split}
& P(x, y) \log \frac{P(x, y)}{P(x) P(y)}\\
 & = 0.5 \log \frac{0.5}{0.5^2}\\
 & = 0.5 \log \frac{1}{0.5}\\
 & = 0.5 \log 2\\
 & = 0.5
\end{split}
\end{equation}$$

Since we're using bits of information we have a $\log_2$ and $\log_2(2) = 1$.
Out of the four terms we should consider in the sum, two of them are zero and two of them are $0.5$, so the sum of all the terms is $1$, as expected.

### Relation to the Kullback-Leibler divergence

From the two previous examples, essentially the mutual information is a way of capturing the degree of dependence between two variables.
If the two variables are strongly dependent, there is a high degree of mutual information and it means that we know a lot more by knowing the joint distribution than by knowing the marginal distribution.

In each of these examples, the marginal distributions are equal, even if the two systems are very different.
- For the __independent__ variables case the marginal distributions gives us the joint distribution, since the joint is the product of the marginals in the independent case. So the joint distribution is a bit redundant and gives no extra information.
- For the __dependent__ variables case the marginal distribution is not really useful, we know a lot more about the dynamics of the system by looking at the joint distribution.

So we kind of have a parallel here:
- independent variables - no mutual information - we get our information from the __marginal__ distribution
- dependent variables - mutual information is 1 - we get our information from the __joint__ distribution

This gives the intuition that mutual information has something to do with the divergence between the joint and mutual distributions. About the informational cost of representing our system as the product of marginals opposed to the joint distribution. More precisely, we can express the mutual information using the Kullback-Leibler divergence from the joint distribution $P(x, y)$, to the product of the marginal distributions $P(x) \cdot P(y)$.

$$I(X; Y) = D_{KL}\big( p(x, y) \; || \; p(x) p(y) \big) \tag{2}\label{eq:2}$$

Where $D_{KL}$ is the Kullback-Leibler divergence, defined between two distributions $p$ and $q$ by:

$$D_{KL}(p || q) \doteq \sum_x p(x) \log \frac{p(x)}{q(x)} \tag{3}\label{eq:3}$$

Equation \ref{eq:1} expresses the mutual information using the divergence between from the joint distribution to the product of the marginal distributions. It can also be useful to express the mutual information in terms of the divergence computed on one random variable only (instead of the two variables in equation 1). To do this, we use __conditional distributions__:

$$
\begin{align}
P(x | y) & = \frac{P(x, y)}{P(y)} \label{eq:4}\tag{4}\\
P(x, y) & = P(x | y) \; P(y) \label{eq:5}\tag{5}
\end{align}
$$

Substituting equation \ref{eq:5} in equation \ref{eq:1}, we get:

$$
\begin{align}
I(X; Y) & = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)} \label{eq:6}\tag{6}\\
 & = \sum_x \sum_y P(x|y) P(y) \log \frac{P(x|y) P(y)}{P(x) P(y)} \label{eq:7}\tag{7}
\end{align}
$$

We simplyfy and split the expressions in their respective sums (expressions that depend on $y$ in the $y$ sum and same for $x$), and we get a KL divergence (see eq. \ref{eq:3}) that is the divergence of the __univariate__ distribution $P(x)$ from the conditional distribution $P(x \|y)$:

$$
\begin{align}
 & = \sum_y P(y) \sum_x P(x|y) \log \frac{P(x|y)}{P(x)} \label{eq:8}\tag{8}\\
 & = \sum_y P(y) D_{KL} \big( p(x|y) \; || \; p(x) \big) \label{eq:9}\tag{9}
\end{align}
$$

With this view, the mutual information can be seen as the expectation of the divergence between the conditional distribution $P(x \|y)$ and $P(x)$. The more different these two distributions are on average, the greater the information gain. It makes sense because if $P(x \|y)$ and $P(x)$ are the same, it means the information about $y$ is useless and the mutual information is indeed zero. On the other end of the problem, if $P(x \|y)$ and $P(x)$ are very different, it means information about $y$ is very important to know about $x$, and the mutual information is high.

Alternatively (as in the paper I was reading), we can write the same equation summing on the other variable using $P(x, y) = P(y \| x) \; P(x)$:

$$I(X; Y) = \sum_x P(x) D_{KL} \big( p(y|x) \; || \; p(y) \big) \label{eq:10}\tag{10}$$


## Wrapping up

The mutual information between random variables $X$ and $Y$ expresses the reduction in uncertainty in one variable when we have information about the other.

$$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)} \tag{1}$$

This formula involves the __joint distribution__ and the __marginal distribution__, and we've seen that the mutual information quantity can be seen as an axis from 0 to 1:

- $I(X, Y) = 0$, no mutual information, the random variables are __independent__ and all the information is contained in the __marginal distributions__
- $I(X, Y) = 1$, the random variables are totally __dependent__ and the information is contained in the __joint distribution__

This view leads to the Kullback-Leibler divergence view of the mutual information:

$$I(X; Y) = D_{KL}\big( p(x, y) \; || \; p(x) p(y) \big) \tag{2}$$

Using conditional probabilities, we can express the mutual information with a divergence using only one of the random variables, which may be a handy formulation:

$$I(X; Y) = \sum_y P(y) D_{KL} \big( p(x|y) \; || \; p(x) \big) \tag{9}$$