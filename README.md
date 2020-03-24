#### Overview

An automated system to respond to a question based on its similarity to one (with an associated answer) in the corpus.

A project for MIT 6.864 - Advanced Natural Language Processing, Fall 2017 done in collaboration with [Raoul Khouri](https://github.com/Keyrat06)

* Information Retrieval - <https://en.wikipedia.org/wiki/Information_retrieval>
* Question Answering - <https://en.wikipedia.org/wiki/Question_answering>

In general, the task of Question Answering - i.e, providing an answer to a question posed (for example on a site like [Stack Exchange](https://stackexchange.com/) is hard to automate. A system for this must parse the natural language query to determine what is being asked, then query its knowledge base for a suitable answer (Information Retrieval) and then construct a response (Natural Language Generation).

In this project we build some of the pieces of an automated system that can handle this task of Question Answering to explore the methods involved.


#### Task - Question Retrieval

We are given a set of questions Q and a training set of similar question pairs {(q1; q11, q12, ...), (q2; q21, q22, ...), ...}. Given a new quest q, we have to return all similar question from Q.

The model will be trained on the [AskUbuntu](https://github.com/taolei87/askubuntu) dataset. The algorithm is based on the paper [Semi-supervised Question Retrieval with Gated Convolutions](https://arxiv.org/pdf/1512.05726.pdf)


#### Task - Transfer Learning

Here we work on the [Android](https://github.com/jiangfeng1124/Android) dataset.

We first apply our model trained on the AskUbuntu dataset on the Android dataset, i.e, direct transfer without domain adaptation. We then apply domain adapatation using techniques explained in [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1409.7495.pdf) and [Aspect-augmented Adversarial Networks for Domain Adaptation](https://arxiv.org/pdf/1701.00188.pdf).
