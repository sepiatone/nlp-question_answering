#### Overview

Question Answering is a subfield in Natural Language Processing dealing with automated systems to respond to a question from the user.

A project for MIT 6.864 - Advanced Natural Language Processing, Fall 2017 done in collaboration with [Raoul Khouri](https://github.com/Keyrat06)

* Information Retrieval - <https://en.wikipedia.org/wiki/Information_retrieval>
* Question Answering - <https://en.wikipedia.org/wiki/Question_answering>

In general, the task of Question Answering - i.e, providing an answer to a question posed (for example on a site like [Stack Exchange](https://stackexchange.com/) is hard to automate. A system for this must parse the natural language query to determine what is being asked, then query its knowledge base for a suitable answer (Information Retrieval) and then construct a response (Natural Language Generation).

In this project we build some of the pieces of an automated system that can handle this task of Question Answering to explore the methods involved.


#### Task - Question Retrieval

We are given a set of questions Q and a training set of similar question pairs {(q1; q11, q12, ...), (q2; q21, q22, ...), ...}. Given a new question q, we have to return all similar question from Q.

The model will be trained on the [AskUbuntu](https://github.com/taolei87/askubuntu) dataset. The algorithm is based on the paper [Semi-supervised Question Retrieval with Gated Convolutions](https://arxiv.org/pdf/1512.05726.pdf)


#### Task - Transfer Learning

Here we work on the [Android](https://github.com/jiangfeng1124/Android) dataset.

We first apply our model trained on the AskUbuntu dataset on the Android dataset, i.e, direct transfer without domain adaptation. We then apply domain adapatation using techniques explained in [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1409.7495.pdf) and [Aspect-augmented Adversarial Networks for Domain Adaptation](https://arxiv.org/pdf/1701.00188.pdf).


Baseline metrics are obatained by using in-house implementations of the  [BM25](http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf) and [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) algorithms.

[Framework](https://github.com/taolei87) from [Tao Lei](https://people.csail.mit.edu/taolei/)
