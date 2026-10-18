# Preface

Just a few years ago, deep learning scientists were not building products at
every major company and startup. When we entered the field, machine learning
didn't make headlines, and our parents had no idea what it was. It was a
blue-skies academic discipline with a narrow set of industrial applications,
such as speech recognition and computer vision, and those often needed so much
domain knowledge that machine learning was just one small component.
Neural networks---the predecessors of the deep learning methods in this
book---were generally regarded as outmoded.

Then, in just a few years, deep learning took the world by surprise. It drove
rapid progress in computer vision, natural language processing, automatic
speech recognition, reinforcement learning, and biomedical informatics, and
even catalyzed new developments in theoretical machine learning and
statistics. We can now build cars that drive themselves with more autonomy
than ever before (though less than some companies would have you believe),
dialogue systems that debug code by asking clarifying questions, and agents
that beat the world's best players at board games such as Go, a feat once
thought to be decades away. These tools are changing how movies are made and
diseases are diagnosed, and play a growing role in the sciences, from
astrophysics to climate modeling, weather prediction, and biomedicine.


## About This Book

This book is our attempt to make deep learning approachable, teaching you the
*concepts*, the *context*, and the *code*.

### One Medium Combining Code, Math, and HTML

For any computing technology to reach its full impact, it must be well
understood, well documented, and supported by mature tools. Its key ideas
should be clearly distilled so newcomers get up to speed quickly, libraries
should automate common tasks, and example code should be easy to modify and
extend. Dynamic web applications are a good example: companies like Amazon
built database-driven web apps in the 1990s, but the technology only reached
creative entrepreneurs at scale in the past ten years, thanks in part to
powerful, well-documented frameworks.

Deep learning is harder to make approachable because every application brings
together several disciplines. You must simultaneously understand
(i) the motivation for casting a problem in a particular way;
(ii) the mathematical form of a given model;
(iii) the optimization algorithms for fitting models to data;
(iv) the statistical principles that tell us when to expect models to
generalize to unseen data, and practical methods to certify that they have;
and (v) the engineering techniques for training models efficiently, avoiding
the pitfalls of numerical computing and making the most of the hardware.
Our goal is a single resource that teaches the critical thinking to formulate
problems, the mathematics to solve them, and the software to implement the
solutions.

When we started, no resource was simultaneously (i) up to date; (ii) broad and
technically deep enough to cover modern machine learning practice; and
(iii) as well written as a textbook while including the clean, runnable code
of a hands-on tutorial. Code examples for particular frameworks or models
(e.g., LeNet, AlexNet, ResNet) were scattered across blog posts and GitHub
repositories, but they showed *how* to implement an approach without
explaining *why* the algorithmic decisions were made. Interactive resources
such as the blog [Distill](http://distill.pub) covered only selected topics
and often lacked code. Textbooks such as :citet:`Goodfellow.Bengio.Courville.2016`,
a comprehensive survey of the basics of deep learning, did not connect the
concepts to code, sometimes leaving readers clueless about how to implement
them. And too many resources were behind the paywalls of commercial courses.

So we set out to create a resource that would (i) be freely available to
everyone; (ii) offer enough technical depth to start you on the path to
becoming an applied machine learning scientist; (iii) include runnable code
showing *how* to solve problems in practice; and (iv) allow rapid updates, by
us and by the community.

These goals often conflict. Equations, theorems, and citations are best laid
out in LaTeX, code is best written in Python, and webpages are native to HTML
and JavaScript. We also wanted the content to be available as executable code,
a physical book, a PDF, and a website. No existing workflow handled all of
this, so we built our own (:numref:`sec_how_to_contribute`): GitHub to share
the source and take community contributions, Jupyter notebooks to mix code,
equations, and text, and Sphinx as the rendering engine. The system isn't
perfect, but it strikes a reasonable compromise, and *Dive into Deep Learning*
might be the first book published with such an integrated workflow.


### Learning by Doing

Many textbooks cover each concept in exhaustive detail before moving on. The
excellent textbook of :citet:`Bishop.2006`, for example, teaches each topic so
thoroughly that reaching the chapter on linear regression takes nontrivial
work. Experts love it for exactly that reason, but it limits its usefulness
for true beginners.

In this book, we teach most concepts *just in time*: you learn each concept at
the moment you need it to accomplish something practical. We spend some time
at the outset on fundamentals like linear algebra and probability, but we want
you to taste the satisfaction of training your first model before worrying
about more esoteric concepts.

Apart from a few preliminary notebooks that give a crash course in the
mathematical background, each chapter introduces a reasonable number of new
concepts along with several self-contained working examples on real datasets.
We follow a policy of *one working example, one notebook*, which makes it as
easy as possible to start your own projects from our code: just copy a
notebook and start modifying it.

We interleave runnable code with background material as needed, and we often
make tools available before explaining them fully, filling in the background
later. For instance, we might use *stochastic gradient descent* before
explaining why it works. This gives you the tools to solve problems quickly,
at the cost of trusting some of our curatorial decisions.

This book teaches deep learning from scratch. Sometimes we dig into details
that modern frameworks normally hide, especially in the basic tutorials, where
we want you to understand everything that happens inside a given layer or
optimizer. In those cases we often present two versions of an example: one
implemented from scratch, using only NumPy-like functionality and automatic
differentiation, and a practical one using the frameworks' high-level APIs.
Once we have explained how a component works, later tutorials use the
high-level API.


### Content and Structure

The book has roughly three parts: preliminaries, deep learning techniques, and
advanced topics focused on real systems and applications
(:numref:`fig_book_org`).

![Book structure.](../img/book-org.svg)
:label:`fig_book_org`


* **Part 1: Basics and Preliminaries**.
:numref:`chap_introduction` introduces deep learning.
:numref:`chap_preliminaries` quickly covers the prerequisites for hands-on
deep learning: storing and manipulating data, and numerical operations based
on elementary linear algebra, calculus, and probability.
:numref:`chap_regression` and :numref:`chap_perceptrons` cover the most
fundamental concepts and techniques: regression and classification, linear
models, multilayer perceptrons, and overfitting and regularization.

* **Part 2: Modern Deep Learning Techniques**.
:numref:`chap_computation` describes the key computational components of deep
learning systems and lays the groundwork for implementing more complex models.
:numref:`chap_cnn` and :numref:`chap_modern_cnn` present convolutional neural
networks (CNNs), the backbone of most modern computer vision systems.
:numref:`chap_rnn` and :numref:`chap_modern_rnn` introduce recurrent neural
networks (RNNs), which exploit sequential (e.g., temporal) structure in data
and are common in natural language processing and time series prediction.
:numref:`chap_attention-and-transformers` describes models based on
*attention mechanisms*, which have displaced RNNs as the dominant architecture
for most natural language processing tasks.

* **Part 3: Scalability, Efficiency, and Applications** (available [online](https://d2l.ai)).
Chapter 12 covers common optimization algorithms for training deep learning
models, and Chapter 13 the key factors behind the computational performance of
deep learning code. Chapter 14 illustrates major applications in computer
vision, and Chapters 15 and 16 show how to pretrain language representation
models and apply them to natural language processing tasks.


### Code
:label:`sec_code`

Most sections of this book feature executable code, because some intuitions
are best developed by trial and error: tweaking the code in small ways and
observing the results. Ideally, an elegant theory would tell us exactly how to
tweak our code, but deep learning practitioners often have to go where no
solid theory provides guidance. Formal explanations for why many techniques
work are still lacking: the mathematics can be very hard, the explanations
probably depend on properties of data that lack clear definitions, and serious
inquiry has only recently kicked into high gear. We hope future editions can
offer insights beyond what is available today.

To avoid repetition, we collect frequently used functions and classes in the
`d2l` package. Code blocks marked with `#@save` (functions, classes, or
imports) are saved to the `d2l` package for later use. :numref:`sec_d2l` gives
a detailed overview of these. The `d2l` package is lightweight and only needs
the following dependencies:

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`pytorch`
Most of the code in this book is based on PyTorch, a popular open-source
framework enthusiastically embraced by the deep learning research community.
All code has been tested with the latest stable PyTorch, but because deep
learning moves fast, some code *in the print edition* may break with future
versions; the online version is kept up to date. If you run into problems,
see :ref:`chap_installation` to update your code and runtime environment.
The dependencies of our PyTorch implementation are listed below.
:end_tab:

:begin_tab:`tensorflow`
Most of the code in this book is based on TensorFlow, an open-source deep
learning framework widely adopted in industry and popular among researchers.
All code has been tested with the latest stable TensorFlow, but some code
*in the print edition* may break with future versions; the online version is
kept up to date. If you run into problems, see :ref:`chap_installation`.
The dependencies of our TensorFlow implementation are listed below.
:end_tab:

:begin_tab:`jax`
Most of the code in this book is based on JAX, an open-source framework for
composable function transformations, such as differentiating arbitrary Python
and NumPy functions, JIT compilation, and vectorization. It is popular in
machine learning research and has an easy-to-learn, NumPy-like API (JAX aims
for 1:1 parity with NumPy, so switching can be as simple as changing an
import). Some code *in the print edition* may break with future versions; the
online version is kept up to date. If you run into problems, see
:ref:`chap_installation`. The dependencies of our JAX implementation are
listed below.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### Target Audience

This book is for students (undergraduate or graduate), engineers, and
researchers who want a solid grasp of the practical techniques of deep
learning. We explain every concept from scratch, so no background in deep or
machine learning is needed. We do assume some basics: modest amounts of linear
algebra, calculus, probability, and Python programming. If you have forgotten
anything, the
[online Appendix](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)
is a refresher on most of the mathematics in this book. We usually prioritize
intuition and ideas over mathematical rigor. To go beyond these prerequisites,
we recommend *Linear Analysis* by :citet:`Bollobas.1999` for linear algebra and
functional analysis in depth, *All of Statistics* :cite:`Wasserman.2013` for a
marvelous introduction to statistics, and Joe Blitzstein's
[books](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)
and [courses](https://projects.iq.harvard.edu/stat110/home) on probability and
inference. If you have not used Python before, try this
[Python tutorial](http://learnpython.org/).


### Notebooks and Website

All our notebooks can be downloaded from the [D2L.ai website](https://d2l.ai)
and from [GitHub](https://github.com/d2l-ai/d2l-en).



## Acknowledgments

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
This book was originally implemented with MXNet as the primary framework.
We thank Anirudh Dagar and Yuan Tang for adapting a majority part of earlier MXNet code into PyTorch and TensorFlow implementations, respectively.
Since July 2021, we have redesigned and reimplemented this book in PyTorch, MXNet, and TensorFlow, choosing PyTorch as the primary framework.
We thank Anirudh Dagar for adapting a majority part of more recent PyTorch code into JAX implementations.
We thank Gaosheng Wu, Liujun Hu, Ge Zhang, and Jiehang Xie from Baidu for adapting a majority part of more recent PyTorch code into PaddlePaddle implementations in the Chinese draft.
We thank Shuai Zhang for integrating the LaTeX style from the press into the PDF building.

On GitHub, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

We thank Amazon Web Services, especially Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
and Andrew Jassy for their generous support in writing this book.
Without the available time, resources, discussions with colleagues,
and continuous encouragement, this book would not have happened.
During the preparation of the book for publication,
Cambridge University Press has offered excellent support.
We thank our commissioning editor David Tranah
for his help and professionalism.


## Summary

Deep learning has revolutionized pattern recognition and now powers
technology across computer vision, natural language processing, and automatic
speech recognition. Applying it successfully requires knowing how to cast a
problem, the basic mathematics of modeling, the algorithms for fitting models
to data, and the engineering to implement it all. This book brings together
prose, figures, mathematics, and code in one place.



## Exercises

1. Install Python on your computer.
1. Look at the book structure in :numref:`fig_book_org`. Which of the preliminaries (linear algebra, calculus, probability, Python) do you feel least confident about, and which section will you study first to fill that gap?
