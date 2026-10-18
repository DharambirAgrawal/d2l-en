# Introduction
:label:`chap_introduction`

Until recently, nearly every program you interacted with was coded as a rigid
set of rules specifying exactly how it should behave. Say we want to build an
e-commerce platform. After a few hours at a whiteboard we might settle on the
broad strokes: (i) users interact through a web or mobile interface; (ii) the
application uses a database to track each user's state and past transactions;
and (iii) at its heart, the *business logic* (the *brains*) spells out rules
mapping every conceivable circumstance to the action the program should take.

To build those brains, we enumerate the events the program must handle. When a
customer adds an item to their cart, add an entry to the shopping cart table
linking the user's ID to the product's ID. Then we step through corner cases
(what if a user checks out with an empty cart?) and fix our rules as needed.
It may take a few test runs to work out the kinks, but we can usually write
such programs and confidently launch them *before* seeing a real customer.
Designing automated systems by hand like this is a remarkable cognitive feat,
and when you can devise solutions that work $100\%$ of the time, you usually
shouldn't be worrying about machine learning.

Fortunately for machine learning scientists, many tasks we'd like to automate
don't bend so easily to human ingenuity. Imagine the smartest people you know
at the whiteboard, tackling one of these:

* Write a program that predicts tomorrow's weather given geographic information, satellite images, and a trailing window of past weather.
* Write a program that takes in a factoid question, expressed in free-form text, and  answers it correctly.
* Write a program that, given an image, identifies every person depicted in it and draws outlines around each.
* Write a program that presents users with products that they are likely to enjoy but unlikely, in the natural course of browsing, to encounter.

Even elite programmers would struggle to code these from scratch. Sometimes the
pattern changes over time, so there is no fixed right answer and a solution
must adapt to a changing world. Sometimes the relationship (say, between pixels
and abstract categories) is too complicated, requiring millions of
computations that follow unknown principles. For image recognition, the
precise steps lie beyond our conscious understanding, even though our
subconscious performs them effortlessly.

*Machine learning* is the study of algorithms that learn from experience: as
they accumulate experience, typically observational data or interactions with
an environment, their performance improves. Contrast this with our e-commerce
platform, which follows the same logic no matter how much experience accrues,
until its developers decide to update it. This book teaches the fundamentals
of machine learning, focusing on *deep learning*, a powerful set of techniques
driving innovation in computer vision, natural language processing,
healthcare, genomics, and more.

## A Motivating Example

Before writing this book, the authors needed coffee. In the car, Alex said
"Hey Siri" to wake up the iPhone's voice recognition, and Mu said "directions
to Blue Bottle coffee shop". The phone transcribed the command, recognized a
request for directions, and opened Maps, which showed several routes, each
with a predicted travel time. The story is made up, but it shows how a few
seconds of everyday phone use can involve several machine learning models.

Now imagine writing a program to respond to a *wake word* such as "Alexa",
"OK Google", or "Hey Siri", alone in a room with just a computer and a code
editor (:numref:`fig_wake_word`). How would you do it from first principles?
The problem is hard. The microphone collects roughly 44,000 samples per
second, each a measurement of the sound wave's amplitude. What rule could
reliably map a snippet of raw audio to a confident prediction in
$\{\textrm{yes}, \textrm{no}\}$ about whether it contains the wake word?
If you're stuck, don't worry: we don't know how to write such a program from
scratch either. That's why we use machine learning.

![Identify a wake word.](../img/wake-word.svg)
:label:`fig_wake_word`


Here's the trick. Even when we can't tell a computer explicitly how to map
inputs to outputs, we can often perform the task ourselves: you can't program
a computer to recognize "Alexa", but you can recognize it. So we collect a huge
*dataset* of audio snippets with labels saying which contain the wake word.
Instead of designing a system *explicitly* to recognize wake words, we define
a flexible program whose behavior is determined by a number of *parameters*,
and use the dataset to find the parameter values that give the best
performance on a chosen measure.

Think of the parameters as knobs that change the program's behavior. Once the
parameters are fixed, we call the program a *model*. The set of all programs
(input--output mappings) we can produce by turning the knobs is a *family* of
models, and the "meta-program" that uses the dataset to choose the parameters
is a *learning algorithm*.

Before running the learning algorithm, we must define the problem precisely:
the exact inputs and outputs, and an appropriate model family. Here, the
model takes a snippet of audio as *input* and produces a choice in
$\{\textrm{yes}, \textrm{no}\}$ as *output*. If all goes to plan, its guesses
will usually be correct.

With the right model family, some setting of the knobs makes the model say
"yes" every time it hears "Alexa". Since the wake word is arbitrary, the family
should be rich enough that another setting fires only on "Apricot": the two
tasks are intuitively similar, so one family should handle both. But
fundamentally different inputs or outputs, such as mapping images to captions
or English sentences to Chinese, may need an entirely different family.

If we set the knobs randomly, the model is unlikely to recognize "Alexa",
"Apricot", or any other word. *Learning* is the process of discovering the
knob settings that produce the desired behavior; in other words, we *train*
the model with data. As shown in :numref:`fig_ml_loop`, training usually
looks like this:

1. Start off with a randomly initialized model that cannot do anything useful.
1. Grab some of your data (e.g., audio snippets and corresponding $\{\textrm{yes}, \textrm{no}\}$ labels).
1. Tweak the knobs to make the model perform better as assessed on those examples.
1. Repeat Steps 2 and 3 until the model is awesome.

![A typical training process.](../img/ml-loop.svg)
:label:`fig_ml_loop`

So instead of coding a wake word recognizer, we code a program that can
*learn* to recognize wake words from a large labeled dataset. You can think
of this as *programming with data*. We can "program" a cat detector by
showing it many examples of cats and dogs, until it learns to output a large
positive number for a cat, a large negative number for a dog, and something
near zero when it's unsure. This barely scratches the surface of machine
learning, and deep learning, explained in more detail later, is just one of
many popular methods.


## Key Components

In the wake word example, we had a dataset of audio snippets with binary
labels and wanted a model mapping snippets to labels. Predicting an unknown
label from inputs, using examples whose labels are known, is called
*supervised learning*. It's one of many kinds of machine learning problems,
but they all share four core components:

1. The *data* that we can learn from.
1. A *model* of how to transform the data.
1. An *objective function* that quantifies how well (or badly) the model is doing.
1. An *algorithm* to adjust the model's parameters to optimize the objective function.

### Data

You can't do data science without data. For now we focus on the key
properties of the datasets we'll use. A dataset is a collection of examples,
which need a suitable numerical representation. Each *example* (or
*data point*, *data instance*, *sample*) consists of attributes called
*features* (or *covariates*, *inputs*) from which the model makes its
predictions. In supervised learning, the goal is to predict a special
attribute, the *label* (or *target*), which is not part of the model's input.

With image data, each example might be a photograph (the features) and a
number for its category (the label). The photograph is represented as three
grids of numbers giving the brightness of red, green, and blue light at each
pixel, so a $200\times 200$ color photo has $200\times200\times3=120000$
numerical values.

With electronic health records, we might predict whether a patient will
survive the next 30 days. The features could be readily available attributes
and measurements such as age, vital signs, comorbidities, current
medications, and recent procedures; the label is whether each patient in the
historical data survived the 30-day window.

When every example has the same number of numerical features, the inputs are
fixed-length vectors, and their (constant) length is the *dimensionality* of
the data. Fixed-length inputs are convenient, but not all data fits that mold.
Microscope images may come from standard equipment, but images from the
Internet vary in resolution and shape. Cropping them to a standard size only
goes so far and risks losing information. Text resists fixed-length
representations even more: customer reviews on sites like Amazon, IMDb, and
TripAdvisor range from "it stinks!" to pages of rambling. A major advantage of
deep learning is how gracefully it handles *varying-length* data.

Generally, more data makes our job easier: we can train more powerful models
and rely less on preconceived assumptions. The shift from small to big data
is a major reason for deep learning's success. Many of the most exciting deep
learning models don't work without large datasets, and others that work with
small data are no better than traditional approaches.

But lots of data, cleverly processed, isn't enough: we need the *right* data.
If the data is full of mistakes, or the features don't predict the target,
learning will fail: *garbage in, garbage out*. And poor accuracy isn't the
only risk. In sensitive applications like predictive policing, resume
screening, and lending risk models, garbage data can do real harm. A common
failure is a dataset in which some groups of people are unrepresented:
imagine a skin cancer recognition system that has never seen black skin.
Data can also reflect societal prejudices: a resume-screening model trained on
past hiring decisions could capture and automate historical injustices, all
without the data scientist intending it, or even noticing.


### Models

Most machine learning transforms data in some way: ingest photos and predict
smiley-ness, or ingest sensor readings and predict how anomalous they are. A
*model* is the computational machinery that takes data of one type and
outputs predictions of a possibly different type. We care about *statistical
models* that can be estimated from data. Simple models handle simple problems
fine, but the problems in this book stretch the limits of classical methods.
Deep learning stands apart mainly through its powerful models, which chain
many successive transformations of the data together, hence the name
*deep learning*. Along the way, we'll also cover some more traditional
methods.

### Objective Functions

Learning means improving at a task over time, but who decides what counts as
an improvement? To build a formal mathematical system of learning machines,
we need formal measures of how good (or bad) our models are. In machine
learning, and optimization generally, these are *objective functions*. By
convention, lower is better (any higher-is-better function can be turned into
one by flipping its sign), so they are also called *loss functions*.

For predicting numerical values, the most common loss is *squared error*, the
square of the difference between prediction and ground truth. For
classification, the most common objective is the error rate, the fraction of
examples where the prediction disagrees with the ground truth. Some objectives
(e.g., squared error) are easy to optimize, while others (e.g., error rate)
are hard to optimize directly because they are not differentiable or have
other complications. Then we optimize a *surrogate objective* instead.

During optimization, we treat the loss as a function of the model's
parameters, with the training data held constant, and find the best
parameters by minimizing the loss on the examples collected for training.
But doing well on training data doesn't guarantee doing well on unseen data.
So we split the data into a *training dataset* (or *training set*) for
learning the parameters, and a *test dataset* (or *test set*) held out for
evaluation, and report performance on both. Training performance is like a
student's scores on practice exams: encouraging results don't guarantee
success on the final, because the student may have memorized the practice
questions without mastering the topic. When a model does well on the training
set but fails to generalize to unseen data, it is *overfitting*.


### Optimization Algorithms

Given data, a model, and an objective function, we need an algorithm to search
for the parameters that minimize the loss. Popular optimization algorithms for
deep learning are based on *gradient descent*: at each step, check how the
training loss would change if each parameter were perturbed slightly, then
update each parameter in the direction that lowers the loss.


## Kinds of Machine Learning Problems

The wake word problem is just one of many problems machine learning can
tackle. To give you a common vocabulary for the rest of the book, here is a
broad overview of the landscape.

### Supervised Learning

In supervised learning, we are given a dataset with both features and labels
and asked to produce a model that predicts the labels from the features. Each
feature--label pair is an example (when the context is clear, we may also call
a collection of inputs *examples* even without their labels). It's
"supervised" because we, the supervisors, give the model labeled examples to
choose its parameters. In probabilistic terms, we usually want the conditional
probability of a label given the features. Supervised learning is just one
paradigm, but it accounts for most successful industrial applications, partly
because many important tasks can be stated crisply as estimating the
probability of something unknown given some available data:

* Predict cancer vs. not cancer, given a computer tomography image.
* Predict the correct translation in French, given a sentence in English.
* Predict the price of a stock next month based on this month's financial reporting data.

Though every supervised problem is "predict labels from features", it can
take diverse forms and require many modeling decisions, depending on the type,
size, and number of inputs and outputs. For example, sequences of arbitrary
length need different models than fixed-length vectors. We will study many of
these problems in depth.

Informally, the process goes like this. Take a big collection of examples with
known features, select a random subset, and get the ground truth label for
each. Sometimes the labels already exist (e.g., did a patient die within the
following year?); other times human annotators must provide them (e.g.,
assigning images to categories). These inputs and labels form the training
set. We feed it to a supervised learning algorithm, a function that takes a
dataset and outputs another function: the learned model. Finally, we feed
previously unseen inputs to the model and use its outputs as predicted labels.
:numref:`fig_supervised_learning` shows the whole process.

![Supervised learning.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`

#### Regression

*Regression* is perhaps the simplest supervised task to grasp. Take a database
of home sales as a table: each row is a house, and each column an attribute
such as square footage, number of bedrooms, number of bathrooms, and minutes
of walking to the town center. Each example is a house, and its feature vector
is one row. In New York or San Francisco (unless you're a big tech CEO), your
home's (sq. footage, bedrooms, bathrooms, walking distance) vector might be
$[600, 1, 1, 60]$; in Pittsburgh, more like $[3000, 4, 3, 10]$. Fixed-length
feature vectors like these are essential for most classic machine learning
algorithms.

What makes a problem regression is the form of the target. To estimate a
house's fair market value from such features, the data might be historical
listings and the labels the observed sale prices. When labels take arbitrary
numerical values (even within some interval), it's a *regression* problem,
and the goal is predictions that closely approximate the actual labels.

Many practical problems are regression problems. Predicting the rating a user
will give a movie is one; a great algorithm for it in 2009 might have won the
[1-million-dollar Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize).
Predicting how long patients stay in the hospital is another. A good rule of
thumb: any *how much?* or *how many?* problem is likely regression. For
example:

* How many hours will this surgery take?
* How much rainfall will this town have in the next six hours?


You have probably solved a regression problem informally already. Say a
contractor spent 3 hours clearing gunk from your sewage pipes and billed you
350 dollars, and your friend hired the same contractor for 2 hours and paid
250 dollars. To predict someone else's bill, you might assume more hours cost
more, with a base charge plus an hourly rate. If so, these two examples
already reveal the pricing: 100 dollars per hour plus 50 dollars to show up.
If you followed that, you understand the high-level idea of *linear*
regression.

Here we could find parameters that exactly match the contractor's prices.
Often that's impossible, e.g., when some variation comes from factors beyond
our two features. Then we learn models that minimize the distance between
predictions and observed values. Most chapters minimize the squared error
loss, which, as we'll see, corresponds to assuming the data were corrupted by
Gaussian noise.

#### Classification

Regression answers *how many?* questions, but many problems don't fit that
template. Say a bank wants its mobile app to read checks from a photo. If we
can segment out the image patch for each handwritten character, the remaining
task is to decide which character from a known set each patch shows. Such
*which one?* problems are called *classification*. They need different tools
from regression, though many techniques carry over.

In *classification*, the model looks at features, e.g., an image's pixel
values, and predicts which *category* (or *class*) from a discrete set an
example belongs to. Handwritten digits have ten classes, 0 through 9. With
only two classes, e.g., animal images labeled $\textrm{\{cat, dog\}}$, it's
*binary classification*. Where regression needs a regressor that outputs a
number, classification needs a classifier that outputs a predicted class.

For reasons we'll see later, models that output only a *firm* choice like
"cat" or "dog" are hard to optimize. It's much easier to speak in
probabilities: given an example's features, the model assigns a probability
to each class. A classifier might say an image is a cat with probability 0.9,
i.e., it is 90\% sure the image shows a cat. The magnitude of the predicted
class's probability conveys a notion of uncertainty (not the only one; we'll
discuss others in advanced chapters).

With more than two classes, it's *multiclass classification*, e.g.,
handwritten character recognition $\textrm{\{0, 1, 2, ... 9, a, b, c, ...\}}$.
Where regression minimizes squared error, classification commonly uses the
*cross-entropy* loss, whose name we'll demystify when we introduce information
theory.

The most likely class isn't necessarily the one you should act on. Say you
find a beautiful mushroom in your backyard (:numref:`fig_death_cap`).

![Death cap---do not eat!](../img/death-cap.jpg)
:width:`200px`
:label:`fig_death_cap`

Your trained poison classifier says the probability that
:numref:`fig_death_cap` shows a death cap is 0.2, i.e., it is 80\% sure the
mushroom is safe. You'd still be a fool to eat it: a delicious dinner isn't
worth a 20\% risk of dying. To decide, we compute the expected detriment of
each action, which depends on both the likely outcomes and the benefit or harm
of each. Eating the mushroom might cost $0.2 \times \infty + 0.8 \times 0 = \infty$,
while discarding it costs $0.2 \times 0 + 0.8 \times 1 = 0.8$. Our caution was
justified: as any mycologist would tell us, the mushroom in
:numref:`fig_death_cap` is actually a death cap.

Classification can get more complicated than binary or multiclass. In
*hierarchical classification*, classes form a hierarchy and not all errors are
equal: if we must err, we'd rather confuse related classes than distant ones.
Think of [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus), who organized
fauna in a hierarchy. Mistaking a poodle for a schnauzer isn't so bad, but
confusing a poodle with a dinosaur deserves a huge penalty. Which hierarchy
matters depends on the use: rattlesnakes and garter snakes are close on the
phylogenetic tree, but mistaking a rattler for a garter could be fatal.

#### Tagging

Some classification problems fit the binary or multiclass setup neatly, such
as telling cats from dogs, which off-the-shelf tools now do easily. But
however accurate that classifier is, it's in trouble when it sees the
*Town Musicians of Bremen*, a German fairy tale featuring four animals
(:numref:`fig_stackedanimals`).

![A donkey, a dog, a cat, and a rooster.](../img/stackedanimals.png)
:width:`300px`
:label:`fig_stackedanimals`

The photo shows a cat, a rooster, a dog, and a donkey, with trees in the
background. If we expect such images, multiclass classification is the wrong
formulation: we want the model to be able to say the image shows a cat, a
dog, a donkey, *and* a rooster.

Predicting classes that are not mutually exclusive is called
*multi-label classification*, and auto-tagging problems are usually best
described this way. Think of tags on a technical blog: "machine learning",
"technology", "gadgets", "programming languages", "Linux", "cloud computing",
"AWS". A typical article has 5--10 tags, and tags are correlated: posts about
"cloud computing" likely mention "AWS", and posts about "machine learning"
likely mention "GPUs".

Label sets can be enormous. The National Library of Medicine employs
professional annotators to tag each article indexed in PubMed with terms from
the Medical Subject Headings (MeSH) ontology, about 28,000 tags. Correct tags
let researchers do exhaustive literature reviews, but tagging is slow, with
typically a one-year lag after archiving. Machine learning can provide
provisional tags until a proper manual review, and the BioASQ organization has
[hosted competitions](http://bioasq.org/) for this task for several years.

#### Search

In information retrieval, we often rank sets of items. In web search, the
goal is less to decide *whether* a page is relevant to a query than which of
the relevant results to show most prominently to a particular user. One way
is to score every item and return the top-rated ones.
[PageRank](https://en.wikipedia.org/wiki/PageRank), the original secret sauce
of Google search, was an early scoring system. Oddly, its scores didn't depend
on the query: a simple relevance filter found candidate pages, and PageRank
prioritized the more authoritative ones. Today, search engines use machine
learning and behavioral models to compute query-dependent relevance scores,
and entire academic conferences are devoted to the subject.

#### Recommender Systems
:label:`subsec_recommender_systems`

Recommender systems are related to search and ranking: both display items
relevant to the user. The difference is the emphasis on *personalization*.
For movies, a science fiction fan and a connoisseur of Peter Sellers comedies
might see very different results. The same goes for retail products, music,
and news.

Sometimes customers give explicit feedback, such as ratings and reviews on
Amazon, IMDb, or Goodreads. Other times feedback is implicit: skipping a song
on a playlist might mean dissatisfaction, or just that the song didn't fit the
moment. In the simplest formulations, these systems estimate a score, such as
an expected star rating or the probability that a user will buy an item.

With such a model, we can recommend the items with the highest scores for a
given user. Production systems are far more advanced and use detailed user
activity and item characteristics. :numref:`fig_deeplearning_amazon` shows
deep learning books Amazon recommended based on Aston's preferences.

![Deep learning books recommended by Amazon.](../img/deeplearning-amazon.jpg)
:label:`fig_deeplearning_amazon`

Despite their enormous economic value, recommender systems built naively on
predictive models have serious conceptual flaws. We only observe
*censored feedback*: users mostly rate movies they feel strongly about, so on
a five-point scale there are many one- and five-star ratings but few
three-star ones. And current purchases are often driven by the current
recommendation algorithm, which learning algorithms don't always account for.
This creates feedback loops: the system pushes an item, which then looks
better (more purchases), and gets recommended even more. Censoring,
incentives, and feedback loops are important open research questions.

#### Sequence Learning

So far, our problems have had a fixed number of inputs and outputs: house
prices from a fixed set of features, class probabilities from a fixed-size
image, star ratings from a user ID and product ID. Once trained, the model
forgets each test example as soon as it's processed; we assume successive
observations are independent, so there's no context to keep.

But what about video, where each snippet has a different number of frames and
our guess about each frame improves with the frames around it? Or language,
e.g., machine translation, where we ingest a sentence in one language and
predict its translation in another? Or medicine: a model monitoring patients
in intensive care should alert us when their risk of dying in the next 24
hours exceeds a threshold, and we wouldn't want it to discard the patient's
history every hour and use only the latest measurements.

These are among the most exciting applications of machine learning, and they
are instances of *sequence learning*: the model ingests sequences of inputs,
emits sequences of outputs, or both. *Sequence-to-sequence learning* is the
case where both inputs and outputs are variable-length sequences, as in
machine translation and speech-to-text transcription. A few special cases are
worth mentioning.

**Tagging and Parsing**. Annotating a text sequence with attributes. Inputs
and outputs are *aligned*: same number, corresponding order. In
*part-of-speech (PoS) tagging*, we label every word with its part of speech,
e.g., "noun" or "direct object". Or we might mark which groups of words are
named entities, like *people*, *places*, or *organizations*. In the
cartoonishly simple example below, we just indicate whether each word is part
of a named entity (tagged "Ent").

```text
Tom has dinner in Washington with Sally
Ent  -    -    -     Ent      -    Ent
```


**Automatic Speech Recognition**. The input is an audio recording of a speaker
(:numref:`fig_speech`) and the output is a transcript. There are far more
audio frames (sound is typically sampled at 8kHz or 16kHz) than characters,
so there's no 1:1 correspondence: thousands of samples may make up one word.
It's sequence-to-sequence learning with an output much shorter than the input.
Humans recognize speech remarkably well even from poor audio, but it's a
formidable challenge for computers.

![`-D-e-e-p- L-ea-r-ni-ng-` in an audio recording.](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Text to Speech**. The inverse of speech recognition: the input is text and
the output is audio, much longer than the input.

**Machine Translation**. Unlike speech recognition, where inputs and outputs
occur in the same order, translation must handle unaligned data: the
sequences can have different lengths, and corresponding parts may appear in a
different order. Consider the German tendency to put verbs at the end of
sentences:

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Have you already looked at this excellent textbook?
Wrong alignment:  Have you yourself already this excellent textbook looked at?
```


Related problems appear elsewhere. Determining the order in which a user
reads a webpage is a two-dimensional layout analysis problem. Dialogue adds
more complications: deciding what to say next requires real-world knowledge
and the state of the conversation across long time spans. These are active
areas of research.

### Unsupervised and Self-Supervised Learning

So far we've focused on supervised learning, where the model gets a giant
dataset of features and labels. The supervised learner has a very specialized
job and a dictatorial boss who says exactly what to do in every situation
until the learner can map situations to actions. That sounds lame, but such a
boss is easy to please: recognize the pattern quickly and imitate.

The opposite is a boss with no idea what they want. As a data scientist, get
used to it: you may be handed a giant dump of data and told to
*do some data science with it!* That's vague because it is vague. This class
of problems is *unsupervised learning*, and the questions we can ask are
limited only by our creativity. We'll cover unsupervised techniques in later
chapters; to whet your appetite, here are a few questions you might ask.

* Can we find a small number of prototypes that accurately summarize the data?
Given photos, can we group them into landscapes, dogs, babies, cats, and
mountain peaks? Given users' browsing activity, can we group users with
similar behavior? This is typically called *clustering*.
* Can we find a few parameters that capture the data's relevant properties?
A ball's trajectory is well described by its velocity, diameter, and mass,
and tailors use a few parameters to describe body shape well enough to fit
clothes. This is *subspace estimation*; if the dependence is linear, it's
*principal component analysis*.
* Is there a representation of (arbitrarily structured) objects in Euclidean
space in which symbolic properties match well? It can describe entities and
their relations, such as "Rome" $-$ "Italy" $+$ "France" $=$ "Paris".
* Is there a description of the root causes of the data we observe? Given
demographic data on house prices, pollution, crime, location, education, and
salaries, can we discover how they relate purely from empirical data? The
fields of *causality* and *probabilistic graphical models* tackle such
questions.
* An exciting recent development is *deep generative models*, which estimate
the density of the data, explicitly or *implicitly*. A trained generative
model can score examples by how likely they are, or sample synthetic examples
from the learned distribution. Early breakthroughs came with
*variational autoencoders* :cite:`Kingma.Welling.2014,rezende2014stochastic`
and *generative adversarial networks* :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`,
followed by normalizing flows :cite:`dinh2014nice,dinh2017density` and
diffusion models :cite:`sohl2015deep,song2019generative,ho2020denoising,song2021score`.



Another development is *self-supervised learning*, which uses some aspect of
unlabeled data to provide supervision. For text, models learn to "fill in the
blanks", predicting randomly masked words from their surrounding words in big
corpora, with no labeling effort :cite:`Devlin.Chang.Lee.ea.2018`! For images,
models can learn to tell the relative position of two crops of the same
image :cite:`Doersch.Gupta.Efros.2015`, to predict an occluded part of an
image from the rest, or to tell whether two examples are perturbed versions
of the same image. Self-supervised models often learn representations that
are then fine-tuned for a downstream task of interest.


### Interacting with an Environment

We haven't yet discussed where data comes from or what happens when a model
produces an output, because supervised and unsupervised learning don't
address this in a sophisticated way. In both, we grab a big pile of data
upfront and run our pattern recognition machines without ever interacting
with the environment again. Since all learning happens after the algorithm is
disconnected from the environment, this is called *offline learning*.
Supervised learning, for example, assumes the simple interaction pattern in
:numref:`fig_data_collection`.

![Collecting data for supervised learning from an environment.](../img/data-collection.svg)
:label:`fig_data_collection`

Offline learning's simplicity has its charms: we can study pattern
recognition in isolation, without the complications of a dynamic
environment. But it's limiting. If you grew up reading Asimov's Robot novels,
you picture intelligent *agents* that take actions in the world, not just
predictive models. Unlike predictions, actions change the environment, so to
train an agent we must account for how its actions affect its future
observations, and offline learning won't do.

Interacting with an environment raises many new modeling questions, such as:

* Does the environment remember what we did previously?
* Does the environment want to help us, e.g., a user reading text into a speech recognizer?
* Does the environment want to beat us, e.g., spammers adapting their emails to evade spam filters?
* Does the environment have shifting dynamics? For example, would future data always resemble the past or would the patterns change over time, either naturally or in response to our automated tools?

These questions raise the problem of *distribution shift*, where training and
test data differ, like an exam written by the lecturer when the homework was
written by the teaching assistants. Next, we briefly describe reinforcement
learning, a rich framework for problems in which an agent interacts with an
environment.


### Reinforcement Learning

To build an agent that interacts with an environment and takes actions,
you'll probably end up using *reinforcement learning*, with applications in
robotics, dialogue systems, and even AI for video games.
*Deep reinforcement learning*, which applies deep learning to reinforcement
learning problems, has surged in popularity. Two prominent examples are the
deep Q-network that beat humans at Atari games from visual input alone
:cite:`mnih2015human`, and AlphaGo, which dethroned the world champion at Go
:cite:`Silver.Huang.Maddison.ea.2016`.

Reinforcement learning is a very general framework in which an agent
interacts with an environment over a series of time steps. At each step, the
agent receives an *observation* from the environment and chooses an *action*,
which is transmitted back to the environment via some mechanism (sometimes
called an *actuator*); after each loop, the agent receives a reward
(:numref:`fig_rl-environment`). Then it receives the next observation,
chooses the next action, and so on. The agent's behavior is governed by a
*policy*, a function mapping observations to actions. The goal of
reinforcement learning is to produce good policies.

![The interaction between reinforcement learning and an environment.](../img/rl-environment.svg)
:label:`fig_rl-environment`

The framework is extremely general. Supervised learning can even be recast as
reinforcement learning: for a classification problem, create an agent with one
action per class and an environment whose reward equals the original loss
function.

Reinforcement learning also handles many problems supervised learning can't.
In supervised learning, each training input comes with the correct label. In
reinforcement learning, the environment doesn't tell us the optimal action
for each observation; we just get some reward, and possibly no indication of
which actions led to it.

In chess, the only real reward comes at the end: say $1$ for a win and $-1$
for a loss. So reinforcement learners face the *credit assignment* problem:
deciding which actions to credit or blame for an outcome. Likewise, an
employee promoted on October 11 probably earned it through many good actions
over the past year, and getting promoted again means figuring out which ones
mattered.

Reinforcement learners may also face partial observability: the current
observation may not reveal the full state. If your cleaning robot is trapped
in one of many identical closets, rescuing it requires inferring its exact
location, which may mean considering observations from before it entered the
closet.

Finally, a reinforcement learner may know one good policy while many better
ones remain untried. It must constantly choose whether to *exploit* the best
currently known strategy or to *explore* the space of strategies, possibly
giving up short-term reward in exchange for knowledge.

In general, then, actions affect subsequent observations, rewards are only
observed for the chosen actions, and the environment may be fully or
partially observed. Handling all of this at once may be too much, and not
every practical problem has all this complexity, so researchers study
special cases. When the environment is fully observed, the problem is a
*Markov decision process*. When the state doesn't depend on previous actions,
it's a *contextual bandit problem*. When there's no state, just a set of
actions with initially unknown rewards, it's the classic
*multi-armed bandit problem*.

## Roots

We've reviewed a small subset of the problems machine learning can address,
and deep learning provides powerful tools for many of them. Although many
deep learning methods are recent, the core ideas of learning from data have
been studied for centuries. The desire to analyze data and predict the future
is at the root of much of natural science and mathematics. Consider the
Bernoulli distribution, named after
[Jacob Bernoulli (1655--1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli),
and the Gaussian distribution discovered by
[Carl Friedrich Gauss (1777--1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
Gauss invented the least mean squares algorithm, still used today for
problems from insurance calculations to medical diagnostics. Such tools
strengthened experimental natural science: Ohm's law relating current and
voltage in a resistor, for instance, is perfectly described by a linear model.

Even in the Middle Ages, mathematicians had a keen intuition for estimates.
The geometry book of [Jacob Köbel (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)
illustrates averaging the feet of 16 adult men to estimate typical foot
length (:numref:`fig_koebel`).

![Estimating the length of a foot.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`


As people left church, 16 adult men lined up and had their feet measured,
and the total was divided by 16 to estimate what we now call one foot. The
"algorithm" was later improved to handle misshapen feet: the men with the
shortest and longest feet were sent away and only the rest averaged. This is
one of the earliest examples of a trimmed mean estimate.

Statistics really took off as data became available. One pioneer,
[Ronald Fisher (1890--1962)](https://en.wikipedia.org/wiki/Ronald_Fisher),
contributed greatly to its theory and to its applications in genetics. Many
of his algorithms (such as linear discriminant analysis) and concepts (such as
the Fisher information matrix) are still foundations of modern statistics,
and the Iris dataset he released in 1936 is still sometimes used to
demonstrate machine learning algorithms. Fisher was also a proponent of
eugenics, a reminder that the morally dubious use of data science has as long
a history as its productive use in industry and science.


Machine learning was also influenced by the information theory of
[Claude Shannon (1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon)
and the theory of computation of
[Alan Turing (1912--1954)](https://en.wikipedia.org/wiki/Alan_Turing).
In *Computing Machinery and Intelligence* :cite:`Turing.1950`, Turing asked
"can machines think?" and proposed what is now called the Turing test: a
machine is *intelligent* if a human evaluator has trouble telling its
replies from a human's, based purely on text.

Neuroscience and psychology were further influences: humans clearly behave
intelligently, and many scholars asked whether that capacity could be
explained and reverse engineered. One of the first biologically inspired
algorithms came from [Donald Hebb (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).
In *The Organization of Behavior* :cite:`Hebb.1949`, he posited that neurons
learn by positive reinforcement, which became the Hebbian learning rule. This
inspired Rosenblatt's perceptron learning algorithm and laid the foundations
of the stochastic gradient descent algorithms behind deep learning today:
reinforce desirable behavior and diminish undesirable behavior to find good
parameter settings for a neural network.

Biological inspiration gave *neural networks* their name. For over a century
(back to the models of Alexander Bain, 1873, and James Sherrington, 1890),
researchers have tried to build computational circuits resembling networks of
interacting neurons. The biology has become less literal over time, but the
name stuck. At its heart are a few principles found in most networks today:

* The alternation of linear and nonlinear processing units, often referred to as *layers*.
* The use of the chain rule (also known as *backpropagation*) for adjusting parameters in the entire network at once.

After rapid early progress, neural network research languished from about
1995 to 2005, for two main reasons. First, training networks is
computationally expensive, and while memory was plentiful at the end of the
last century, compute was scarce. Second, datasets were small: Fisher's 1936
Iris dataset was still a popular benchmark, and MNIST, with 60,000
handwritten digits, was considered huge.

With scarce data and compute, strong statistical tools such as kernel
methods, decision trees, and graphical models were empirically superior in
many applications. Unlike neural networks, they didn't take weeks to train,
and they gave predictable results with strong theoretical guarantees.

## The Road to Deep Learning

Much of this changed with massive amounts of data, thanks to the World Wide
Web, companies serving hundreds of millions of users online, cheap
high-quality sensors, inexpensive storage (Kryder's law), and cheap
computation (Moore's law). In particular, GPUs, originally engineered for
gaming, revolutionized computation for deep learning. Suddenly, algorithms
and models that seemed computationally infeasible were within reach, as
:numref:`tab_intro_decade` illustrates.

:Dataset vs. computer memory and computational power
:label:`tab_intro_decade`

|Decade|Dataset|Memory|Floating point calculations per second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (house prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|


Memory has not kept pace with the growth in data, while compute has grown
faster than datasets. So statistical models must be more memory efficient,
and they can spend more cycles optimizing parameters. As a result, the sweet
spot in machine learning and statistics moved from (generalized) linear models
and kernel methods to deep neural networks. This is also why many mainstays
of deep learning, such as multilayer perceptrons :cite:`McCulloch.Pitts.1943`,
convolutional neural networks :cite:`LeCun.Bottou.Bengio.ea.1998`, long
short-term memory :cite:`Hochreiter.Schmidhuber.1997`, and Q-Learning
:cite:`Watkins.Dayan.1992`, were essentially "rediscovered" in the past decade
after lying dormant for a long time.

Recent progress in models, applications, and algorithms has been likened to
the Cambrian explosion, a moment of rapid progress in the evolution of
species. And the state of the art isn't just old algorithms with more
resources. The ideas below barely scratch the surface of what drove the past
decade's progress.


* Novel methods for capacity control, such as *dropout*
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`, which injects noise
  :cite:`Bishop.1995` throughout the network during training, have helped
  mitigate overfitting.
* *Attention mechanisms* solved a problem that had plagued statistics for over
  a century: how to increase a system's memory and complexity without
  increasing the number of learnable parameters. The elegant solution is a
  *learnable pointer structure* :cite:`Bahdanau.Cho.Bengio.2014`. Instead of
  remembering an entire text sequence (e.g., for machine translation) in a
  fixed-dimensional representation, the model only stores a pointer to the
  intermediate state of the translation. This greatly improved accuracy on
  long sequences, since the model no longer had to remember the whole input
  before generating the output.
* Built solely on attention, the *Transformer* architecture
  :cite:`Vaswani.Shazeer.Parmar.ea.2017` scales better: it improves with more
  data, larger models, and more training compute :cite:`kaplan2020scaling`. It
  has succeeded in natural language processing
  :cite:`Devlin.Chang.Lee.ea.2018,brown2020language`, computer vision
  :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`, speech recognition
  :cite:`gulati2020conformer`, reinforcement learning :cite:`chen2021decision`,
  and graph neural networks :cite:`dwivedi2020generalization`. A single
  Transformer pretrained on text, images, joint torques, and button presses
  can play Atari, caption images, chat, and control a robot
  :cite:`reed2022generalist`.
* *Language models* model the probability of text sequences and can predict
  text from other text. Scaling up data, models, and compute has unlocked a
  growing range of capabilities through human-like text generation
  :cite:`brown2020language,rae2021scaling,hoffmann2022training,chowdhery2022palm,openai2023gpt4,anil2023palm,touvron2023llama,touvron2023llama2`.
  Aligned with human intent :cite:`ouyang2022training`, OpenAI's
  [ChatGPT](https://chat.openai.com/) lets users solve problems such as code
  debugging and creative writing in conversation.
* Multi-stage designs, such as memory networks
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015` and the neural
  programmer-interpreter :cite:`Reed.De-Freitas.2015`, let modelers describe
  iterative reasoning. The network's internal state is modified repeatedly,
  carrying out successive steps in a chain of reasoning, much as a processor
  modifies memory during a computation.
* A key development in *deep generative modeling* was
  *generative adversarial networks*
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`. Traditional density
  estimation focused on finding proper probability distributions and (often
  approximate) samplers, which limited flexibility. GANs instead use an
  arbitrary algorithm with differentiable parameters as the sampler, adjusted
  until a discriminator (effectively a two-sample test) can't tell fake from
  real data. This opened density estimation to a wide variety of techniques:
  galloping zebras :cite:`Zhu.Park.Isola.ea.2017`, fake celebrity faces
  :cite:`Karras.Aila.Laine.ea.2017`, and photorealistic images from amateur
  sketches of a scene's layout :cite:`Park.Liu.Wang.ea.2019`.
* A diffusion process gradually adds random noise to data; *diffusion models*
  :cite:`sohl2015deep,ho2020denoising` learn to reverse it, gradually
  constructing samples from random noise. They have started replacing GANs in
  recent generative models such as DALL-E 2 :cite:`ramesh2022hierarchical` and
  Imagen :cite:`saharia2022photorealistic`, which generate images from text
  descriptions.
* Often a single GPU can't process all the training data, and parallel and
  distributed training has improved greatly over the past decade. A key
  challenge is that stochastic gradient descent, the workhorse of deep
  learning optimization, processes relatively small minibatches, while small
  batches limit GPU efficiency. Training on 1,024 GPUs with 32 images each
  means an aggregate minibatch of about 32,000 images. Work by
  :citet:`Li.2017`, and then :citet:`You.Gitman.Ginsburg.2017` and
  :citet:`Jia.Song.He.ea.2018`, pushed this to 64,000 observations, cutting
  ResNet-50 training on ImageNet from days to under 7 minutes.
* Parallel computation also drove progress in *reinforcement learning*,
  leading to superhuman performance in Go, Atari games, Starcraft, and physics
  simulations (e.g., MuJoCo) wherever environment simulators exist; see
  :citet:`Silver.Huang.Maddison.ea.2016` on AlphaGo. Reinforcement learning
  works best with plenty of (state, action, reward) tuples, and simulation
  provides them.
* Deep learning frameworks have been crucial for spreading ideas. The first
  generation of open-source frameworks, [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch), and [Theano](https://github.com/Theano/Theano),
  was used for many seminal papers. It was superseded by
  [TensorFlow](https://github.com/tensorflow/tensorflow) (often used via its
  high-level API [Keras](https://github.com/keras-team/keras)),
  [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2),
  and [Apache MXNet](https://github.com/apache/incubator-mxnet). The third
  generation consists of *imperative* tools, a trend arguably started by
  [Chainer](https://github.com/chainer/chainer), which described models with
  NumPy-like syntax, and adopted by [PyTorch](https://github.com/pytorch/pytorch),
  the [Gluon API](https://github.com/apache/incubator-mxnet) of MXNet, and
  [JAX](https://github.com/google/jax).


This division of labor, with systems researchers building better tools and
modelers building better networks, has simplified things enormously. In 2014,
training a linear logistic regression model was a nontrivial homework problem
for new machine learning Ph.D. students at Carnegie Mellon University. Today
it takes under 10 lines of code, within reach of any programmer.


## Success Stories

AI has long delivered results that would be hard to achieve otherwise. Mail
sorting with optical character recognition has been deployed since the 1990s
(the source of the famous MNIST dataset of handwritten digits). So has reading
checks for bank deposits and scoring applicants' creditworthiness. Financial
transactions are automatically checked for fraud, the backbone of payment
systems such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, and MasterCard.
Chess programs have been competitive for decades. Machine learning powers
search, recommendation, personalization, and ranking on the Internet. It is
pervasive, though often hidden from sight.

Only recently has AI been in the limelight, mostly for solving
consumer-facing problems previously considered intractable, and many of these
advances are due to deep learning.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, and Google's
  assistant, respond to spoken requests reasonably accurately, from menial
  jobs like switching on lights to booking barber's appointments and phone
  support dialog. This is probably the most noticeable sign of AI in our lives.
* A key ingredient of digital assistants is accurate speech recognition, which
  has reached parity with humans for certain applications
  :cite:`Xiong.Wu.Alleva.ea.2018`.
* Object recognition has come a long way too. In 2010, identifying the object
  in a picture was hard: on the ImageNet benchmark, researchers from NEC Labs
  and the University of Illinois at Urbana-Champaign achieved a top-five error
  rate of 28% :cite:`Lin.Lv.Zhu.ea.2010`. By 2017, it was 2.25%
  :cite:`Hu.Shen.Sun.2018`. There have been similarly stunning results in
  identifying birdsong and diagnosing skin cancer.
* Games used to be a measuring stick for human ability. Starting with
  TD-Gammon, which played backgammon using temporal difference reinforcement
  learning, algorithmic and computational progress has produced algorithms for
  a wide range of applications. Chess has a far more complex state space and
  action set than backgammon; DeepBlue beat Garry Kasparov using massive
  parallelism, special-purpose hardware, and efficient game-tree search
  :cite:`Campbell.Hoane-Jr.Hsu.2002`. Go, with its huge state space, is harder
  still; AlphaGo reached human parity in 2015 by combining deep learning with
  Monte Carlo tree sampling :cite:`Silver.Huang.Maddison.ea.2016`. Poker's
  state space is large and only partially observed (we don't know the
  opponents' cards); Libratus exceeded human performance using efficiently
  structured strategies :cite:`Brown.Sandholm.2017`.
* Self-driving vehicles are another sign of progress. Full autonomy isn't
  here yet, but Tesla, NVIDIA, and Waymo ship products with partial autonomy.
  Full autonomy is hard because driving requires perceiving, reasoning, and
  following rules. Today deep learning is used mainly for the visual part;
  the rest is heavily tuned by engineers.



This barely scratches the surface. Robotics, logistics, computational
biology, particle physics, and astronomy owe some of their most impressive
recent advances at least partly to machine learning, which is becoming a
ubiquitous tool for engineers and scientists.

Non-technical articles often raise fears of an AI apocalypse or a
*singularity*: that machine learning systems will become sentient and make
decisions that affect human lives independently of their programmers. AI
already affects livelihoods directly: creditworthiness is assessed
automatically, autopilots mostly navigate vehicles, and bail decisions use
statistical data. More frivolously, we can ask Alexa to switch on the coffee
machine.

Fortunately, we are far from a sentient AI that could deliberately manipulate
its creators. First, AI systems are engineered, trained, and deployed for
specific goals; whatever illusion of general intelligence they give, they are
a combination of rules, heuristics, and statistical models. Second, there are
currently no tools for *artificial general intelligence* that can improve
themselves, reason about themselves, and modify, extend, and improve their own
architecture while solving general tasks.

A much more pressing concern is how AI is used in daily life. Many routine
tasks now done by humans can and will be automated. Farm robots may lower
costs for organic farmers but also automate harvesting. This phase of the
industrial revolution may profoundly affect large parts of society, since
menial jobs provide much of the employment in many countries. And statistical
models applied without care can introduce racial, gender, or age bias, raising
reasonable concerns about procedural fairness when they drive consequential
decisions. These algorithms must be used with care. With what we know today,
this is a far more pressing concern than a malevolent superintelligence
destroying humanity.


## The Essence of Deep Learning

So far we've talked broadly about machine learning. Deep learning is the
subset concerned with models based on many-layered neural networks: it is
*deep* because its models learn many *layers* of transformations. That may
sound narrow, but it has produced a dizzying array of models, techniques,
problem formulations, and applications. Arguably, all machine learning has
many layers of computation, the first being feature processing. What sets
deep learning apart is that the operations at each of the many layers of
representations are learned jointly from data.

The problems we've discussed, learning from raw audio, raw image pixels, or
mapping sentences of arbitrary length to their translations, are exactly
where deep learning excels and traditional methods falter: many-layered
models handle low-level perceptual data in ways earlier tools could not.
Arguably the most significant common thread is *end-to-end training*: instead
of assembling a system from individually tuned components, we build the whole
system and tune it jointly. Computer vision, for example, used to separate
*feature engineering* from building machine learning models. The Canny edge
detector :cite:`Canny.1987` and Lowe's SIFT feature extractor :cite:`Lowe.2004`
reigned for over a decade as ways to map images to feature vectors, and the
crucial step in applying machine learning was hand-engineering such
transformations into a form shallow models could use. But human ingenuity can
only do so much compared with an algorithm consistently evaluating millions of
choices. When deep learning took over, these feature extractors were replaced
by automatically tuned filters with superior accuracy.

So a key advantage of deep learning is that it replaces not only the shallow
models at the end of traditional pipelines but also labor-intensive feature
engineering. And by replacing much domain-specific preprocessing, it has
erased many boundaries between computer vision, speech recognition, natural
language processing, medical informatics, and other areas, offering a unified
set of tools for diverse problems.

Beyond end-to-end training, we are moving from parametric statistical
descriptions to fully nonparametric models. With scarce data, useful models
need simplifying assumptions about reality; with abundant data, nonparametric
models that fit the data better can replace them. This mirrors physics in the
mid-twentieth century: once computers arrived, instead of solving parametric
approximations of electron behavior by hand, physicists could numerically
simulate the underlying partial differential equations. The models became
much more accurate, though often less interpretable.

Another difference from earlier work is the acceptance of suboptimal
solutions, of nonconvex nonlinear optimization problems, and of trying things
before proving them. This new empiricism, combined with a rapid influx of
talent, has led to rapid progress in practical algorithms, though often by
modifying and reinventing tools that existed for decades.

Finally, the deep learning community prides itself on sharing tools across
academic and corporate boundaries, releasing many excellent libraries,
statistical models, and trained networks as open source. In that spirit, the
notebooks forming this book are freely available to distribute and use. We
have worked hard to lower the barriers for anyone who wants to learn deep
learning, and we hope our readers benefit.


## Summary

Machine learning studies how computer systems can use experience (often data)
to improve at specific tasks. It combines ideas from statistics, data mining,
and optimization, and is often used to implement AI solutions.
Representation learning, a class of machine learning, focuses on
automatically finding the right way to represent data. Deep learning is
multi-level representation learning through many layers of transformations,
and it replaces not only the shallow models at the end of traditional
pipelines but also labor-intensive feature engineering. Much recent progress
was triggered by abundant data from cheap sensors and Internet-scale
applications, and by major advances in computation, mostly GPUs. Efficient
deep learning frameworks have also made designing and implementing whole
system optimization much easier, a key component of high performance.

## Exercises

1. Which parts of code that you are currently writing could be "learned",
   i.e., improved by learning and automatically determining design choices
   that are made in your code?
   Does your code include heuristic design choices?
   What data might you need to learn the desired behavior?
1. Which problems that you encounter have many examples for their solution,
   yet no specific way for automating them?
   These may be prime candidates for using deep learning.
1. Describe the relationships between algorithms, data, and computation. How do characteristics of the data and the current available computational resources influence the appropriateness of various algorithms?
1. Name some settings where end-to-end training is not currently the default approach but where it might be useful.
