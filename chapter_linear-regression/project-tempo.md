# Project: Tempo
:label:`sec_project_tempo`

*Honest delivery-time estimates for a food-delivery app.*

**Uses:** everything up to this point, especially linear regression and
minibatch SGD (:numref:`sec_linear_regression`), the `Module` / `DataModule` /
`Trainer` design (:numref:`sec_oo-design`), the squared-loss vs. robust-loss
trade-off, generalization and validation (:numref:`sec_generalization_basics`),
and weight decay (:numref:`sec_weight_decay`); plus pandas preprocessing and
probability from Chapter 2. Nothing beyond Chapter 3.

**Time:** two or three evenings. **Build it in your own GitHub repository**
named `tempo`.

## The Problem

When you order food, the app promises "arrives in 32 min". That number is a
regression model's prediction, and it matters: promise too little and
customers are angry when the food is late; promise too much and they order
somewhere else. DoorDash, Uber Eats, Swiggy, and Zomato all run models like
this, and simple, well-engineered linear models are a strong, interpretable
baseline that teams keep around even after adding deep learning.

You'll build **Tempo**, an ETA model that predicts delivery time in minutes,
explains what drives it, and quotes an honest range rather than a single
number.

## Data

[Download deliveries.csv](../_static/projects/deliveries.csv) (4,000
simulated orders over 90 days, in time order). One row per order:

| Column | Meaning |
|:--|:--|
| `order_day` | day index 0--89 (orders are sorted by time) |
| `weekday` | 0 = Monday ... 6 = Sunday |
| `hour` | hour the order was placed (10--22) |
| `restaurant_id` | one of 60 restaurants, `R00`--`R59` |
| `distance_km` | restaurant-to-customer distance |
| `prep_min` | kitchen's estimated preparation time |
| `items` | number of items in the order |
| `raining` | 1 if it was raining |
| `delivery_min` | **target**: actual minutes from order to door |

Like real logs, it's messy: some `prep_min` and `distance_km` values are
missing, restaurants differ in ways no column captures directly, rush hours
and weekends are slower, noise grows with distance, and about 1% of orders
are disasters (a courier got lost) that took 35--90 extra minutes.

## Steps

Tempo is built **step by step while you read the chapter**: each step appears
at the end of its section, right after you learn what it needs, and
**revisits** earlier topics so nothing fades. All steps are listed here too.

```{=html}
<div class="d2l-project-steps"></div>
```

## Acceptance Criteria

- Time-based split, imputation fitted on the training data only.
- Test MAE (approximate; your numbers will differ slightly):

  | Model | Test MAE (min) |
  |:--|--:|
  | Average (baseline) | about 9.4 |
  | Numeric features | about 6.3 |
  | + rush hour, weekend | about 5.4 |
  | + restaurants | about 4.2 |
  | + Huber loss | about 4.0 |

- A coefficient table in minutes per unit (e.g., minutes per km, minutes per
  rainy order) with a one-line interpretation of each.
- The range from Step 7 covers 75--85% of test orders.
- Loss curves for at least one model, and the train/test gap for the
  one-week model.

## Stretch Goals

- Make the interval width depend on distance (model the noise level), and
  compare coverage for short vs. long trips.
- Compare your SGD solution with the closed-form least-squares solution (the "Analytic Solution" in
  :numref:`sec_linear_regression`). Do they agree?
- Predict `log(delivery_min)` instead and compare MAE. When is that better?

## Optional Side Projects

Short, fun builds for more practice with this chapter's ideas. Each is a
weekend at most; pick whichever excites you. Same rule: only what you've
learned so far.

### Scout: Find Good Hyperparameters Automatically

*A small hyperparameter search tool, like a mini Google Vizier.*

- **Why it's real:** choosing learning rates, batch sizes, and weight decay
  is a big part of every ML team's compute bill; Google, Meta, and Anthropic
  all run internal tuning services.
- **Uses:** validation sets and $K$-fold cross-validation
  (:numref:`sec_generalization_basics`), weight decay (:numref:`sec_weight_decay`),
  your SGD `Trainer` (:numref:`sec_linear_scratch`).
- **Build:** a function `search(train_fn, space, budget)` that runs random
  search over log-scaled ranges (learning rate, batch size, $\lambda$), uses
  $K$-fold cross-validation, and returns the best settings with a results
  table.
- **Done when:** on Tempo's data, Scout finds settings at least as good as
  your hand-tuned ones, and you show that random search beats a grid with
  the same budget.

### Drift: A Model That Keeps Up With the World

*Online learning that adapts when the data changes.*

- **Why it's real:** prices, demand, and user behavior shift over time
  (concept drift). Models at ride-sharing, delivery, and ad companies update
  continuously instead of retraining from scratch.
- **Uses:** SGD on one example at a time, learning rates, and why the IID
  assumption matters (:numref:`sec_generalization_basics`).
- **Build:** stream Tempo's orders in time order. Halfway through, simulate
  a change (rain now adds 15 minutes). Compare a model trained once on the
  first half against one that keeps doing SGD updates on each new order.
- **Done when:** a plot of rolling error over time shows the frozen model
  degrading and the online model recovering, and you explain how the
  learning rate trades stability for speed of adaptation.

### Watt: Forecast a Building's Energy Use

*Predict tomorrow's electricity demand from the weather.*

- **Why it's real:** DeepMind famously cut the energy used to cool Google's
  data centers by about 40% with learned models; utilities forecast demand
  every hour to plan power generation.
- **Uses:** linear regression, feature engineering, squared vs. robust loss,
  weight decay, time-based validation (all of this chapter).
- **Build:** simulate (or download a public dataset of) hourly temperature
  and energy use. Engineer features (temperature, temperature squared for
  heating *and* cooling, hour of day, weekday), train with a time-based
  split, and compare with "same hour yesterday".
- **Done when:** your model beats the "yesterday" baseline on the last two
  weeks, and you can show which features matter most.

## For Your README and Resume

Suggested layout: `tempo/data.py` (loading, splitting, features),
`tempo/model.py` (Module, losses), `tempo/train.py` (Trainer), a notebook with
the results table and plots, and `tests/`.

> *Built **Tempo**, a delivery-time (ETA) model for a food-delivery dataset:
> linear regression trained with my own minibatch-SGD framework, feature
> engineering (rush hour, restaurant effects), Huber loss for outliers, and
> calibrated 80% prediction ranges; cut ETA error by about 57% (MAE 9.4 → 4.0
> minutes) versus the average-time baseline.*

Use your own measured numbers in the bullet.
