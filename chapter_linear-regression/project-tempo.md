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

## Milestones

1. **Split by time, not at random.** Train on days 0--71 and test on days
   72--89, the way the model would actually be used (predicting the future
   from the past). Why would a random split give an overly optimistic test
   error here? (Hint: :numref:`sec_generalization_basics` and IID data.)
2. **Baseline.** Predict the training-set average for everyone and report
   the **mean absolute error (MAE)** in minutes on the test set. This is the
   number to beat.
3. **Linear model from scratch.** Impute missing values (training means
   only!), standardize features, and train linear regression with your own
   `Module`, `DataModule`, and `Trainer` classes (the pattern from
   :numref:`sec_oo-design`) and minibatch SGD. Start with the numeric
   features. Plot training and validation loss per epoch.
4. **Feature engineering.** Add rush-hour (11--13, 18--20) and weekend
   indicators, then one-hot encode `restaurant_id`. Report test MAE after each
   step. Which restaurants are slowest to hand off orders?
5. **Robust loss.** The lost-courier orders have huge squared errors. Replace
   squared loss with **Huber** loss (:numref:`sec_linear_concise` exercises)
   and compare test MAE and the learned coefficients.
6. **Regularization when data is scarce.** Retrain using only the first week
   (about 300 orders) with all features. Show the gap between training and
   test error, then sweep the weight decay $\lambda$ on a log scale and pick it
   with a validation split taken *from the first week*. Explain what you
   observe (it may not be a big win; say why).
7. **An honest promise.** Using the residuals on the validation set, quote a
   range such as "28--36 min" that contains the actual time for about 80% of
   orders. Check the coverage on the test set. (Hint: the noise grows with
   distance; does one fixed width work for short and long trips?)

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
- The interval in milestone 7 covers 75--85% of test orders.
- Loss curves for at least one model, and the train/test gap for the
  one-week model.

## Stretch Goals

- Make the interval width depend on distance (model the noise level), and
  compare coverage for short vs. long trips.
- Compare your SGD solution with the closed-form least-squares solution (the "Analytic Solution" in
  :numref:`sec_linear_regression`). Do they agree?
- Predict `log(delivery_min)` instead and compare MAE. When is that better?

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
