# Project: Hush
:label:`sec_project_alert_triage`

**Uses:** everything up to this point: data preprocessing with pandas
(:numref:`sec_pandas`), tensors and broadcasting (:numref:`sec_ndarray`),
norms (:numref:`sec_linear-algebra`), derivatives and gradients
(:numref:`sec_calculus`), and Bayes' theorem and expectations
(:numref:`sec_prob`). Nothing beyond Chapter 2.

*Smarter on-call alerts: wake engineers up only when it matters.*

**Time:** one or two evenings. **Build it in your own GitHub repository.** Name the repository `hush`.

## The Problem

Every company that runs a web service has engineers *on call*. Monitoring
tools watch metrics such as CPU, latency, and error rate, and *page* (wake
up) an engineer when something looks wrong. Too few alerts and real outages
go unnoticed; too many and engineers drown in false alarms ("alert fatigue")
and start ignoring pages. Monitoring and on-call companies (Datadog,
PagerDuty, Grafana) and every large infrastructure team work on exactly this
trade-off.

You'll build **Hush**, a small, explainable alert-triage system: score how unusual
each time window is, quantify why most alerts are false alarms, and choose
the alert threshold that minimizes the real cost of paging.

## Data

[Download metrics.csv](../_static/projects/metrics.csv) (2,000 rows,
simulated service telemetry). Each row is a 5-minute window:

| Column | Meaning |
|:--|:--|
| `cpu` | CPU usage (%) |
| `latency_ms` | median request latency (ms) |
| `error_rate` | percent of requests that failed |
| `region` | data center: `us-east`, `eu-west`, `ap-south` |
| `incident` | 1 if a real incident happened (the label) |

Like real telemetry it is messy: about 4% of values are missing (`NA`),
some rows have no region, incidents are rare (about 3%), and there are
harmless traffic spikes that look alarming but aren't incidents.

## Milestones

1. **Clean the data.** Load it with pandas. Impute missing numeric values
   with the column mean, and one-hot encode `region` with a separate column
   for missing regions. Report the fraction of missing values per column and
   the incident base rate.
2. **Score each window.** Standardize each metric with broadcasting,
   $z_{ij} = (x_{ij} - \mu_j)/\sigma_j$, and score each window by the
   $\ell_2$ norm of its $z$-vector. Explain why standardizing matters: what
   happens to raw-distance scores if latency is measured in seconds instead
   of milliseconds?
3. **Measure the false-alarm problem.** For a threshold $t$, compute alerts,
   precision $P(\textrm{incident} \mid \textrm{alert})$, recall, and missed
   incidents. Then, with Bayes' theorem, show analytically why a detector
   with 90% recall and a 5% false-positive rate still produces mostly false
   pages at a 3% base rate (the same effect as the HIV test in
   :numref:`subsec_probability_hiv_app`).
4. **Choose the threshold by cost.** With `page_cost` per page and
   `miss_cost` per missed incident, find the threshold minimizing
   `page_cost * alerts + miss_cost * missed` by searching over candidate
   thresholds. Plot total cost against threshold.
5. **Compare against a naive rule.** Compare your threshold with "page on
   the top 10% of scores". Report pages, precision, recall, and cost for both.
6. **Combine detectors.** Build a second, simpler detector (e.g., only
   `error_rate` above its 95th percentile). Using conditional independence
   and Bayes' theorem, estimate the precision when *both* detectors must
   fire, then measure it on the data. When does the independence assumption
   break down?

## Acceptance Criteria

- The cleaned feature table has no missing values and one row per window.
- Scores are unchanged (up to floating point) if a metric's units change.
- With `page_cost = 1` and `miss_cost = 10`, your threshold should catch
  about 47 of the 52 incidents with about 115 pages, versus 200 pages for
  the naive top-10% rule (exact numbers depend on details such as how you
  impute).
- Your Bayes calculation in milestone 3 gives about 36% precision.
- A plot of cost vs. threshold with the chosen threshold marked.

## Stretch Goals

- Replace the $\ell_2$ score with the *Mahalanobis distance*, which uses the
  covariance matrix from :numref:`sec_prob` to account for correlated
  metrics (e.g., CPU and latency rise together). Does it separate incidents
  from traffic spikes better?
- Make the threshold *per region*.
- Write unit tests for your cleaning and scoring functions.

## For Your README and Resume

Suggested repository layout: `hush/` (cleaning, scoring, evaluation
modules), `notebooks/analysis.ipynb` (plots and results), `tests/`, and a
README with the problem, approach, results table, and cost plot.

> *Built **Hush**, an explainable alert-triage system for service telemetry
> (pandas, NumPy): standardized anomaly scoring, Bayesian analysis of alert
> precision, and cost-optimal thresholding that cut on-call pages by about
> 40% versus a percentile baseline at comparable recall.*

Use your own measured numbers in the bullet.
