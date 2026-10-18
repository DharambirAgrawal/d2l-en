```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Data Preprocessing
:label:`sec_pandas`

So far our data arrived in ready-made tensors. In the wild, we must extract
messy data stored in arbitrary formats and preprocess it. The *pandas*
[library](https://pandas.pydata.org/) does much of the heavy lifting. This
section is no substitute for a proper *pandas*
[tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html),
but gives you a crash course in the most common routines.

## Reading the Dataset

Comma-separated values (CSV) files are everywhere for tabular
(spreadsheet-like) data. Each line is a record made of several
comma-separated fields, e.g.,
"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics".
To demonstrate loading CSV files with `pandas`, we create the file
`../data/house_tiny.csv`, a dataset of homes in which each row is a home and
the columns are the number of rooms (`NumRooms`), the roof type
(`RoofType`), and the price (`Price`).

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Now let's import `pandas` and load the dataset with `read_csv`.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Data Preparation

In supervised learning, we train models to predict a *target* value from
some *input* values, so first we separate the input columns from the target
column. We can select columns by name or by integer position (`iloc`).

Notice that `pandas` replaced every `NA` entry with a special `NaN`
(*not a number*) value; the same happens for empty entries such as
"3,,,270000". These *missing values* are the "bed bugs" of data science, a
persistent menace throughout your career. They are handled by *imputation*,
which replaces them with estimates, or *deletion*, which discards the rows or
columns that contain them.

Some common imputation heuristics: for categorical input fields, treat `NaN`
as a category. `RoofType` takes the values `Slate` and `NaN`, so `pandas` can
convert it into two columns, `RoofType_Slate` and `RoofType_nan`. A row with
roof type `Slate` gets 1 and 0 in these columns; a row with a missing
`RoofType` gets the reverse.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

For missing numerical values, a common heuristic is to replace `NaN` with the
mean of the column.

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Conversion to the Tensor Format

Now that all entries in `inputs` and `targets` are numerical, we can load them
into a tensor (recall :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Discussion

You now know how to partition data columns, impute missing values, and load
`pandas` data into tensors; :numref:`sec_kaggle_house` adds more data
processing skills. Real data processing gets hairy. Data may be spread across
several files from a relational database, e.g., customer addresses in one
table and purchases in another. Beyond categorical and numerical data, there
are text strings, images, audio, and point clouds, and advanced tools and
efficient algorithms are often needed to keep data processing from becoming
the bottleneck of the machine learning pipeline (we'll see this in computer
vision and natural language processing). And data quality matters:
real-world datasets suffer from outliers, faulty sensor measurements, and
recording errors that must be fixed before any modeling. Visualization tools
such as [seaborn](https://seaborn.pydata.org/), [Bokeh](https://docs.bokeh.org/),
or [matplotlib](https://matplotlib.org/) help you inspect the data and develop
intuitions about the problems you need to address.


## Exercises

1. Try loading datasets, e.g., Abalone from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?
1. Try indexing and selecting data columns by name rather than by column number. The pandas documentation on [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) has further details on how to do this.
1. How large a dataset do you think you could load this way? What might be the limitations? Hint: consider the time to read the data, representation, processing, and memory footprint. Try this out on your laptop. What happens if you try it out on a server? 
1. How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?
1. What alternatives to pandas can you think of? How about [loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Check out [Pillow](https://python-pillow.org/), the Python Imaging Library.
