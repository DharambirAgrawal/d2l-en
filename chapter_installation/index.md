# Installation
:label:`chap_installation`

To run the book's code, you need Python, Jupyter Notebook, the relevant
libraries, and the book's notebooks.

The interactive exercises on this site run Python directly in your browser,
so you don't need any of this to do them. Set up a local environment when you
want to run the book's PyTorch code yourself.

## Installing Miniconda

The simplest option is [Miniconda](https://conda.io/en/latest/miniconda.html)
(Python 3.x is required; skip this if you already have conda). Download the
version for your system and architecture. For Python 3.9 (our tested
version) on an Intel Mac, download the script whose name contains "MacOSX"
and run it from the download location:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


On Linux, download the file whose name contains "Linux" and run:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


On Windows, follow the [online instructions](https://conda.io/en/latest/miniconda.html),
and use the Command Prompt (search for `cmd`) to run commands.

Next, initialize the shell so you can run `conda` directly:

```bash
~/miniconda3/bin/conda init
```


Close and reopen your shell, then create a new environment:

```bash
conda create --name d2l python=3.9 -y
```


and activate it:

```bash
conda activate d2l
```


## Installing the Deep Learning Framework and the `d2l` Package

First check whether your machine has a suitable GPU (the display GPU of a
standard laptop doesn't count). If you have NVIDIA GPUs with
[CUDA](https://developer.nvidia.com/cuda-downloads) installed, you're all set.
If not, don't worry: a CPU is plenty for the first few chapters. You'll want
GPU access before training larger models.


:begin_tab:`pytorch`

Install PyTorch (these versions were tested at the time of writing) with CPU or GPU support:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
You can install TensorFlow with either CPU or GPU support as follows:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
You can install JAX and Flax with either CPU or GPU support as follows:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


Without NVIDIA GPUs or CUDA, install the CPU version:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


Then install the `d2l` package, which bundles the functions and classes used
throughout the book:

```bash
pip install d2l==1.0.3
```


## Downloading and Running the Code

Download the notebooks so you can run each code block: click "Notebooks" at
the top of any page on [the D2L.ai website](https://d2l.ai/) and unzip the
file, or fetch them from the command line:

:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

(If `unzip` is missing, run `sudo apt-get install unzip` first.) Start the
Jupyter Notebook server:

```bash
jupyter notebook
```


Open http://localhost:8888 in your browser (it may open automatically) and run
the code for each section. In every new terminal window, run
`conda activate d2l` before running the notebooks or updating packages, and
`conda deactivate` to leave the environment.
