# Setup Instructions on a GPU Machine

**Important** These steps should all be performed on a GPU machine.

### 0. Clone the SALIENT++ artifact repo

Before you with installing the SALIENT++ artifact dependencies, please make sure
you have cloned this repo locally and changed into the cloned repo directory.
Steps 5 & 6 below assume that your current working directory is the SALIENT++
cloned repo directory.

```bash
git clone git@github.com:MITIBMxGraph/SALIENTplus_artifact.git
cd SALIENTplus_artifact
```

### 1. Install Conda

Follow instructions on the [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, to install Miniconda on an x86 Linux machine:

```bash
curl -Ls https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VER}-Linux-${conda_arch}.sh -o /tmp/Miniconda.sh &&\
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```

SALIENT++ has been tested on Python 3.9.5.

**It is highly recommended to create a new environment and do the subsequent
steps therein.**  Otherwise, it is possible that you will run into unexpected
issues with the installation.

For example, to create a new environment called `salientplus`:

```bash
conda create -n salientplus python=3.9.5 -y
conda activate salientplus
```

### 2. Install PyTorch

Follow instructions on the [PyTorch homepage](https://pytorch.org). For example, to install on a linux machine with CUDA 11.7:

```bash
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

SALIENT++ should be compatible with newer versions of PyTorch (e.g., PyTorch 2), but we have not tested it. SALIENT++ was developed against PyTorch 1.10 and has been tested up to PyTorch 1.13.1.

### 3. Install OGB

```bash
conda install -y -c conda-forge ogb
```

SALIENT++ has been tested on OGB 1.3.5.

### 4. Install PyTorch-Geometric (PyG) and PyTorch Sparse

To get the latest version of PyG, follow the instructions on the [PyG Github page](https://github.com/pyg-team/pytorch_geometric).

```bash
conda install -y pyg -c pyg -c conda-forge
conda install -y pytorch-sparse -c pyg
```

SALIENT++ has been tested on PyTorch Geometric version 2.2.0.

### 5. Install SALIENT++'s fast_sampler

**Important** As with prior steps, you must install the `fast_sampler` module on a GPU machine.

Build and install the local `fast_sampler` module:

```bash
cd fast_sampler
python setup.py install
cd ..
```

To check that `fast_sampler` is properly installed, start python and run:

```python
>>> import torch
>>> import fast_sampler
>>> help(fast_sampler)
```

You should see information of the package.

> **NOTE:** Compilation requires a C++ compiler that supports C++17 (e.g., gcc >= 7).

### 6. Install METIS (optional, if using pre-generated partitions)

The following dependencies are required only if you plan to use our scripts for
partitioning graph datasets (e.g., `partitioners/run_4constraint_partition.py`).

We recommend that you install METIS from source using our provided repository.
We require that METIS is built with 64-bit types, which precludes the use of
commonly distributed METIS libraries in existing packages.

```bash
git clone git@github.com:MITIBMxGraph/METIS-GKlib.git
cd METIS-GKlib
make config shared=1 cc=gcc prefix=$(realpath ../pkgs) i64=1 r64=1 gklib_path=GKlib/
make install
```

Next, install `torch-metis`, which provides a Python module named `torch_metis` with
METIS bindings that accept PyTorch tensors.

```bash
cd ..
git clone git@github.com:MITIBMxGraph/torch-metis.git
cd torch-metis
python setup.py install
cd ..
```

> **NOTE:** The `torch_metis` module normally requires some configuration of
> environment variables in order to function properly.  In our attempt to
> streamline the process of exercising the SALIENT++ artifact, we made sure
> that the relevant scripts that use METIS set these variables internally.
> Please be advised that, outside of these artifact scripts, `import torch_metis`
> will _not_ work without setting the necessary environment variables.

### 7. Install other dependencies

Install the following extra dependencies.

```bash
conda install -y -c conda-forge nvtx
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge prettytable
```

### 8. Post-installation: activating the SALIENT++ environment

To re-enable the SALIENT++ environment and exercise the artifact in a new
terminal session, simply activate the `salientplus` conda environment:

```bash
conda activate salientplus
```
