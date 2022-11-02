# Installation

## Install
There are two ways to install basedet.
1. pip install

    This installation method is specified to support product model and released as basedet_snapdet and is recommanded for users to train product models.
    ```shell
    pip3 install basedet
    ```
2. source code install

This way of installation is recommanded for developers. Use the follwing commands to install BaseDet by source code.
* clone OpenSource version BaseDet repo and enter the directory
```shell
git clone https://github.com/megvii-research/basedet.git
cd BaseDet
```
* install requirments
```shell
python3 -m pip install -r requirements.txt
```
* For developers, you should install pre-commit to lint your code before git commit.
```
python3 -m pip install pre-commit
pre-commit install
```
* install BaseDet
```shell
python3 -m pip install -v -e .
```
or
```shell
python3 setup.py develop
```

**PS: We highly recommand developers to use virtual environment management tools to manage your python environment.**

## Check installation

Run the following command to make sure BaseDet is installed in a correct way.
```shell
python3 -c "import basedet; print(basedet.__version__)"
```
