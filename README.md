# BaseDet

English | [中文](README_CN.md)

BaseDet is a fast and easy-to-use detection toolbox. Learn more at our documentation.

## Features
* This repo is  powered by the [MegEngine](https://github.com/MegEngine/MegEngine) deep learning framework.
* It provides serveral classic SOTA models and related components, which could be used as basic libraray.

## Installation

See [INSTALL.md](INSTALL.md).

## Getting Started

1. Make sure that BaseDet is installed in a right way.

    You could run the following command to check.
    ```shell
    python3 -c "import basedet; print(basedet.__version__)"
    ```
2. Prepare dataset
3. Train model

    BaseDet provides a simple command `det_train` to train model. Just simply run the follwing command to train network (config is used to describe the training process).
    ```shell
    basedet_train -f config.py
    ```
4. Test model

    Like training, simply use the fellowing command.
    ```shell
    basedet_test -f config.py
    ```

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Model Zoo](MODEL_ZOO.md).


## License

BaseDet  is released under the [Apache 2.0 license](LICENSE).
