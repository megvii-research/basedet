## Features
* BaseDet采用[MegEngine](https://github.com/MegEngine/MegEngine) 作为深度学习引擎。
* 提供了一些经典的检测SOTA模型以及相关组件。

## Installation

安装请参考[INSTALL.md](INSTALL.md).

## Getting Started

1. 确保BaseDet已经正确安装

    可以在Shell运行如下命令来确保BaseDet已经正确安装。
    ```shell
    python3 -c "import basedet; print(basedet.__version__)"
    ```
2. 准备数据集

    有两种方式准备数据集，当你已经有数据集在本地的某个位置上时，可以使用环境变量 `BASEDET_DATA_DIR`。 在此文件夹下，BaseDet需要的格式如下所示:
    ```
    $BASEDET_DATA_DIR/
    COCO/
    ```
    其中， [COCO detection](https://cocodataset.org/#download) 的数据集格式如下：
    ```
    COCO/
    annotations/
        instances_{train,val}2017.json
    {train,val}2017/
        # image files that are mentioned in the corresponding json
    ```

    可以通过export设置环境变量：
    ```shell
    export BASEDET_DATA_DIR=/path/to/your/datasets
    ```
    如果不想设置 `BASEDET_DATA_DIR` 环境变量，默认的数据读取地址就是basedet package下面的`datasets` 文件夹，需要用户自行创建：
    ```shell
    cd basedet
    mkdir datasets
    ```

3. 训练模型

    BaseDet提供了一个简单命令`basedet_train`来训练模型。只需要简单使用如下命令(config用来描述训练过程)。
    ```shell
    basedet_train -f config.py
    ```
4. 测试模型

    和training一样，使用`basedet_test`命令测试模型。
    ```shell
    basedet_test -f config.py
    ```
5. MegStudio（可选）

    如果你想要保姆式手把手教学，可以参考megstudio上的这个项目：[链接](https://studio.brainpp.com/project/28826?name=BaseDet%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)。进入项目后请不要使用draft，可以使用version1。

## Model Zoo and Baselines

BaseDet在[Model Zoo](MODEL_ZOO.md)提供了一些已经训练完成的模型和基本信息。

## License

BaseDet 使用[Apache 2.0 license](LICENSE)，请确保对BaseDet的使用满足License限制。
