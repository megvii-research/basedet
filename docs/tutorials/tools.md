# BaseDet tools
BaseDet provides tools to train, test and analyze models. We will introduce 3 basic command to users in the following part.

## basedet_train
`basedet_train` is a basic command for training models.

Some args used for training:
* `-f/--file`: given a config file to describe training process. defalut value is config.py.
* `-w/--weight-file`:  given file that contains weights, defalut value: None.
* `-n/--ngpus`: given total number of gpus for training. If not given, program will use all devices.
* `-d/--dataset_dir`: given a path where contains training data, defalut value: None.
*  `--resume`: resume training or not. If given, program will use lastest checkpoint to train. This parser is usually used to continue interrupted/crashed training process.
*  `--dtr`: use DTR or not. If given, MegEngine will use DTR to make user train a larger model which might cause GPU OOM if not enable DTR.
*  `--dtr-thresh`: given dtr threshold, might be deprecated in the future.
*  `--sync_level`: given sync level of MegEngine, this parser is often set to 0 to debug MegEngine error, otherwise please don't set it.
*  `--fastrun`: using fastrun or not. If fastrun is open, MegEngine will search a best execution method for model.
* opts: usually use to override config value, but using opts to override config is not recommanded. In most cases, please modify your config in your config.py file

## basedet_test
`basedet_test` is a basic command for testing models.

Some args used for testing(note that it's similar to `basedet_train`):
* `-f/--file`: given a config file to describe testing process. defalut value is config.py.
* `-w/--weight-file`:  given file that contains weights, defalut value: None.
* `-n/--ngpus`: given total number of gpus for training. If not given, program will use all
* `-d/--dataset_dir`: describes a path where contains training data, defalut value: None.

## basedet_analyze
`basedet_analyze` is a basic command for analyzing models.

Some  args used for analyzing:

* `-f/--file`: given a config file to describe . defalut value is config.py.
* `-w/--weight-file`:  given file that contains weights, defalut value: None.
* `-p/--profile`: profile model or not. If enable, a json file contains profile info will be dumped. User could load and check in web browser.
* `--height`: given height of input image.
* `--width`: given width of input image.
* `channels`: given channels of input image, default value: 3.
* opts: usually use to override config value, but using opts to override config is not recommanded. In most cases, please modify your config in your config.py file
