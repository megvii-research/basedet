# Dataloader

## Overview of dataloader builder
Dataloader in BaseDet is build through registry. For users, BaseDet provides register method to train custom data.

Here is an example code of Dataloader builder in BaseDet(code could also be found in basedet.data.build):

```python3
@registers.dataloader.register()
class DataloaderBuilder:

    @classmethod
    def build(cls, cfg):
        dataset = build_dataset(cfg)
        transform = build_transform(cfg)
        sampler = cls.build_sampler(dataset, cfg)
        collator = cls.build_collator(cfg)
    
        train_dataloader = DataLoader(
            dataset,
            sampler=sampler,
            transform=transform,
            collator=collator,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
        return train_dataloader
    
    @classmethod
    def build_sampler(cls, dataset, cfg):
        batch_size = cfg.MODEL.BATCHSIZE
        sampler = AspectRatioGroupSampler(dataset, batch_size)
        if cfg.DATA.ENABLE_INFINITE_SAMPLER:
            sampler = Infinite(sampler)
        return sampler
    
    @classmethod
    def build_collator(cls, cfg):
        return DetectionPadCollator()
```


From the code, we could notice that a dataloader is actually composed by four parts:
* Dataset, control what data is provided to users.
* Transfrom, control what kind of data augmentation is applied on data from dataset.
* Sampler, decide how data is sampled from dataset. 
* Collator, decide how single data becomes batched data. For example, detection task often applies image padding in collator.

## Use custom sampler/collator
For users who just simply want to change sampler and collator, just override `build_sampler` and `build_collator` and register dataloader.

## Register dataloader

1. write registered dataloader builder python code. 
For a self-defined dataloader, a `build(cls, cfg)` function is always required, other function in the above example like `build_sampler`, `build_collator` might not matter.
If developers guarantee returned dataloader could get data through `next()` method in python(which means implements `__next__` method in class), self-defined dataloader builder could be registered. 
users could use following code to register your own dataloader builder.
```python3
from basedet.utils import registers

@registers.dataloader.register()
class SelfDefinedBuilder:

	def build(cls, cfg):
		# your build logic here.
```

2. import builder. 
After register dataloader in python code, users should make sure that registry works, a common way to do such things is import them. The following code is an example.
```python3
from new_dataloader import SelfDefineBuilder
```

3. change config value
Change config.DATA.BUILDER_NAME to class name of self-defined builder.
For example, users could use the following code to change values in config.
```python3
cfg.DATA.BUILDER_NAME = "SelfDefinedBuilder"
```
or 
```python3
diff_dict = dict(
	DATA=dict(BUILDER_NAME="SelfDefinedBuilder"),
)
cfg.merge(diff_dict)
```
