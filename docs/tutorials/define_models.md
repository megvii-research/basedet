# Define Models

BaseDet provides models for training and testing, however,  you might be willing to do define your own models and train them.  Therefore, BaseDet provides mechanisms that let users define their own models.

## Define by inheriting BaseNet class
BaseDet provides a basic BaseNet concept to define stanard model's behavior.
The following are some important methods to override.
* `preprocess_image`: defines how to preprocess input data before it becomes a tensor, usually normalize (minus mean and then divide by std).
* `network_forward`: defines how your model process input tensors. Logic inside this function should be unified no matter what status your model is worked with.
* `get_losses`: defines how a network calculates output losses by given  input data. This function should call `network_forward` implicitly.
* `inference`: define how a network inference output values by given input data . This function should call `network_forward` implicitly.

Note: `forward` is define as the following behavior in BaseNet 
```python3
def forward(self, inputs):
    if self.training:
        return self.get_losses(inputs)
    else:
        return self.inference(inputs)
```

##  Register models

1. write registered model python code.
users could use following code to register your own model.
```python3
from basedet.utils import registers
from basedet.models import BaseNet

@registers.models.register()
class MyModel(BaseNet):
	...
```

2. import network.
After register network in python code, users should make sure that registry works, a common way to do such things is import them. The following code is an example.
```python3
from mymodel import MyModel
```

3. change config value
Change config.MODEL.NAME to class name of self-defined model.
For example, users could use the following code to change values in config.
```python3
cfg.MODEL.NAME = "MyModel"
```
or
```python3
diff_dict = dict(
    MODEL=dict(NAME="MyModel"),
)
cfg.merge(diff_dict)
