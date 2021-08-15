# tiny-deep

API Structure

```
from model import *

neural_net = Model([[1,4], [2, 5], [3, 6]], [[1],[2]]).add_layer({'hidden_unit': 4, 'activation': "Relu"}).add_layer({'hidden_unit': 1, 'activation': "Sigmoid"}).train()
```
