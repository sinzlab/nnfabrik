Preparation
++++++++++++++++++++++++++++++++++



Dataset
---------------------------------
The dataset function

.. literalinclude:: ../nnfabrik/examples/mnist/dataset.py
   :lines: 8-42

Model
---------------------------------
We define a simple two layer neural network with flexible hidden size :code:`h_dim` and :code:`ReLU` non-linearity.

.. literalinclude:: ../nnfabrik/examples/mnist/model.py
   :lines: 7-18


Initializing this model with for a given config is then done in the :code:`mnist_model_fn`.

.. literalinclude:: ../nnfabrik/examples/mnist/model.py
   :lines: 21-38



Trainer
---------------------------------
Finally, we define the trainer, which gets the model and dataloaders to execute the actual training.

.. literalinclude:: ../nnfabrik/examples/mnist/trainer.py
   :lines: 10-52

The corresponding trainer function sets up the training, executes it and finally returns output and score.

.. literalinclude:: ../nnfabrik/examples/mnist/trainer.py
   :lines: 55-78
