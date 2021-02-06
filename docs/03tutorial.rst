Tutorial 3: Do transfer learning with NNFabrik
++++++++++++++++++++++++++++++++++

Replace the :code:`TrainedModel`-table with :code:`TransferredTrainedModel`: ::

    @schema
    class TransferredTrainedModel(TransferredTrainedModelBase):
        table_comment = "Transferred trained models"

If we have filled the tables in the same way we did in tutorial 1, we can already run the first stage of training
in the same way we did before. ::

    TransferredTrainedModel.populate()

Afterwards, we can take a look at the resulting table ( :code:`TransferredTrainedModel()` ) and we notice that it
has some additional columns that will be used to keep track of our transfer history.

Now that we have trained a model on MNIST, we want to transfer it.
For this, we consider two different scenarios, both of which can be fully automated in our transfer framework.

In both cases, the operations boil down to:

* what datasets and/or models and/or trainers you would like to use at a specific transfer step

    * specify this in the recipe

* which component should be handed over between two consecutive transfer steps (data and/or model state)

    * specify this in the recipe and use an appropriate :code:`trainer`

* which already-exisiting entries you would like this transfer step to be applied on

    * specify this in the :code:`populate` restrictions


Model-state Transfer
----------------------------------

First, let's add the dataset that we want to use for the transfer. ::

    dataset_fn = "nnfabrik.examples.mnist.dataset.mnist_dataset_fn"

    dataset_config = dict(batch_size=64, apply_augmentation=True) # we specify all the inputs except the ones required by nnfabrik

    Dataset().add_entry(dataset_fn=dataset_fn, dataset_config=dataset_config,
                        dataset_fabrikant="Your Name", dataset_comment="Augmented MNIST")

Now, we look up the identifiers of our dataset (look up in :code:`Dataset()`) and define the recipe. ::

    transfer_from = {"dataset_fn": 'nnfabrik.examples.mnist.dataset.mnist_dataset_fn', "dataset_hash": '9aee736870714f8b7c3cc084087ce886'}
    transfer_to = {"dataset_fn": 'nnfabrik.examples.mnist.dataset.mnist_dataset_fn', "dataset_hash": '28aefc2308569727c6017c66c9122d77'}
    DatasetTransferRecipe().add_entry(transfer_from=transfer_from, transfer_to=transfer_to, transfer_step=1)

To use this recipe, we need to register it with the :code:`TransferredTrainedModel`-table. ::

    TransferredTrainedModel.transfer_recipe = [DatasetTransferRecipe()]

If we call :code:`TransferredTrainedModel.populate()` now, it will automatically apply this recipe and transfer
the model (i.e. its :code:`state_dict`) we trained above to our target setting.
That means it will train on the new dataset starting from the model state of our first training.

Data Transfer (Knowledge Distillation)
--------------------------------

Now that we have seen a simple transfer of the model state between two training runs on similar datasets,
let's consider a more challenging scenario.
Assume we want to transfer knowledge between two slightly different models (e.g. with different hidden size)
and at the same time have a domain shift in the data.

The intermediate transfer step (dataset generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this step, we keep everything the same, but replace the trainer by a "pseudo-trainer" that
simply captures and saves the logits for the whole dataset.

.. literalinclude:: ../nnfabrik/examples/mnist_transfer/trainer.py
   :lines: 12-36

Of course, we also need to insert this one into our tables::

    trainer_fn = "nnfabrik.examples.mnist_transfer.trainer.mnist_data_gen_fn"

    trainer_config = dict(batch_size=64, apply_augmentation=False)

    Trainer().add_entry(trainer_fn=trainer_fn, trainer_config=trainer_config,
                        trainer_fabrikant="Your Name", trainer_comment="Transfer MNIST Logits");

Again, we need to look up the identifiers and create the corresponding recipe.::

    transfer_from = {"trainer_fn": 'nnfabrik.examples.mnist.trainer.mnist_trainer_fn', "trainer_hash": '79e921430b7f44a205d79d0087b59dc0'}
    transfer_to = {"trainer_fn": 'nnfabrik.examples.mnist_transfer.trainer.mnist_data_gen_fn', "trainer_hash": 'ab91f734757071bf0b98ab74c6e8583c'}
    TrainerTransferRecipe().add_entry(transfer_from=transfer_from, transfer_to=transfer_to, transfer_step=1, data_transfer=True)
    TransferredTrainedModel.transfer_recipe = [TrainerTransferRecipe()]


Final transfer step (using the generated dataset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we need a dataset that loads the transferred dataset and a trainer that can handle it. Let's introduce those here.


.. literalinclude:: ../nnfabrik/examples/mnist_transfer/dataset.py
   :lines: 8-23

.. literalinclude:: ../nnfabrik/examples/mnist_transfer/trainer.py
   :lines: 37-66
   :emphasize-lines: 23,24,10


Of course, we need to add those to the tables as well.::

    trainer_fn = "nnfabrik.examples.mnist_transfer.trainer.mnist_trainer_fn"
    trainer_config = dict(batch_size=64, apply_augmentation=False)
    Trainer().add_entry(trainer_fn=trainer_fn, trainer_config=trainer_config,
                        trainer_fabrikant="Your Name", trainer_comment="Use Transferred MNIST Logits")

    dataset_fn = "nnfabrik.examples.mnist_transfer.dataset.mnist_dataset_fn"
    dataset_config = dict(batch_size=64, apply_augmentation=True) # we specify all the inputs except the ones required by nnfabrik
    Dataset().add_entry(dataset_fn=dataset_fn, dataset_config=dataset_config,
                        dataset_fabrikant="Your Name", dataset_comment="Augmented MNIST with Knowledge Distallation")

    transfer_from = {"trainer_fn": 'nnfabrik.examples.mnist_transfer.trainer.mnist_data_gen_fn',
                     "trainer_hash": 'ab91f734757071bf0b98ab74c6e8583c',
                     "dataset_fn": 'nnfabrik.examples.mnist.dataset.mnist_dataset_fn',
                     "dataset_hash": '9aee736870714f8b7c3cc084087ce886'
                    }
    transfer_to = {"trainer_fn": 'nnfabrik.examples.mnist_transfer.trainer.mnist_trainer_fn',
                     "trainer_hash": 'ab91f734757071bf0b98ab74c6e8583c',
                     "dataset_fn": 'nnfabrik.examples.mnist_transfer.dataset.mnist_dataset_fn',
                     "dataset_hash": '28aefc2308569727c6017c66c9122d77'
                    }

    TrainerDatasetTransferRecipe().add_entry(transfer_from=transfer_from, transfer_to=transfer_to, transfer_step=2, data_transfer=False)
    TransferredTrainedModel.transfer_recipe = [TrainerDatasetTransferRecipe()]

