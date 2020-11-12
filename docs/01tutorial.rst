Tutorial 1: Train your first model with NNFabrik
++++++++++++++++++++++++++++++++++

In this tutorial, we will go through one example usage of the nnfabrik pipeline.

The goal in this tutorial is to train multiple models with different hyperparameters
(batchsize, number of layer, learning rate, etc.) in a parallel fashion,
and to save the corresponding results in a centralized database table.

We have already defined our three functions dataset-function, model-function and trainer-function.
Those are the basic functions which, if used together, train a simple MLP model on the well-known MNIST dataset.
Therefore, what we are left with is to fill up the corresponding Datajoint tables with different hyper-parameter
values and train models for every possible combination of those hyper-parameters.

The Fabrikant Table
---------------------------------------
Fabrikant tables keeps a record of the users that interact with a specific schema.
It is simply an extra level of information to know who is accountable for the entries
in the Dataset, Model, and Trainer tables. simply add your information as follows: ::

    fabrikant_info = dict(fabrikant_name="Your Name", email="your@email.com", affiliation="thelab", dj_username="yourname")
    Fabrikant().insert1(fabrikant_info)



The Dataset Table
---------------------------------------
Here we need to specify **dataset function** and the arguments passed to the dataset function, **dataset config**.
The dataset function is specified as a string. the structure of this string is important since under the hood nnfabrik performs a dynamic import by parsing this string. For example, if you can import the function as
:code:`from nnfabrik.datasets import toy_dataset_fn`
then you should specify the dataset function as:
:code:`"nnfabrik.datasets.toy_dataset_fn"`
::

    # specify dataset function as string (the function must be importable) as well as the dataset config
    dataset_fn = "nnfabrik.examples.mnist.dataset.mnist_dataset_fn"
    dataset_config = dict(batch_size=64) # we specify all the inputs except the ones required by nnfabrik

    Dataset().add_entry(dataset_fn=dataset_fn, dataset_config=dataset_config,
                        dataset_fabrikant="Your Name", dataset_comment="A comment about the dataset!");

.. tip::

    Since nnfabrik would need to import the function, the dataset function must be importable.
    Also note that :code:`dataset_config` is a dictionary that contains all the arguments that are **not** required by nnfabrik.

The Model Table
---------------------------------------

Here we need to specify **dataset function** and the arguments passed to the dataset function, **dataset config**.
Everything explained for the dataset function applied to model function as well.::

    # specify model function as string (the function must be importable) as well as the model config
    model_fn = "nnfabrik.examples.mnist.model.mnist_model_fn"
    model_config = dict(h_dim=5) # we specify all the inputs except the ones required by nnfabrik

    Model().add_entry(model_fn=model_fn, model_config=model_config,
                      model_fabrikant="Your Name", model_comment="A comment about the model!");

Let's also try :code:`h_dim = 15`: ::

    model_config = dict(h_dim=15) # we specify all the inputs except the ones required by nnfabrik
    Model().add_entry(model_fn=model_fn, model_config=model_config,
                      model_fabrikant="Your Name", model_comment="A comment about the model!");


The Trainer Table
---------------------------------------
Here we need to specify **trainer function** and the arguments passed to the trainer function, **trainer config**.
Everything explained for the dataset function applied to trainer function as well. ::

    # specify trainer function as string (the function must be importable) as well as the trainer config
    trainer_fn = "nnfabrik.examples.mnist.trainer.mnist_trainer_fn"
    trainer_config = dict(epochs=1) # we specify all the inputs except the ones required by nnfabrik

    Trainer().add_entry(trainer_fn=trainer_fn, trainer_config=trainer_config,
                      trainer_fabrikant="Your Name", trainer_comment="A comment about the trainer!");

The Seed Table
---------------------------------------
Now we have one final table to fill up before we start training our models with all the combinations in Dataset, Model,
and Trainer tables. That table is the **Seed** table. ::

    Seed().insert1({'seed': 2020})



The TrainedModel Table
---------------------------------------
Once we have bunch of trained models, the downstream analysis might be different for each specific project.
For this reason, we keep the TrainedModel tables separate from the tables provided in the library.
However, the process of creating your own TrainedModel has becomes very easy with the template(s) provided by nnfabrik.

Create your TrainedModel table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inheritance of the :code:`TrainedModelBase` template ::

    from nnfabrik.templates.trained_model import TrainedModelBase
    from nnfabrik.examples import nnfabrik

    @schema
    class TrainedModel(TrainedModelBase):
        table_comment = "Trained models"
        nnfabrik = nnfabrik

Populate (fill up) the TrainedModel table
~~~~~~~~~~~~

Calling :code:`populate` on this table fills all combinations of :code:`Trainer`, :code:`Dataset`, :code:`Model` and
:code:`Seed` (unless we restrict it) ::

    TrainedModel.populate(display_progress=True)
