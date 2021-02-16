Tutorial 2: Train a model with checkpointing
++++++++++++++++++++++++++++++++++

We introduce two new methods in our trainer. First a method to save the current training state

.. literalinclude:: ../nnfabrik/examples/mnist_checkpoint/trainer.py
   :lines: 27-39

Then a method to restore a the state if training is resumed after an interruption

.. literalinclude:: ../nnfabrik/examples/mnist_checkpoint/trainer.py
   :lines: 40-51

Both methods make use of the :code:`call_back` function that should be passed to the trainer from
the :code:`TrainedModel` table.
The important difference is the epoch count that is passed to :code:`call_back` here.
A count of :code:`-1` signals the function that we want to retrieve the last checkpoint if there is one.
A positive count on the other hand signals that we are in the process of training and want to save the
current state as a checkpoint.

Finally, we also have to update the training procedure itself to call :code:`self.restore()` before the training starts
and :code:`self.save()` after every epoch.

.. literalinclude:: ../nnfabrik/examples/mnist_checkpoint/trainer.py
   :lines: 53-69
   :emphasize-lines: 5,15

Once we have a trainer that supports the checkpointing feature, all we need to do is to switch from :code:`TrainedModel`
table to :code:`TrainedModelChkpt`. ::

    from nnfabrik.templates.checkpoint import TrainedModelChkptBase, my_checkpoint

    Checkpoint = my_checkpoint(nnfabrik)

    @nnfabrik.schema
    class TrainedModelChkpt(TrainedModelChkptBase):
        table_comment = "My Trained models with checkpointing"
        nnfabrik = nnfabrik
        checkpoint_table = Checkpoint

Now this table can be used just as :code:`TrainedModel`, i.e. we can simply populate it. ::

    TrainedModelChkpt.populate()

The training state will be saved automatically in the :code:`Checkpoint`-table after each epoch
and can be retrieved from there should the training be interrupted.
Thus, after an interruption, we can just clear the error state with: ::

    # delete all jobs in error state:
    (schema.jobs & "status='error'").delete()

And then we just restart the training, by calling :code:`populate()` again.
