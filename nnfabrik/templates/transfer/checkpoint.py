import copy

import datajoint as dj

from nnfabrik.main import Trainer
from nnfabrik.templates import TransferredTrainedModelBase
from nnfabrik.templates.checkpoint import TrainedModelChkptBase
from nnfabrik.utility.dj_helpers import CustomSchema, clone_conn


def my_checkpoint(nnfabrik):
    conn_clone = clone_conn(dj.conn())
    schema_clone = CustomSchema(nnfabrik.schema.database, connection=conn_clone)

    @schema_clone
    class TransferredCheckpoint(dj.Manual):
        storage = "minio"

        @property
        def definition(self):
            definition = """
            # Checkpoint table
            -> nnfabrik.Trainer
            -> nnfabrik.Dataset
            -> nnfabrik.Model
            -> nnfabrik.Seed
            prev_trainer_fn:                   varchar(64)
            prev_trainer_hash:                 varchar(64)
            prev_dataset_fn:                   varchar(64)
            prev_dataset_hash:                 varchar(64)
            prev_model_fn:                     varchar(64)
            prev_model_hash:                   varchar(64)
            collapsed_history:                 varchar(64)  # transfer         
            transfer_step:                     int  # transfer         
            data_transfer:                     tinyint
            epoch:                             int          # epoch of creation
            ---
            score:                             float        # current score at epoch
            state:                             attach@{storage}  # current state
            ->[nullable] nnfabrik.Fabrikant
            trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(
                storage=self.storage
            )
            return definition

    return TransferredCheckpoint


class TransferredTrainedModelChkptBase(
    TransferredTrainedModelBase, TrainedModelChkptBase
):
    checkpoint_table = None  # TransferredCheckpoint
    keys = [
        "model_fn",
        "model_hash",
        "dataset_fn",
        "dataset_hash",
        "trainer_fn",
        "trainer_hash",
        "collapsed_history",
        "prev_model_fn",
        "prev_model_hash",
        "prev_dataset_fn",
        "prev_dataset_hash",
        "prev_trainer_fn",
        "prev_trainer_hash",
    ]

    def make(self, key):
        orig_key = copy.deepcopy(key)
        TransferredTrainedModelBase.make(self, key)
        # Clean up checkpoints after training:
        trainer_config = (self.trainer_table & orig_key).fetch1("trainer_config")
        if not trainer_config.get("keep_checkpoints"):
            safe_mode = dj.config["safemode"]
            dj.config["safemode"] = False
            (self.checkpoint_table & orig_key).delete(verbose=False)
            print("Deleting intermediate checkpoints...")
            dj.config["safemode"] = safe_mode
