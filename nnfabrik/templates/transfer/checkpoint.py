import copy

import datajoint as dj

from nnfabrik.main import Trainer
from nnfabrik.templates import TransferredTrainedModelBase
from nnfabrik.templates.checkpoint import TrainedModelChkptBase
from nnfabrik.utility.dj_helpers import CustomSchema, clone_conn

conn_clone = clone_conn(dj.conn())
schema_clone = CustomSchema(
    dj.config.get("schema_name", "nnfabrik_core"), connection=conn_clone
)


@schema_clone
class TransferredCheckpoint(dj.Manual):
    storage = "minio"

    @property
    def definition(self):
        definition = """
        # Checkpoint table
        transfer_step:                     int  # transfer         
        collapsed_history:                 varchar(64)  # transfer         
        data_transfer:                     tinyint
        -> Trainer
        -> Trainer.proj(prev_trainer_fn='trainer_fn', prev_trainer_hash='trainer_hash')
        -> Dataset
        -> Dataset.proj(prev_dataset_fn='dataset_fn', prev_dataset_hash='dataset_hash')
        -> Model
        -> Model.proj(prev_model_fn='model_fn', prev_model_hash='model_hash')
        -> Seed
        epoch:                             int          # epoch of creation
        ---
        score:                             float        # current score at epoch
        state:                             attach@{storage}  # current state
        ->[nullable] Fabrikant
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(
            storage=self.storage
        )
        return definition


class TransferredTrainedModelChkptBase(
    TransferredTrainedModelBase, TrainedModelChkptBase
):
    checkpoint_table = TransferredCheckpoint
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
        trainer_config = (Trainer & orig_key).fetch1("trainer_config")
        if not trainer_config.get("keep_checkpoints"):
            safe_mode = dj.config["safemode"]
            dj.config["safemode"] = False
            (self.checkpoint_table & orig_key).delete(verbose=False)
            print("Deleting intermediate checkpoints...")
            dj.config["safemode"] = safe_mode
