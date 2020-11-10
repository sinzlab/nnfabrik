import copy
from typing import Dict, Tuple

import datajoint as dj
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from nnfabrik.builder import get_all_parts, get_model, get_trainer
from nnfabrik.templates.trained_model import TrainedModelBase
from nnfabrik.utility.dj_helpers import make_hash, clone_conn, CustomSchema
from nnfabrik.builder import resolve_data
from datajoint.fetch import DataJointError
from nnfabrik.main import schema


conn_clone = clone_conn(dj.conn())
schema_clone = CustomSchema(
    dj.config.get("schema_name", "nnfabrik_core"), connection=conn_clone
)


@schema_clone
class Checkpoint(dj.Manual):
    storage = "minio"

    @property
    def definition(self):
        definition = """
        # Checkpoint table
        -> Trainer
        -> Dataset
        -> Model
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


class TrainedModelChkptBase(TrainedModelBase):
    checkpoint_table = Checkpoint
    keys = [
        "model_fn",
        "model_hash",
        "dataset_fn",
        "dataset_hash",
        "trainer_fn",
        "trainer_hash",
    ]

    def call_back(self, uid=None, epoch=None, model=None, state=None):
        """
        This method is periodically called by the trainer and is used to save the training state in a remote table.
        Args:
            uid - Unique identifier for the trained model entry
            epoch - the iteration count
            model - current model under training
            state - Additional information provided by the trainer
                score: float = 0.0,
                maximize_score: bool = True,
                keep_last_n: int = 1,
                keep_best_n: int = 1,
                keep_selection: Tuple = (),
        """
        maximize_score = state.pop("maximize_score", True)
        if epoch >= 0:  # save current epoch
            assert "score" in state, "Score value needs to be provided"
            score = state.pop("score", 0.0)
            keep_best_n = state.pop("keep_best_n", 1)
            keep_last_n = state.pop("keep_last_n", 1)
            keep_selection = state.pop("keep_selection", ())

            # add to checkpoint table
            with tempfile.TemporaryDirectory() as temp_dir:
                key = copy.deepcopy(uid)
                for k in self.keys:
                    if k not in key:
                        key[k] = ""
                key["epoch"] = epoch
                key["score"] = score
                filename = make_hash(uid) + ".pth.tar"
                filepath = os.path.join(temp_dir, filename)
                state["net"] = model.state_dict()
                torch.save(
                    state, filepath,
                )
                key["state"] = filepath
                self.checkpoint_table.insert1(
                    key
                )  # this is NOT in transaction and thus immediately completes!

            # fetch all fitting entries from checkpoint table
            checkpoints = (self.checkpoint_table & uid).fetch(
                *self.keys, "seed", "score", "epoch", as_dict=True,
            )

            # select checkpoints to be kept
            keep_checkpoints = []
            best_checkpoints = sorted(
                checkpoints, key=lambda chkpt: chkpt["score"], reverse=maximize_score
            )
            for c in checkpoints:
                del c["score"]  # restricting with a float is not a good idea -> remove
            keep_checkpoints += best_checkpoints[:keep_best_n]  # w.r.t. performance
            last_checkpoints = sorted(
                checkpoints, key=lambda chkpt: chkpt["epoch"], reverse=True
            )
            keep_checkpoints += last_checkpoints[:keep_last_n]  # w.r.t. temporal order
            for chkpt in checkpoints:
                if chkpt["epoch"] in keep_selection:
                    keep_checkpoints.append(chkpt)  # keep selected epochs

            # delete the others
            safe_mode = dj.config["safemode"]
            dj.config["safemode"] = False
            ((self.checkpoint_table & uid) - keep_checkpoints).delete(verbose=False)
            dj.config["safemode"] = safe_mode

        else:  # restore existing epoch
            # retrieve all fitting entries from checkpoint table
            checkpoints = (self.checkpoint_table & uid).fetch(
                "score", "epoch", "state", as_dict=True,
            )
            if not checkpoints:
                return
            if epoch == -1:  # restore last epoch
                last_checkpoints = sorted(
                    checkpoints, key=lambda chkpt: chkpt["epoch"], reverse=False
                )
                checkpoint = last_checkpoints[-1]
            elif epoch == -2:  # restore best epoch
                best_checkpoints = sorted(
                    checkpoints,
                    key=lambda chkpt: chkpt["score"],
                    reverse=maximize_score,
                )
                checkpoint = best_checkpoints[0]

            # restore the training state
            state["epoch"] = checkpoint["epoch"]
            state["score"] = checkpoint["score"]
            loaded_state = torch.load(checkpoint["state"])
            for key, state_entry in loaded_state.items():
                if key in state and hasattr(state[key], "load_state_dict"):
                    state[key].load_state_dict(state_entry)
                else:
                    state[key] = state_entry

    def make(self, key):
        orig_key = copy.deepcopy(key)
        super().make(key)
        # Clean up checkpoints after training:
        trainer_config = (Trainer & orig_key).fetch1("trainer_config")
        if not trainer_config.get("keep_checkpoints"):
            safe_mode = dj.config["safemode"]
            dj.config["safemode"] = False
            (self.checkpoint_table & orig_key).delete(verbose=False)
            print("Deleting intermediate checkpoints...")
            dj.config["safemode"] = safe_mode
