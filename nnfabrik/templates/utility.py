import datajoint as dj
from .main import Model, Dataset, Trainer, Seed, Fabrikant
from .builder import resolve_data


class DataInfoBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    DataInfo table. By default, this will inherit from the Dataset as found in nnfabrik.main.
    To change this behavior, overwrite the dataset_table` property.
    """

    dataset_table = Dataset
    user_table = Fabrikant

    # table level comment
    table_comment = "Table containing information about i/o dimensions and statistics, per data_key in dataset"

    @property
    def definition(self):
        definition = """
            # {table_comment}
            -> self.dataset_table
            ---
            data_info:                     longblob     # Dictionary of data_keys and i/o information

            ->[nullable] self.user_table
            datainfo_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(table_comment=self.table_comment)
        return definition

    def make(self, key):
        """
        Given a dataset from nnfabrik, extracts the necessary information for building a model in nnfabrik.
        'data_info' is expected to be a dictionary of dictionaries, similar to the dataloaders object.
        For example:
            data_info = {
                        'data_key_0': dict(input_dimensions=[N,c,h,w, ...],
                                           input_channels=[c],
                                           output_dimension=[o, ...],
                                           img_mean=mean_train_images,
                                           img_std=std_train_images),
                        'data_key_1':  ...
                        }
        """

        dataset_fn = resolve_data(key["dataset_fn"])
        dataset_config = (self.dataset_table & key).fetch1("dataset_config")
        data_info = dataset_fn(**dataset_config, return_data_info=True)

        fabrikant_name = self.user_table.get_current_user()

        key['fabrikant_name'] = fabrikant_name
        key['data_info'] = data_info
        self.insert1(key)