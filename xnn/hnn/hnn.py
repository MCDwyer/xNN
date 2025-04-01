import numpy as np
import time

from ..cbnn.cbnn import CBNNModel
from . import hnn_utils as utils
from ..nearest_neighbour.leaf_list import NavigatingNetList, LeafList

DEFAULT_LEAF_LIST_TYPE = NavigatingNetList

class HNNModel(CBNNModel):
# TODO: try storing the bt_B etc. on this tree instead? Primarily the bt_B?
    def __init__(self, rho, binning_radius, T, leaf_list_type=DEFAULT_LEAF_LIST_TYPE):
        super().__init__(rho, binning_radius, T)
        self.H_sets = None
        # self.dataset = None
        self.trial_number = 0

    def train(self, dataset, set_info=False):

        self.set_dataset(dataset)

        # if self.dataset is None:
        #     self.set_dataset(dataset)
        # else:
        #     # check dataset is the right type (otherwise the action etc. is going to be diff and won't work??)
        #     assert isinstance(dataset, type(self.dataset))

        all_training_info = []

        H_sets = self.H_sets
        trial_number = self.trial_number

        if self.bt_B is None:
            bt_B = self.create_bt_B()
        else:
            bt_B = self.bt_B

        # if self.leaf_list is None:
        #     leaf_list = LeafList()
        # else:
            
        leaf_list = self.leaf_list

        if self.eta is None:
            self.K = len(self.actions)
            self.eta = self.calculate_learning_rate()

        bt_Z = self.bt_Z
        tt_H_Z = self.tt_H_Z

        start_time = time.time()
        last_time = start_time

        dataset_size = len(dataset.dataset)

        # print(f"Training starting on dataset with size: {len(dataset.dataset)} at time {int(start_time)}.")
        # print(f"rho: {self.rho}, eta: {self.eta}")

        for i, data in enumerate(dataset.dataset):
            x = data[:-1]
            label = data[-1]
            H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, action, loss = utils.hnn(self, H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, x, label, trial_number)
            # if set_info:
            #     info.append(H_sets.get_max_depth())

            if action is not None:
                all_training_info.append([action, loss])

            trial_number += 1

        # info_array = np.array(all_training_info)
        # avg_loss = np.mean((info_array[:, 0]).astype(float))

        # print(f"Training finished after {dataset_size} trials with average loss {avg_loss:.3f}.")
        # print(f"Training took: {int(time.time()) - start_time} seconds.")

        # cm = create_confusion_matrix([info_array], dataset.actions)
        # print(cm.astype(int))
        # if cm.sum(axis=1)[:, np.newaxis].all() != 0:
        #     print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

        # print("Max depth: ", H_sets.get_max_depth())
        self.H_sets = H_sets
        self.bt_B = bt_B
        self.bt_Z = bt_Z
        self.tt_H_Z = tt_H_Z
        self.leaf_list = leaf_list
        self.trial_number = trial_number

        return np.array(all_training_info)

    def predict(self, x):

        if self.H_sets is None:
            raise RuntimeError("Model not trained, run model.train(dataset) first.")


    def save_model(self, filepath):
        filepath += "_hnn"
        return super().save_model(filepath)
    
    def load_model(self, filepath):
        filepath += "_hnn"
        return super().load_model(filepath)


