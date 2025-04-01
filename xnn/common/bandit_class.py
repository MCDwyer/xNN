import copy
import numpy as np
import pickle
import os

MAX_VALUE = 1e50
MIN_VALUE = 1e-50

class BanditAlgorithm:

    def __init__(self) -> None:
        # self.dataset = None
        self.dataset_type = None
        self.action_function = None
        self.actions = None
        self.find_diff = None

    def train(self, dataset):
        raise NotImplementedError("Subclasses must implement abstract method")

    def predict(self, x):
        raise NotImplementedError("Subclasses must implement abstract method")
    
    def save_training_info(self, info, filepath):
        with open(f"{filepath}.npy", "wb", buffering=0) as outf:
            np.save(outf, info)

    def save_model(self, filepath):
        if "." not in filepath:
            filepath += ".pkl"

        try:
            with open(filepath, 'wb', buffering=0) as filehandler:
                pickle.dump(self, filehandler, pickle.HIGHEST_PROTOCOL)
            print(f"Model saved to: {filepath}")
        except Exception as e:
            os.remove(filepath)
            print(f"Model couldn't be saved: {e}")
            raise
        finally:
            filehandler.close()

        return filepath

    def load_model(self, filepath):
        if "." not in filepath:
            filepath += ".pkl"

        with open(filepath, 'rb') as filehandler:
            model = pickle.load(filehandler)
        print(f"Model loaded from: {filepath}")

        return model

    def validate(self, val_dataset):
        validation_info = []

        # doing this means that we don't keep any of the learning from the validation dataset?
        model_copy = copy.deepcopy(self)
        model_copy.dataset.add_dataset(val_dataset)

        for x in val_dataset[:, :-1]:
            model = copy.deepcopy(model_copy) # reset model between each datapoint of the validation set
            info = model.predict(x)
            
            if info is not None:
                validation_info.append(info)

        return np.array(validation_info)

class BanditDataset:

    def __init__(self, dataset, actions, shuffle=True) -> None:
        self.dataset = np.array(dataset)
        self.dataset_size = len(dataset)
        self._actions = None
        self.num_actions = None

        self.actions = actions

        if shuffle:
            self.shuffle_dataset()

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, actions):
        self._actions = actions
        self.num_actions = len(actions)

    @staticmethod
    def normalise_data(dataset):
        num_dim = dataset.shape[1] - 1 # ignoring label bit

        max_value = np.max(dataset[:, :-1])
        min_value = np.min(dataset[:, :-1])

        dataset[:, :-1] = (dataset[:, :-1] - min_value)/(max_value - min_value) # standard normalising
        # dataset[:, :-1] = dataset[:, :-1]/num_dim # to make sure that the distances are never more than 1

        # dataset[:, :-1] = np.clip(dataset[:, :-1], 0, MAX_VALUE)

        return dataset

    @staticmethod
    def reduce_data_range(dataset, dims_to_reduce=None):
        """Reduces the dataset range column by column to be between 0 and 1
        i.e. relative scaling between columns is lost"""

        num_dims = dataset.shape[1] - 1 # to remove label

        if dims_to_reduce is None:
            dims_to_reduce = range(num_dims)

        for i in range(num_dims):
            if i in dims_to_reduce:
                # TODO: sometimes get infs in max_value - probably need to fix that in better way at some point??
                max_value = np.max(dataset[:, i])
                min_value = np.min(dataset[:, i])

                # print(f"{i}: min_value = {min_value}, max_value = {max_value}")
                if max_value == min_value:
                    if max_value != 0:
                        dataset[:, i] = dataset[:, i]/max_value
                else:
                    # print(f"pre change: {dataset[0, i]}")
                    dataset[:, i] = (dataset[:, i] - min_value)/(max_value - min_value)
                    # print(f"post change: {dataset[0, i]}")

        # dataset[:, :-1] = np.clip(dataset[:, :-1], 0, MAX_VALUE)

        return dataset

    def add_noise(self, std=0.1, noise_threshold=0.5):

        total_num_trials = self.dataset.shape[0]

        noise_threshold = int(total_num_trials*noise_threshold)

        self.dataset[noise_threshold:, :-1] += np.random.normal(0, std, ((total_num_trials - noise_threshold), (self.dataset.shape[1] - 1)))

        self.dataset = self.normalise_data(self.dataset) # TODO: should I re-normalise here or clip it?

        return self.dataset

    def add_drift(self, num_dim=None, drift_threshold=0.5, start_dim=0):

        if num_dim is None:
            # if num_dim not specified, will just add drift to all dimensions
            num_dim = (self.dataset.shape[1] - 1)

        total_num_trials = self.dataset.shape[0]

        end_dim = start_dim + num_dim

        max_drift = np.max(self.dataset[:, start_dim:end_dim])

        drift_threshold = total_num_trials*drift_threshold

        for i, entry in enumerate(self.dataset):

            if i > drift_threshold:
                percentage_drift = i/total_num_trials
                # gradually add drift to each data point?
                entry[start_dim:end_dim] = entry[start_dim:end_dim] + np.random.uniform(0, max_drift*percentage_drift)

        self.dataset = self.normalise_data(self.dataset) # TODO: should I re-normalise here or clip it?

        return self.dataset

    def add_dataset(self, dataset):
        stacked_data = np.vstack((self.dataset, dataset))
        self.dataset = stacked_data

    def shuffle_dataset(self):
        np.random.shuffle(self.dataset)

    @staticmethod
    def find_diff(x, y):
        x = np.array(x)
        y = np.array(y)

        diff = np.linalg.norm(x - y)

        return diff

    def action_function(self, x, a, label=None):
        raise NotImplementedError("Subclasses must implement abstract method")

    def classification_action(self, x, a, label):

        if isinstance(a, int):
            # then a is an index? so need to find the corresponding action
            a = self.actions[a]
            # label = int(label)

        if a == label:
            return 0
        else:
            return 1

    def bandit_action(self, x, a, label, prob_a, prob_b):
        # For class 1 the loss of action 1 is 0 with probability a and the loss of action 2 is 0 with probability b.
        # For class 2 the loss of action 1 is 0 with probability b and the loss of action 2 is 0 with probability a.
        # class - correct chosen prob a of 0, else prob b of 0

        # if self.classification_action(x, a) is 0 it was correctly chosen
        # TODO: fix
        loss, label = self.classification_action(x, a, label)
        label_correctly_chosen = True if loss == 0 else False

        if label_correctly_chosen:
            probabilities = [prob_a, (1 - prob_a)]
        else:
            probabilities = [prob_b, (1 - prob_b)]

        return np.random.choice([0, 1], p=probabilities)
    
# class BanditEnv:

