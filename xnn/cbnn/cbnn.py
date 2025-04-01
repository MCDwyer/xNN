"""CBNN bandit model set up"""
import numpy as np
import uuid

from . import cbnn_utils as utils
from ..common.bandit_class import BanditAlgorithm
from ..nearest_neighbour.leaf_list import NavigatingNetList, LeafList

# constants needed
TLeafL = "L"
TLeafN = "N"
TNodeN = "N"
TNodeL = "L"
TNodeR = "R"
LEFT = "L"
CENTRE = "C"
RIGHT = "R"
UP = "UP"

DEFAULT_LEAF_LIST_TYPE = NavigatingNetList
        

class CBNNBinaryTree:
    """Class for the binary tree vertices that make up the binary tree B in the CBNN algorithm.
    The leaves of this tree are the actions."""
    def __init__(self, value=None):
        self.value = value
        self.unique_id = uuid.uuid4()
        self.left = None
        self.right = None
        self.A_v = None # contraction
        self.H_A_v = None # TST of contraction
        self.A_v_dict = {}
        self.H_A_v_dict = {}
        self.height = 1

class CBNNModel(BanditAlgorithm):

    def __init__(self, rho=None, binning_radius=None, T=None, leaf_list_type=DEFAULT_LEAF_LIST_TYPE, actions=None, action_function=None, find_diff=None, ):
        super().__init__()
        self.rho = rho
        self.T = T
        self.K = None
        self.binning_radius = binning_radius

        self.bt_B = None
        self.bt_Z = None
        self.tt_H_Z = None
        self.leaf_list = None
        self.leaf_list_type = leaf_list_type
        self.eta = None

        self.temp_memory = None
        self.prev_temp_memory = None

        self.history = {}
        self.history_for_saving_and_loading = []

        self.u_dict = {}
        self.nn_dict = {}

        self.bt_Z_dict = {}
        self.tt_H_Z_dict = {}
        self.bt_B_bt_J_dict = {}
        self.bt_B_tt_H_J_dict = {}

        self.use_loss_delay = False
        self.loss_delay_length = None
        # self.loss_delay = []
        self.loss_delay_history = []

        if actions is not None:
            self.actions = actions
            self.K = len(actions)
            self.eta = self.calculate_learning_rate()
            self.bt_B = self.create_bt_B()

        if action_function is not None:
            self.action_function = action_function

        if find_diff is not None:
            self.find_diff = find_diff
            self.leaf_list = leaf_list_type(find_diff)
            
    def create_bt_B(self):

        actions = self.actions
        leaves = []
        for action in actions:
            leaf_node = CBNNBinaryTree(action)
            leaves.append(leaf_node)

        # build tree from bottom up:
        parents = []
        children = leaves

        while len(children) != 1:
            num_parents = int(np.ceil(len(children)/2))

            for i in range(num_parents):
                left_child_index = i*2
                right_child_index = left_child_index + 1

                parent = CBNNBinaryTree(None)

                if right_child_index < len(children):
                    parent.left = children[left_child_index]
                    parent.right = children[right_child_index]
                    parent.height = np.max([children[left_child_index].height, children[right_child_index].height]) + 1

                    parents.append(parent)
                else:
                    parents.append(children[left_child_index])

            children = parents
            parents = []

        # print(children[0]) # root of bt_B
        bt_B = children[0]

        self.bt_B = bt_B
        return bt_B

    def set_dataset(self, dataset):
        self.actions = dataset.actions
        self.action_function = dataset.action_function
        self.find_diff = dataset.find_diff
        if self.leaf_list is None:
            self.leaf_list = self.leaf_list_type(dataset.find_diff)
        self.dataset_type = type(dataset)

        self.K = dataset.num_actions
        self.eta = self.calculate_learning_rate()

        self.bt_B = self.create_bt_B()
        
        return
    
    def calculate_learning_rate(self):
        # K is number of actions in set
        # T is number of trials
        return self.rho * np.sqrt(np.log(self.K) * np.log(self.T) / (self.K*self.T))

    def set_env(self, env, find_diff):
        self.actions = env.actions
        self.find_diff = find_diff
        if self.leaf_list is None:
            self.leaf_list = self.leaf_list_type(find_diff)

        self.K = env.num_actions
        self.eta = self.calculate_learning_rate()

        return

    def train(self, dataset, tree_info=False):
        """CBNN algorithm from 'Nearest Neighbour with Bandit Feedback' paper.
        This function performs the grow subroutine on the binary tree Z (bt_Z) and then recursively runs
        through the binary tree B (bt_B) in _CBNN to decide which action to do.
        """

        self.set_dataset(dataset)
        # if self.dataset is None:
        #     self.set_dataset(dataset)
        # else:
        #     # check dataset is the right type (otherwise the action etc. is going to be diff and won't work??)
        #     assert isinstance(dataset, type(self.dataset))
        
        if self.bt_B is None:
            bt_B = self.create_bt_B()
        else:
            bt_B = self.bt_B

        leaf_list = self.leaf_list

        bt_Z = self.bt_Z
        tt_H_Z = self.tt_H_Z

        training_info = []

        dataset_size = len(dataset.dataset)

        # start_time = time.time()
        # last_time = start_time

        # print(f"Training starting on dataset with size: {len(dataset.dataset)} at time {int(start_time)}.")
        # print(f"rho: {self.rho}, eta: {self.eta}")

        for i, data in enumerate(dataset.dataset):
            # print(f"trial {i}")
            x = data[:-1]
            y = data[-1]
            bt_B, bt_Z, tt_H_Z, leaf_list, action, loss = utils.cbnn(self, bt_B, bt_Z, tt_H_Z, leaf_list, x, y)

            if action is not None:
                training_info.append([action, loss])

        # info_array = np.array(training_info)
        # avg_loss = np.mean((info_array[:, 0]).astype(float))

        # print(f"Training finished after {dataset_size} trials with average loss {avg_loss:.3f}.")
        # print(f"Training took: {int(time.time()) - start_time} seconds.")
        # cm = create_confusion_matrix([info_array], dataset.actions)
        # print(cm.astype(int))
        # if cm.sum(axis=1)[:, np.newaxis].all() != 0:
        #     print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        # print(tt_H_Z.height)

        # print(f"bt_Z.height: {bt_Z.height}, bt_B.left.A_v.height: {bt_B.left.A_v.height}")
        self.bt_B = bt_B
        self.bt_Z = bt_Z
        self.tt_H_Z = tt_H_Z
        self.leaf_list = leaf_list

        return np.array(training_info)

    def initialise_env(self, env, find_diff):
        self.set_env(env, find_diff)
        if self.bt_B is None:
            self.bt_B = self.create_bt_B()

    def predict_q_value(self, x):
        self.prev_temp_memory = self.temp_memory

        action = self.predict(x)

        return action

    def learn_q_value(self, x, loss):
        if self.prev_temp_memory is not None:
            action, pi_sum, history_for_saving, u_uuid = self.prev_temp_memory
        else:
            action, pi_sum, history_for_saving, u_uuid = self.temp_memory

        u = self.u_dict[u_uuid]
        nn = self.nn_dict[u_uuid]

        # print(self.u_dict)

        path = history_for_saving["path"]
        pi_values = history_for_saving["pi_values"]

        if isinstance(x, (int, float)):
            assert x == u.value, print(f"x: {x}, u: {u.value}, uuid: {u_uuid}, prev_temp: {self.prev_temp_memory}, temp: {self.temp_memory}")
        elif isinstance(x, (list, np.ndarray)):
            assert np.equal(u.value, x).all()
        else:
            print("The variable is an object or of another type.")

        utils.create_history_for_saving(self, u, nn, action, pi_sum, history_for_saving, loss)

        self.bt_B = utils.cbnn_learn_from_loss(self, self.bt_B, u, path, pi_values, loss, pi_sum)
        # self.prev_temp_memory = None

        return self

    def predict(self, x):
        if self.actions is None:
            raise RuntimeError("Actions not provided, model needs actions before it can predict.")

        action, pi_sum, history_for_saving, u, nn, self.bt_B, self.bt_Z, self.tt_H_Z, self.leaf_list = utils.cbnn_grow_and_find_action(self, self.bt_B, self.bt_Z, self.tt_H_Z, self.leaf_list, x)
        self.u_dict[u.unique_id] = u
        self.nn_dict[u.unique_id] = nn
        self.temp_memory = (action, pi_sum, history_for_saving, u.unique_id)

        if not history_for_saving["path"]:
            print(x, u.value, u.unique_id)

        return action

    def learn(self, x, loss):
        action, pi_sum, history_for_saving, u_value = self.temp_memory

        u = self.u_dict[u_value]
        nn = self.nn_dict[u_value]

        path = history_for_saving["path"]
        pi_values = history_for_saving["pi_values"]

        if isinstance(x, (int, float)):
            assert x == u.value, print(f"x: {x}, u: {u.value}")
        elif isinstance(x, (list, np.ndarray)):
            assert np.equal(u.value, x).all()
        else:
            print("The variable is an object or of another type.")

        utils.create_history_for_saving(self, u, nn, action, pi_sum, history_for_saving, loss)

        self.bt_B = utils.cbnn_learn_from_loss(self, self.bt_B, u, path, pi_values, loss, pi_sum)
        self.temp_memory = None


    def cbnn_one_trial(self, x, label=None):
        action = self.predict(x)

        loss = self.action_function(x, action, label)

        self.learn(x, loss)

        return action, loss
    

    def single_step_initialisation(self, dataset):
        self.set_dataset(dataset)
        
        if self.bt_B is None:
            self.bt_B = self.create_bt_B()

        print(f"rho: {self.rho}, eta: {self.eta}")

    def single_step(self, state):
        bt_B = self.bt_B

        leaf_list = self.leaf_list

        bt_Z = self.bt_Z
        tt_H_Z = self.tt_H_Z

        bt_B, bt_Z, tt_H_Z, leaf_list, info = utils.cbnn(self, bt_B, bt_Z, tt_H_Z, leaf_list, state)

        self.bt_B = bt_B
        self.bt_Z = bt_Z
        self.tt_H_Z = tt_H_Z
        self.leaf_list = leaf_list

        return state, np.array(info)
    
    def load_model(self, filepath):
        # filepath += "_cbnn"

        # load model in
        model = super().load_model(filepath)

        # print(model.bt_Z)
        # replace object links
        model.replace_object_links()
        # print(model.bt_Z)

        return model

