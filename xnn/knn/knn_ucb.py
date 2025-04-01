import numpy as np

from . import knn_ucb_utils as utils
from ..common.bandit_class import BanditAlgorithm

KNN_UCB = "KNN_UCB"
KNN_KL_UCB = "KNN_KL_UCB"

class KNNModel(BanditAlgorithm):

    def __init__(self, index_function_name, theta) -> None:
        super().__init__()
        self.history = None
        self.theta = theta
        self.index_function_name = index_function_name
        self.action_function = None
        self.find_diff = None
        self.num_actions = None

        print(f"Index function: {index_function_name}")

        if index_function_name == KNN_UCB:
            self.index_function = self.knn_ucb_index
        elif index_function_name == KNN_KL_UCB:
            self.index_function = self.knn_kl_ucb_index

    def train(self, dataset):
  
        self.action_function = dataset.action_function
        self.find_diff = dataset.find_diff
        self.num_actions = dataset.num_actions
        # if self.dataset is None:
        #     self.dataset = dataset
        # else:
        #     # check dataset is the right type (otherwise the action etc. is going to be diff and won't work??)
        #     assert isinstance(dataset, type(self.dataset))

        training_info = []

        if self.history is None:
            history = []
        else:
            history = self.history

        for data in dataset.dataset:
            x = np.array(data[:-1]).astype(float)
            label = data[-1]

            if len(history) < dataset.num_actions:
                history, loss, action = self.knn_index_initialisation(x, history, label)
            else:
                history, loss, action = self.knn_index_strategy(x, history, label)

            action = dataset.actions[int(action)]
            training_info.append([action, loss])

        self.history = history

        return np.array(training_info)

    def predict(self, data):

        x = np.array(data[:-1])
        label = data[-1]

        if self.history is None:
            raise RuntimeError("Model not trained, run model.train(dataset) first.")
        
        _, loss, action = self.knn_index_strategy(x, self.history, label)

        info = [loss, action, label]

        return info

    def knn_index_initialisation(self, x, history, label):

        if history is None:
            t = 0
            history = []
        else:
            t = len(history)

        loss = self.action_function(x, t, label)
        reward = 1 - loss
        history.append([x, t, reward])

        return history, loss, t
    
    def knn_index_strategy(self, x, history, label):

        max_pi = -100
        t = len(history) + 1

        ordered_indices = utils.nearest_neighbour(x, history, self.find_diff)

        for action in range(self.num_actions):
            # find the k neighbours for each action:
            knns = self.find_knn(action, ordered_indices, t, history)

            pi = self.index_function(knns, t)
            # print(f"action: {action}, pi: {pi}")

            if pi > max_pi:
                max_pi = pi
                best_action = action

        loss = self.action_function(x, best_action, label)
        reward = 1 - loss # for loss in range 0 - 1

        history.append([x, best_action, reward])

        return history, loss, best_action

    def knn_ucb_index(self, knns, t):
        # Ia t,k(x) = ˆ fa t,k(x) + U a t,k(x).

        # f_hat = S/N -> mean reward
        rewards = [row[-1] for row in knns]
        rewards = np.array(rewards)

        mean_reward = np.mean(rewards)

        # index = f(x) + calculate_uncertainty(t, N, reward)
        N = len(knns)

        index = mean_reward + utils.calculate_uncertainty(t, N, rewards[-1], self.theta)

        return index

    def knn_kl_ucb_index(self, knns, t):
        # Ia t,k(x) = sup { ω ∈ [0, 1] : N a t,k(x) · d (ˆ fa t,k(x), ω ) ≤ θ · log t } + φ(t) · rt,k(x).
        N = len(knns)

        # f_hat = S/N -> mean reward
        rewards = [row[-1] for row in knns]
        rewards = np.array(rewards)

        mean_reward = np.mean(rewards)

        num = int((1 / (0.05)) + 1)
        omegas = np.linspace(0, 1, num)
        
        theta_logt = self.theta*np.log(t)
        # print(f"theta_logt: {theta_logt}")

        w_max = 0

        for w in omegas:
            test = N * utils.kl_divergence(mean_reward, w)
            
            if test <= theta_logt:
                w_max = w

        psi = utils.calculate_psi(t)

        index = w_max + (psi*rewards[-1])

        return index
    
    def find_knn(self, action, ordered_indices, t, history):

        k_history = []

        action_mask = (ordered_indices[:, 1] == action)

        # Use the mask to filter the array and only look at the data where the action is the same
        action_indices = ordered_indices[action_mask]
        action_indices = action_indices[:, 0]

        min_uncertainty_index = 0
        min_uncertainty = utils.calculate_uncertainty(t, 1, history[min_uncertainty_index][-1], self.theta)

        for i, index in enumerate(action_indices):
            index = int(index)
            uncertainty = utils.calculate_uncertainty(t, (i+1), history[index][-1], self.theta)
            k_history.append(history[index])
            # print(f"i: {i}, index: {index}, uncertainty: {uncertainty}")

            if uncertainty < min_uncertainty:
                min_uncertainty = uncertainty
                min_uncertainty_index = i

        return k_history[:min_uncertainty_index + 1]
    
    def save_model(self, filepath):
        filepath += f"_{self.index_function_name}"
        return super().save_model(filepath)
    
    def load_model(self, filepath):
        filepath += f"_{self.index_function_name}"
        return super().load_model(filepath)