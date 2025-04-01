import numpy as np


def calculate_psi(t):
    # φ(n) = O(log n) from paper for best bounds(?)

    return np.log(t)

def calculate_uncertainty(t, N, reward, theta):
    # Ua t,k (x) := √ (θ log t) /N a t,k(x) + φ(t) · rt,k(x).
    psi = calculate_psi(t)
    uncertainty = np.sqrt(theta*np.log(t)/N) + (psi*reward)

    return uncertainty

def log_calc(p, q):
    # from the kl_ucb paper -> 0 log 0 = 0 log 0/0 = 0 and x log x/0 = +∞ for x > 0

    if p == 0:
        pq_log = 0
    elif q == 0:
        pq_log = np.inf
    else:
        pq_log = p*np.log(p/q)

    return pq_log

def kl_divergence(p, q):
    # d(p, q) := p log (p/q) + (1 − p) · log ((1 − p) / (1 − q)) .

    pq_log = log_calc(p, q)
    pq_minus_log = log_calc((1-p), (1-q))

    # print(f"p: {p}, q: {q}")
    # print(f"pq log: {pq_log}, pq - log: {pq_minus_log}")
    divergence = pq_log + pq_minus_log

    return divergence

def nearest_neighbour(x, history, find_diff):
    """Return ordered list of history, ordered by distance to x (most similar first)"""
    distances = []

    for i, previous in enumerate(history):
        rho = find_diff(x, previous[0])

        distances.append([i, rho, previous[1]])

    distances = np.array(distances)

    # sort by distances
    distances = distances[distances[:, 1].argsort()]

    indices = np.column_stack((distances[:, 0], distances[:, 2]))

    return indices

    # def knn_index_initialisation(self, x):

    #     if self.history is None:
    #         t = 0
    #         self.history = []
    #     else:
    #         t = len(self.history)

    #     loss, label = self.dataset.action_function(x, t)
    #     reward = 1 - loss
    #     self.history.append([x, t, reward])

    #     return loss, t, label
    
    # def knn_index_strategy(self, x):

    #     max_pi = -100
    #     t = len(self.history) + 1

    #     ordered_indices = utils.nearest_neighbour(x, self.history, self.dataset.find_diff)

    #     for action in self.dataset.actions:
    #         # find the k neighbours for each action:
    #         knns = self.find_knn(action, ordered_indices, t)

    #         pi = self.index_function(knns, t)
    #         # print(f"action: {action}, pi: {pi}")

    #         if pi > max_pi:
    #             max_pi = pi
    #             best_action = action

    #     loss, label = self.dataset.action_function(x, best_action)
    #     reward = 1 - loss # for loss in range 0 - 1

    #     self.history.append([x, best_action, reward])

    #     return loss, best_action, label

    # def knn_ucb_index(self, knns, t):
    #     # Ia t,k(x) = ˆ fa t,k(x) + U a t,k(x).

    #     # f_hat = S/N -> mean reward
    #     rewards = [row[-1] for row in knns]
    #     rewards = np.array(rewards)

    #     mean_reward = np.mean(rewards)

    #     # index = f(x) + calculate_uncertainty(t, N, reward)
    #     N = len(knns)

    #     index = mean_reward + utils.calculate_uncertainty(t, N, rewards[-1], self.theta)

    #     return index

    # def knn_kl_ucb_index(self, knns, t):
    #     # Ia t,k(x) = sup { ω ∈ [0, 1] : N a t,k(x) · d (ˆ fa t,k(x), ω ) ≤ θ · log t } + φ(t) · rt,k(x).
    #     N = len(knns)

    #     # f_hat = S/N -> mean reward
    #     rewards = [row[-1] for row in knns]
    #     rewards = np.array(rewards)

    #     mean_reward = np.mean(rewards)

    #     num = int((1 / (0.05)) + 1)
    #     omegas = np.linspace(0, 1, num)
        
    #     theta_logt = self.theta*np.log(t)
    #     # print(f"theta_logt: {theta_logt}")

    #     w_max = 0

    #     for w in omegas:
    #         test = N * utils.kl_divergence(mean_reward, w)
            
    #         if test <= theta_logt:
    #             w_max = w

    #     psi = utils.calculate_psi(t)

    #     index = w_max + (psi*rewards[-1])

    #     return index
    
    # def find_knn(self, action, ordered_indices, t):

    #     k_history = []

    #     action_mask = (ordered_indices[:, 1] == action)

    #     # Use the mask to filter the array and only look at the data where the action is the same
    #     action_indices = ordered_indices[action_mask]
    #     action_indices = action_indices[:, 0]

    #     min_uncertainty_index = 0
    #     min_uncertainty = utils.calculate_uncertainty(t, 1, self.history[min_uncertainty_index][-1], self.theta)

    #     for i, index in enumerate(action_indices):
    #         index = int(index)
    #         uncertainty = utils.calculate_uncertainty(t, (i+1), self.history[index][-1], self.theta)
    #         k_history.append(self.history[index])
    #         # print(f"i: {i}, index: {index}, uncertainty: {uncertainty}")

    #         if uncertainty < min_uncertainty:
    #             min_uncertainty = uncertainty
    #             min_uncertainty_index = i

    #     return k_history[:min_uncertainty_index + 1]

# def knn_index_initialisation(self, x):

#     if self.history is None:
#         t = 0
#         self.history = []
#     else:
#         t = len(self.history)

#     loss, _ = self.dataset.action_function(x, t)
#     reward = 1 - loss
#     self.history.append([x, t, reward])

#     return
    
#     def knn_index_strategy(self, x):

#         max_pi = -100
#         t = len(self.history) + 1

#         ordered_indices = utils.nearest_neighbour(x, self.history, self.dataset.find_diff)
#         # print(ordered_indices)

#         for action in self.dataset.actions:
#             # find the k neighbours for each action:
#             knns = self.find_knn(action, ordered_indices, t)

#             pi = self.index_function(knns, t)
#             # print(f"action: {action}, pi: {pi}")

#             if pi > max_pi:
#                 max_pi = pi
#                 best_action = action

#         loss, label = self.dataset.action_function(x, best_action)
#         reward = 1 - loss # for loss in range 0 - 1

#         self.history.append([x, best_action, reward])

#         return loss, best_action, label

#     def knn_ucb_index(self, knns, t):
#         # Ia t,k(x) = ˆ fa t,k(x) + U a t,k(x).

#         # f_hat = S/N -> mean reward
#         rewards = [row[-1] for row in knns]
#         rewards = np.array(rewards)

#         mean_reward = np.mean(rewards)

#         # index = f(x) + calculate_uncertainty(t, N, reward)
#         N = len(knns)

#         index = mean_reward + utils.calculate_uncertainty(t, N, rewards[-1], self.theta)

#         return index

#     def knn_kl_ucb_index(self, knns, t):
#         # Ia t,k(x) = sup { ω ∈ [0, 1] : N a t,k(x) · d (ˆ fa t,k(x), ω ) ≤ θ · log t } + φ(t) · rt,k(x).
#         N = len(knns)

#         # f_hat = S/N -> mean reward
#         rewards = [row[-1] for row in knns]
#         rewards = np.array(rewards)

#         mean_reward = np.mean(rewards)

#         num = int((1 / (0.05)) + 1)
#         omegas = np.linspace(0, 1, num)
        
#         theta_logt = self.theta*np.log(t)
#         # print(f"theta_logt: {theta_logt}")

#         w_max = 0

#         for w in omegas:
#             test = N * utils.kl_divergence(mean_reward, w)
            
#             if test <= theta_logt:
#                 w_max = w

#         psi = utils.calculate_psi(t)

#         index = w_max + (psi*rewards[-1])

#         return index
    
#     def find_knn(self, action, ordered_indices, t):

#         k_history = []

#         action_mask = (ordered_indices[:, 1] == action)

#         # Use the mask to filter the array and only look at the data where the action is the same
#         action_indices = ordered_indices[action_mask]
#         action_indices = action_indices[:, 0]

#         min_uncertainty_index = 0
#         min_uncertainty = utils.calculate_uncertainty(t, 1, self.history[min_uncertainty_index][-1], self.theta)

#         for i, index in enumerate(action_indices):
#             index = int(index)
#             uncertainty = utils.calculate_uncertainty(t, (i+1), self.history[index][-1], self.theta)
#             k_history.append(self.history[index])
#             # print(f"i: {i}, index: {index}, uncertainty: {uncertainty}")

#             if uncertainty < min_uncertainty:
#                 min_uncertainty = uncertainty
#                 min_uncertainty_index = i

#         return k_history[:min_uncertainty_index + 1]