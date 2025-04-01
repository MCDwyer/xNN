import numpy as np

from ..common.exp3 import Exp3
from ..common.bandit_class import BanditAlgorithm

D = 0

T_0_FACTOR = 0.5

MIN_RADIUS = 1e-50

# dY = 0 and cY = O(√|Y |)

class CB_Ball:

    def __init__(self, value, radius, algo_b):
        self.value = value
        self.radius = radius
        self.n = 0
        self.reward = 0
        self.children = {}
        self.is_full = False
        self.v = algo_b.initialise_v()

class SlivkinsCB(BanditAlgorithm):

    def __init__(self, T, t0factor=0.5, rho=1, algo_b_type=Exp3) -> None:
        self.T = T
        self.t0factor = t0factor
        self.rho = rho
        self.A = None
        self.algo_b = None
        self.action_function = None
        self.find_diff = None
        self.actions = None
        self.num_actions = None
        self.algo_b_type = algo_b_type

        super().__init__()

    def set_dataset(self, dataset):
        self.algo_b = self.algo_b_type(self.T, len(dataset.actions), self.rho)
        self.action_function = dataset.action_function
        self.find_diff = dataset.find_diff
        self.actions = dataset.actions
        self.num_actions = dataset.num_actions
    
    def train(self, dataset):

        self.set_dataset(dataset)
        # if self.dataset is None:
        #     self.set_dataset(dataset)
        # else:
        #     # check dataset is the right type (otherwise the action etc. is going to be diff and won't work??)
        #     assert isinstance(dataset, type(self.dataset))

        training_info = []
        for data in dataset.dataset:
            x = data[:-1]
            label = data[-1]
            self.A, loss, action = self.slivkins_CB(self.A, x, label)
            training_info.append([action, loss])

        return np.array(training_info)
    
    def predict(self, data):

        if self.A is None:
            raise RuntimeError("Model not trained, run model.train(dataset) first.")

        x = data[:-1]
        label = data[-1]

        _, loss, action, label = self.slivkins_CB(self.A, x, label)

        return [loss, action, label]
    
    def T_0(self, radius):
        # cY r−(2+dY) log( 1 r )

        # print(f"radius: {radius}")

        # if radius > MIN_RADIUS:
        #     log_term = np.log(1/radius)
        #     radius_term = radius**(-(2+D))
        # else:
        #     # TODO: check how to handle this?
        #     log_term = 1
        #     radius_term = 1


        # t_0 = self.t0factor*np.sqrt(self.dataset.num_actions)*radius_term*log_term
        t_0 = self.t0factor*radius**(-2)*np.sqrt(2*np.log(self.num_actions)*self.num_actions)

        # r^{-2}sqrt{2\ln(K)K}

        # t_0 = radius*self.T*self.t0factor
        # TODO: check how best to determine the T_0_FACTOR??

        return int(t_0)

    def find_relevant_balls(self, A, x):

        relevant_inactive_balls_radii = []
        relevant_active_balls = []

        for key in A.children:
            for ball in A.children[key]:
                diff = self.find_diff(ball.value, x)

                # print(f"diff: {diff}, radius: {ball.radius}")

                if diff < ball.radius:
                    if ball.is_full:
                        relevant_inactive_balls_radii.append(ball.radius)
                    else:
                        relevant_active_balls.append(ball)
        
        return relevant_active_balls, relevant_inactive_balls_radii

    def initialise_A(self, x):
        A = CB_Ball(None, None, self.algo_b)
        A.children[1] = [CB_Ball(x, 1, self.algo_b)]

        return A
    
    def slivkins_CB(self, A, x, label):
        # algorithm 2 in paper
        if A is None:
            A = self.initialise_A(x)
        
        relevant_active_balls, relevant_inactive_balls_radii = self.find_relevant_balls(A, x) # should return list with size: num_actions >= list_size > 0 (i.e. will be at least size 1 and have all_space ball in it?) 

        if relevant_active_balls:
            # non-empty list so exists
            ball_index = int(np.random.randint(0, (len(relevant_active_balls))) - 1)
            B = relevant_active_balls[ball_index]
            # says any but dunno why/how to pick? might be better picking the smallest radius one first maybe??
            # or maybe just randomly select one?
        else:
            # activate a new ball
            # find a ball of minimum radius in which the context would exist
            # then if that ball is not active then create a new ball with that radius?
            radius = min(relevant_inactive_balls_radii)
            B = CB_Ball(x, radius/2, self.algo_b)

            radius_key = (radius/2)
            if radius_key in A.children:
                A.children[radius_key].append(B)
            else:
                A.children[radius_key] = [B]

        y = self.algo_b.draw_action(B.v) # arm selected by ALG_B
        action = self.actions[int(y)]

        pi = self.action_function(x, action, label)

        # print(f"\n\nAction {y} chosen, loss: {pi}, label: {label}")
        # print(f"Pre-update probabilities: {B.v}")

        # report pi to ALG_B
        B.v = self.algo_b.update_prob(B.v, pi, y)
        B.n += 1

        # print(f"Post-update probabilities: {B.v}")
        # print(f"B.n = {B.n}, T_0(r): {self.T_0(B.radius)}")

        if B.n >= self.T_0(B.radius):
            # ball B is full, so remove from A*
            # A_star = copy.deepcopy(A)
            # A_star.children[(B.radius)].remove(B)

            # maybe actually maintain a value on the ball about it being full??
            B.is_full = True

        return A, pi, action

    def save_model(self, filepath):
        filepath += "_slivkins"
        return super().save_model(filepath)
    
    def load_model(self, filepath):
        filepath += "_slivkins"
        return super().load_model(filepath)
