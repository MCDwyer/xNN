import numpy as np

MIN_VALUE = 1e-50

EXP3 = "exp3"
EXP3P = "exp3p"
EXP3S = "exp3s"

ALGO = EXP3

class Exp3():

    def __init__(self, T, num_actions, rho=1) -> None:
        self.v = None
        self.T = T
        self.num_actions = num_actions
        self.rho = rho
        self.eta = self.calculate_eta()
        self.indices = np.arange(self.num_actions)

    def calculate_eta(self):
        return self.rho*np.sqrt(2*np.log(self.num_actions)/(self.num_actions*self.T))

    def initialise_v(self):
        v = np.ones(self.num_actions)
        v = v/len(v)

        return v

    def draw_action(self, v):
        random_index = np.random.choice(self.indices, p=v)
    
        return random_index
    
    def update_prob(self, v, loss, i):

        for j in range(self.num_actions):

            if i == j:
                x_hat = loss/v[j]
            else:
                x_hat = 0

            v[j] = v[j]*np.exp(-self.eta*x_hat)

        Z = np.sum(v)

        v = v/Z

        return v



# def initialise(num_actions, t, eta):
#     global NUM_ACTIONS, T, ETA
#     NUM_ACTIONS = num_actions
#     T = t
#     ETA = np.sqrt(2*np.log(NUM_ACTIONS)/(NUM_ACTIONS*T)) # From slides Stephen shared

# def draw_action(ball):

#     indices = np.arange(NUM_ACTIONS)
#     random_index = np.random.choice(indices, p=ball.w)
    
#     return random_index

# def update_w(ball, loss, i):

#     for j in range(NUM_ACTIONS):

#         if i == j:
#             x_hat = loss/ball.w[j]
#         else:
#             x_hat = 0

#         ball.w[j] = ball.w[j]*np.exp(-ETA*x_hat)


#     Z = np.sum(ball.w)

#     ball.w = ball.w/Z

#     return ball
