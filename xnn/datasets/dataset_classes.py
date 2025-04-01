import numpy as np
from scipy.io import arff

from xnn.common.bandit_class import BanditDataset

class ExampleScalarData(BanditDataset):
    def __init__(self, dataset=None, actions=[0, 1], dataset_size=None) -> None:
        if dataset is None:
            dataset = np.zeros((dataset_size, 2))
            class_label = 0

            for i in range(dataset_size):

                dataset[i][0] = np.random.uniform(0, 0.5)

                if i > dataset_size/2:
                    class_label = 1
                    dataset[i][0] = 1 - dataset[i][0]

                dataset[i][1] = class_label

        super().__init__(dataset, actions)

    def action_function(self, x, a, label=None):
        return self.classification_action(x, a, label)

class BasicClassificationData(BanditDataset):

    def __init__(self, dataset=None, actions=None, class_splits=None, dataset_size=1000, radius=30, class_centres=None) -> None:
        if class_splits is None:
            class_splits = [0.5, 0.5]
        
        self.class_splits = class_splits
        self.dataset_size = dataset_size
        self.radius = radius

        if actions is None:
            actions = range(0, len(class_splits))

        if dataset is None:
            # generate data
            dataset = self.generate_dataset(radius, class_centres)

        dataset = np.array(dataset)

        dataset = self.normalise_data(dataset)

        super().__init__(dataset, actions)

    def action_function(self, x, a, label=None):
        return self.classification_action(x, a, label)

    @staticmethod
    def create_class_dataset(centre_point, radius, size, label):
        x_list = []
        y_list = []
        xy_list = []

        while len(x_list) < size:
            x = centre_point[0] + np.random.uniform(-radius, radius)
            y = centre_point[1] + np.random.uniform(-radius, radius)

            distance = np.linalg.norm(np.array([x, y]) - centre_point)

            if distance < radius:
                x_list.append(x)
                y_list.append(y)
                xy_list.append([x, y, label])
        
        return x_list, y_list, xy_list
    
    def generate_dataset(self, radius, class_centres=None):
        # radius of class circles

        if class_centres is None:
            # define the class ranges:
            # centre points
            class_1_centre = [radius, radius]
            class_2_centre = [radius, 3*radius]
            class_3_centre = [3*radius, 3*radius]
            class_4_centre = [3*radius, radius]
        else:
            class_1_centre = class_centres[0]
            class_2_centre = class_centres[1]

        class_1_size = self.class_splits[0]*self.dataset_size
        class_2_size = self.class_splits[1]*self.dataset_size

        class_1_x, class_1_y, class_1 = self.create_class_dataset(class_1_centre, radius, class_1_size, 0)
        class_2_x, class_2_y, class_2 = self.create_class_dataset(class_2_centre, radius, class_2_size, 1)

        entries = class_1 + class_2

        if len(self.class_splits) > 2:
            class_3_size = self.class_splits[2]*self.dataset_size
            class_4_size = self.class_splits[3]*self.dataset_size
            class_3_x, class_3_y, class_3 = self.create_class_dataset(class_3_centre, radius, class_3_size, 2)
            class_4_x, class_4_y, class_4 = self.create_class_dataset(class_4_centre, radius, class_4_size, 3)

            entries += class_3 + class_4

        return entries
    
    # def generate_boundary_constants(self, boundary_type):

    #     if boundary_type == "linear":
    #         # Ax + By + C = 0
    #         size = 3
    #     elif boundary_type == "quadratic":
    #         # Ax^2 + Bx + Cy + D = 0
    #         size = 4

    #     constants = np.random.uniform(-self.radius, self.radius, size)

    #     return constants

    # def check_boundary(self, x, y):

    #     point = self.constants[0]*x + self.constants[1]*y + self.constants[2]

    #     if len(self.constants) > 3:
    #         point += self.constants[3]*(x**2)

    #     # if Ax + By + C > 0
    #     if point > 0:
    #         return 0
    #     # if = 0 -> 
    #     elif point < 0:
    #         return 1
    #     else:
    #         return np.random.choice([0, 1])

    # def generate_dataset(self):


class HNNExampleData(BanditDataset):
    def __init__(self, dataset=None, actions=None, class_splits=[0.5, 0.5], dataset_size=1000, radius=30, class_centres=None, num_dim=2, boundary_type="0") -> None:
        self.class_splits = class_splits
        self.dataset_size = dataset_size

        epsilon = (1/np.sqrt(self.dataset_size/2))
        self.prob_a = 0.5 + epsilon
        self.prob_b = 0.5

        if actions is None:
            actions = []
            for i in range(0, len(class_splits)):
                actions.append(i)

        if dataset is None:
            # generate data
            dataset = self.generate_dataset(radius, class_centres, num_dim, boundary_type)

        dataset = np.array(dataset)
        dataset = self.normalise_data(dataset)

        super().__init__(dataset, actions)

    def action_function(self, x, a):
        return self.classification_action(x, a)
        # return self.bandit_action(x, a, self.prob_a, self.prob_b)

    @staticmethod
    def create_data(centre_point, radius, size, boundary_type):
        data_list = []


        while len(data_list) < size:
            data = np.zeros(len(centre_point))
            for i in range(len(centre_point)):
                data[i] = centre_point[i] + np.random.uniform(-radius, radius)

            distance = np.linalg.norm(data - centre_point)

            if distance < radius:
                label = 0 if data[1] < 0 else 1
                labelled_data = np.zeros((len(centre_point) + 1))
                labelled_data[:-1] = data
                labelled_data[-1] = label
                data_list.append(labelled_data)
        
        return data_list
    
    def generate_dataset(self, radius, class_centres, num_dim, boundary_type):
        # radius of class circles
        radius_2 = self.dataset_size**(-0.5)
        
        if class_centres is None:
            # define the class ranges:
            # centre points
            class_1_centre = np.zeros(num_dim)
            class_2_centre = np.zeros(num_dim)
            
            class_1_centre[0] = radius
            class_2_centre[0] = (2*radius + radius_2)
        else:
            class_1_centre = class_centres[0]
            class_2_centre = class_centres[1]

        # class_1_size = self.class_splits[0]*self.dataset_size
        # class_2_size = self.class_splits[1]*self.dataset_size

        # # create function for boundary
        # if boundary_type == "0":
        #     # class is 0 if y is -ve and 1 otherwise
        #     def which_class(data):
        #         if data[1] < 0:
        #             return 0
        #         else:
        #             return 1
                
        #     self.which_class = which_class

        # elif boundary_type == "linear":
        #     # ax + by + c = 0
        #     a = 0
        #     b = 0
        #     c = 0
        #     boundary = np.array([a, b, c])
            
        #     def which_class(data):

        # elif boundary_type == "quadratic":
        #     # y = ax^2 + bx + c

        class_1 = self.create_data(class_1_centre, radius, self.dataset_size, boundary_type)

        # radius_2 = self.dataset_size**(-0.5)
        # class_2 = self.create_data(class_2_centre, radius_2, class_2_size, boundary_type)

        entries = class_1

        return entries

class ProbabilisticExampleData(BanditDataset):
    def __init__(self, dataset=None, actions=None, class_splits=[0.5, 0.5], dataset_size=1000, radius=30, w=0.5) -> None:
        self.class_splits = class_splits
        self.dataset_size = dataset_size
        self.w = w

        if actions is None:
            actions = range(0, len(class_splits))

        if dataset is None:
            # generate data
            dataset = self.generate_dataset(radius)

        dataset = np.array(dataset)

        max_x = np.max(dataset[:, 0])
        max_y = np.max(dataset[:, 1])

        dataset[:, 0] = dataset[:, 0]/(max_x*2)
        dataset[:, 1] = dataset[:, 1]/(max_y*2)

        super().__init__(dataset, actions)

    def action_function(self, x, a):
        # a is action
        hard_loss, label = self.classification_action(x, a)

        dist_from_decision_boundary = np.abs(x[1] - 0)
        # With probability p(x) the loss is (0,1) and with probability 1-p(x) the loss is (1,0)

        if dist_from_decision_boundary > self.w:
            # If a context x is at distance more than w from the decision boundary then p(x)=1 or p(x)=0 (depending on which side of the decision boundary x is)
            return hard_loss, label
        else:
            # when x is within a distance d<w from the decision boundary then p(x)=(1/2)+(d/2w) or p(x)=(1/2)-(d/2w) 
            # (depending on which side of the decision boundary x is - so it is consistent with the previous probabilities when d=w).
            p_x = 0.5 + (x[1]/(2*self.w))

            drawn_loss = np.random.choice([0, 1], p=[p_x, (1-p_x)])

            return drawn_loss, label

    @staticmethod
    def create_data(centre_point, radius, size):
        x_list = []
        y_list = []
        xy_list = []

        while len(x_list) < size:
            x = centre_point[0] + np.random.uniform(-radius, radius)
            y = centre_point[1] + np.random.uniform(-radius, radius)

            distance = np.linalg.norm(np.array([x, y]) - centre_point)

            if distance < radius:
                x_list.append(x)
                y_list.append(y)
                label = 0 if y < 0 else 1
                xy_list.append([x, y, label])
        
        return x_list, y_list, xy_list
    
    def generate_dataset(self, radius):
        centre = [0, 0]

        x, y, data = self.create_data(centre, radius, self.dataset_size)

        return data

class LineExampleData(BanditDataset):
    def __init__(self, dataset=None, actions=None, dataset_size=None, num_segments=30, segment_size=30, num_dim=2) -> None:
        self.class_splits = [0.5, 0.5]
        self.factor = 10
        if dataset_size is None:
            self.dataset_size = num_segments*segment_size
            self.num_segments = num_segments
            self.segment_size = segment_size
        else:
            n = int(np.sqrt(dataset_size))
            self.dataset_size = n*n
            self.num_segments = n
            self.segment_size = n

        epsilon = (1/np.sqrt(self.segment_size/2))
        self.prob_a = 0.5 + epsilon
        self.prob_b = 0.5

        print(self.prob_b)

        if actions is None:
            actions = []
            for i in range(0, len(self.class_splits)):
                actions.append(i)

        if dataset is None:
            # generate data
            dataset = self.generate_dataset(self.num_segments, self.segment_size, num_dim)

        dataset = np.array(dataset, dtype=np.float64)
        dataset = self.normalise_data(dataset)

        super().__init__(dataset, actions, shuffle=False)

        self.shuffle_dataset()

    def generate_dataset(self, num_segments, segment_size, num_dim):
        data = []

        # For the n-th segment we have (1,2n), (2,2n), (3,2n)…(M,2n)
        if num_dim == 2:
            for n in range(num_segments):
                for m in range(segment_size):
                    label = 0 if m < (segment_size/2) else 1
                    new_data = [m, (n*self.factor), label]
                    # print(f"m: {m}, n: {n}, label: {label}")
                    data.append(new_data)
        else:
            grid_size = int(np.sqrt(segment_size))
            self.segment_size = grid_size
            for n in range(num_segments):
                for m in range(grid_size):
                    label = 0 if m < (grid_size/2) else 1
                    for k in range(grid_size):
                        new_data = [m, (n*self.factor), (k*self.factor), label]
                        # print(f"m: {m}, n: {n}, label: {label}")
                        data.append(new_data)

        return data

    def shuffle_dataset(self):
        # to shuffle but still maintain ordering of each segment, we only shuffle the segment order
        shuffled_data = np.zeros_like(self.dataset)

        segments = list(range(self.num_segments))
        np.random.shuffle(segments)

        for n in range(self.num_segments):
            random_n = segments[n]
            start = random_n*self.segment_size
            end = start + self.segment_size

            shuffled_data[(n*self.segment_size):((n+1)*self.segment_size)] = self.dataset[start:end]

        self.dataset = shuffled_data

        return shuffled_data

    def action_function(self, x, a):
        return self.bandit_action(x, a, self.prob_a, self.prob_b)
