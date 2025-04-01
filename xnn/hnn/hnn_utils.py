from ..cbnn import cbnn_utils as utils

VALUE = 'value'
DIFF = 'diff'
TRIAL_NUM = 'trial_num'

class HItem:

    def __init__(self, value, node) -> None:
        self.value = value
        self.node = node

class HSet:
    def __init__(self, depth, find_diff, leaf_list_type) -> None:
        self.d = depth
        self.trials = leaf_list_type(find_diff)

class AllHSets:
    def __init__(self, binning_radius, rho, T, find_diff, leaf_list_type):#, dataset) -> None:
        self.HSets = {}
        self.binning_radius = binning_radius
        self.rho = rho
        self.T = T
        self.find_diff = find_diff
        self.leaf_list_type = leaf_list_type

    def get_set(self, depth):
        return self.HSets[depth]
    
    def add_new_set(self, depth):
        if depth in self.HSets:
            print("Set already exists")
        else:
            self.HSets[depth] = HSet(depth, self.find_diff, self.leaf_list_type)

        return self.HSets[depth]
    
    def add_to_set(self, depth, x, node):
        self.HSets[depth].trials.add_leaf(HItem(x, node))

    # def get_trial(self, depth, s):
    #     H_Set = self.get_set(depth)
        
    #     for trial in H_Set:
    #         if trial.trial_number == s:
    #             return trial
            
    #     print(f"Trial no. {s} doesn't exist in this set at depth: {depth}")
    #     return None

    def get_max_depth(self):
        max_depth = len(self.HSets) - 1

        return max_depth
    
    def nearest_neighbour(self, x, depth):
        H_set = self.get_set(depth)

        H_item, diff = H_set.trials.nearest_neighbour(x)
        
        return diff, H_item.node

        # min_diff = 1000000
        # for trial in self.HSets[depth].trials:
        #     y = trial.value
        #     diff = self.dataset.find_diff(x, y)

        #     if diff < min_diff:
        #         min_diff = diff
        #         s = trial.node

        # return min_diff, s
    
    def binning(self, x):

        x_exists = False
        d = None
        s = None

        for depth in self.HSets:
            diff, nn = self.nearest_neighbour(x, depth)

            if diff <= self.binning_radius:
                x_exists = True
                d = depth
                s = nn
                return x_exists, d, s

            # for trial in self.HSets[depth].trials:
            #     y = trial.value
            #     diff = self.dataset.find_diff(x, y)

            #     if diff <= self.binning_radius:
            #         x_exists = True
            #         d = depth
            #         s = trial.node
            #         return x_exists, d, s

        return x_exists, d, s

def hnn(model, H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, x, label, trial_number):

    x_exists = False
    diff = 1

    if H_sets is None:
        # set per d, so H_set[d] = the d depth set
        H_sets = AllHSets(model.binning_radius, model.rho, model.T, model.find_diff, model.leaf_list_type)#, model.dataset)
        s_delta = 0
        delta = -1
        h = -1
    else:
        x_exists, d, s = H_sets.binning(x)

        if x_exists:
            s_delta = s
            delta = d
        else:
            h = H_sets.get_max_depth()
            delta = 0
            s_delta = 0

            depths = range(0, h+1)

            for d in depths[::-1]:
                diff, nn = H_sets.nearest_neighbour(x, d)

                f_d = 0.5**d
                if diff <= f_d:
                    delta = d
                    s_delta = nn
                    break

    if diff > 1:
        print("Error: Data must be within a radius of 1, something has gone wrong, maybe need to normalise the data?")
        raise ValueError

    bt_B, bt_Z, tt_H_Z, leaf_list, u, action, loss = utils.cbnn_for_hnn(model, bt_B, bt_Z, tt_H_Z, leaf_list, trial_number, (s_delta, diff), x, label)

    if not x_exists:
        d = delta + 1
        # insert s into the H_set
        if d > h:
            H_sets.add_new_set(d)

        H_sets.add_to_set(d, x, u)

    return H_sets, bt_B, bt_Z, tt_H_Z, leaf_list, action, loss
