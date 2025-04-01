import numpy as np

from .navigating_net import NavigatingNet, RNet

class LeafListBase:
    def __init__(self, find_diff):
        # the function for finding the difference between the items in this list
        self.find_diff = find_diff

    def add_leaf(self, leaf):
        raise NotImplementedError("Subclasses must implement abstract method")

    def nearest_neighbour(self, value):
        # value is not the same as leaf items, it is the value of the item
        raise NotImplementedError("Subclasses must implement abstract method")

    def update_leaf(self, old_leaf, new_leaf):
        # this is for when binning essentially, as need to update the pointer in the leaf list to 
        # point at the most recent leaf node corresponding to the same value, otherwise can end up 
        # getting recursion issues (esp. if large binning radii or very similar data - i.e. often binned)
        raise NotImplementedError("Subclasses must implement abstract method")

class NavigatingNetList(LeafListBase):
    def __init__(self, find_diff):
        # print("Navigating Net being used.")
        self.net = NavigatingNet()
        self.last_net = None
        self.node_dict = {}
        self.find_diff = find_diff

    def add_leaf(self, leaf):
        r_net = RNet(leaf)

        if self.net.nets is None:
            self.net.nets = r_net
        else:
            self.net.insert_nns_not_rec(r_net, self.find_diff)

    def nearest_neighbour(self, q):

        # print(f"nn search: {q}")
        nn_net, dist = self.net.approx_nns_not_rec(q, self.find_diff)

        self.last_net = nn_net

        return nn_net.value_pointer, dist

    def update_leaf(self, old_leaf, new_leaf):
        # this corresponds to the last net found, which should point at the old_leaf
        if np.equal(self.last_net.value, old_leaf.value).all() and np.equal(old_leaf.value, new_leaf.value).all():
            self.last_net.value_pointer = new_leaf

        self.last_net = None # clear the history?

class LeafList(LeafListBase):
    def __init__(self, find_diff) -> None:
        print("full search")
        self.leaf_list = []
        self.find_diff = find_diff

    def add_leaf(self, leaf):
        self.leaf_list.append(leaf)

    def nearest_neighbour(self, value,):
        min_diff = 1000000

        nearest_node = None
        y = np.array(value, dtype=np.float128)

        for leaf in self.leaf_list:
            x = np.array(leaf.value, dtype=np.float128)

            diff = self.find_diff(x, y)

            if diff < min_diff:
                min_diff = diff
                nearest_node = leaf

        # print(f"dist: {min_diff}, value: {nearest_node.value}")

        return nearest_node, min_diff

    def update_leaf(self, old_leaf, new_leaf):
        # check they are the same value (as otherwise don't want to mess with it?)
        if np.equal(old_leaf.value, new_leaf.value).all():
            # remove old leaf from list
            self.leaf_list.remove(old_leaf)
            self.add_leaf(new_leaf)
