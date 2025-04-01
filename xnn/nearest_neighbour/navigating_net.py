import numpy as np
import uuid

MIN_VALUE = 1e-200

class RNet:

    def __init__(self, value_pointer):
        self.value_pointer = value_pointer
        self.unique_id = uuid.uuid4()
        self.subnets = []
        self.parent_nets = []

    @property
    def value(self):
        return self.value_pointer.value

    def add_sub_net(self, net):
        self.subnets.append(net)
        net.add_parent_net(self)

    def add_parent_net(self, net):
        self.parent_nets.append(net)

class NavigatingNet:
    def __init__(self, r_max=0.5, epsilon=0):
        self.r_max = r_max
        self.epsilon = epsilon
        self.nets = None

    def insert_nns_not_rec(self, q, find_diff):
        r = self.r_max
        nets = self.nets

        if np.array_equal(q.value, nets.value):
            # already exists in net, don't need to insert it again
            return

        nets_to_search = []
        nets_to_search.append(nets)

        while nets_to_search:
            diameter = r # this is the subnet diameter, as the subnet always has radius r/2

            next_nets_to_search = []

            for net in nets_to_search:
                subnets_to_search = []
    
                if not net.subnets:
                    # insert below this net
                    net.subnets.append(q)
                else:
                    for subnet in net.subnets:
                        dist = find_diff(q.value, subnet.value)

                        if dist == 0:
                            # already exists in net, don't need to insert it again
                            return
                        
                        if dist < diameter:
                            subnets_to_search.append(subnet)

                    if subnets_to_search and (r/2) > MIN_VALUE:
                        next_nets_to_search = next_nets_to_search + subnets_to_search # add subnets into the search list
                    else:
                        # insert below this net
                        net.subnets.append(q)

            nets_to_search = next_nets_to_search

            r = r/2

        return

    def insert_nns(self, q, find_diff, nets=None, r=None):
        # max r value to start with
        if r is None:
            r = self.r_max
            nets = self.nets

        # starting at top level, don't need to check as it will fit in this level regardless? but do need to check if it's the same?
        if np.array_equal(q.value, nets.value):
            return

        if not nets.subnets:
            # no subnets in this level yet
            # new_net = rNet(q, r/2)
            nets.subnets.append(q)
            return
        else:
            # subnets exist
            subnets_in_range = []
            diameter = r # this is the subnet diameter, as the subnet always has radius r/2

            insert_to_this_level = True

            for net in nets.subnets:
                dist = find_diff(q.value, net.value)

                if dist == 0:
                    return

                if dist < diameter:
                    subnets_in_range.append(net)
                    self.insert_nns(q, find_diff, net, r/2)
                    insert_to_this_level = False

            if insert_to_this_level:
                nets.subnets.append(q)

            return

    def print_nets(self, nets=None, level=0):

        if nets is None:
            nets = self.nets

        print(f"\nr/{2**level}")

        if level == 0:
            print(f"{nets.value}, {nets.value_pointer}")
            level += 1
            return self.print_nets(nets.subnets, level)

        subnets = []

        for net in nets:
            print(f"{net.value}, {net.value_pointer}")

            if net.subnets:
                for subnet in net.subnets:
                    subnets.append(subnet)

        if subnets:
            level += 1
            self.print_nets(subnets, level)

    def approx_nns_not_rec(self, q, find_diff):
        Z_r = self.nets
        Z_r_dist = find_diff(q, Z_r.value)
        r = self.r_max

        level_nn = []

        approx_nn_found = False
        min_changed = True

        while not approx_nn_found:
            if min_changed:
                level_nn.append([Z_r_dist, Z_r])

            if not Z_r.subnets:
                # bottom of nets radius found
                break

            Z_r_2 = []

            if self.epsilon != 0:
                # if epsilon is 0, want to find the actual nearest neighbour, not the approx. one
                approx_diameter = 2*r*(1 + 1/self.epsilon)

                if approx_diameter <= Z_r_dist:
                    # this is the approx. nearest neighbour
                    approx_nn_found = True
                    break
            
            min_Z_r_2_net = Z_r
            min_Z_r_2_dist = Z_r_dist
            min_changed = False

            for net in Z_r.subnets:
                if net in Z_r_2:
                    # this net is already in the subnets list, don't need to readd it
                    continue
                else:
                    dist = find_diff(q, net.value) #np.linalg.norm(q.value - net.value)

                    if dist <= (Z_r_dist + r):
                        Z_r_2.append(net)

                        if dist < min_Z_r_2_dist:
                            min_Z_r_2_dist = dist
                            min_Z_r_2_net = net
                            min_changed = True

            if Z_r_2:
                r = r/2
                Z_r = min_Z_r_2_net
                Z_r_dist = min_Z_r_2_dist
            else:
                break

            if r < MIN_VALUE:
                # also break, as otherwise get issues with r becoming 0?
                break

        level_nn = np.array(level_nn)

        min_index = np.argmin(level_nn[:, 0])
        min_net = level_nn[min_index][1]
        min_distance = level_nn[min_index][0]

        return min_net, min_distance

    def approx_nns(self, q, find_diff):
        Z_r = self.nets
        Z_r_dist = find_diff(np.array(q), np.array(Z_r.value))

        return self._approx_nns(q, Z_r, self.r_max, Z_r_dist, find_diff)

    def _approx_nns(self, q, Z_r, r, Z_r_dist, find_diff):
        Z_r_2 = []

        if self.epsilon != 0:
            # if epsilon is 0, want to find the actual nearest neighbour, not the approx. one
            approx_diameter = 2*r*(1 + 1/self.epsilon)

            if approx_diameter <= Z_r_dist:
                # this is the approx. nearest neighbour
                return Z_r, Z_r_dist

        min_Z_r_2_net = Z_r
        min_Z_r_2_dist = Z_r_dist

        for net in Z_r.subnets:
            if net in Z_r_2:
                # this net is already in the subnets list, don't need to readd it
                continue
            else:
                dist = find_diff(np.array(q), np.array(net.value)) #np.linalg.norm(q.value - net.value)

                if dist <= (Z_r_dist + r):
                    Z_r_2.append(net)

                    if dist < min_Z_r_2_dist:
                        min_Z_r_2_dist = dist
                        min_Z_r_2_net = net

        if Z_r_2:
            r = r/2
            return self._approx_nns(q, min_Z_r_2_net, r, min_Z_r_2_dist, find_diff)
        else:
            return min_Z_r_2_net, min_Z_r_2_dist
