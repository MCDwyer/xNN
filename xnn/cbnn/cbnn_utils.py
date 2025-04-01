"""functions required for the CBNN implementation"""
import numpy as np

from .trees.tree_classes import BinaryTreeZ, BinaryTreeContraction
from .trees.tree_utils import bt_find_furthest_child, find_least_common_ancestor

from .tt_rebalancing import rebalance
from .cbnn_belief_prop import update_big_omega, update_psi, update_omega

np.random.seed(42)

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
MAX_VALUE = 1e50
MIN_VALUE = 1e-50

def calculate_nu(tt, u, u_prime):
    # E is ternary tree of given binary tree
    E = tt

    if u.get_id() == u_prime.get_id():
        return UP

    # find u and u_prime on E
    s_tilde = u.s
    s_tilde_prime = u_prime.s

    s_tilde_path = s_tilde.get_path()
    s_tilde_prime_path = s_tilde_prime.get_path()

    # find least common ancestor between the two
    # least common ancestor is the deepest/lowest/furthest down the tree vertex
    # which has both as a descendant
    s_star, _, s_star_index = find_least_common_ancestor(E, s_tilde_path, s_tilde_prime_path)

    s_hat = None

    # s_tilde_path[s_star_index] gives the direction that the lca came from, need to add one to the 
    # index to look at s_star's subtrees
    reduced_s_tilde_path = s_tilde_path[(s_star_index + 1):]
    reduced_s_tilde_prime_path = s_tilde_prime_path[(s_star_index + 1):]

    if reduced_s_tilde_path[0] == CENTRE:
        s_hat = s_star.centre
    else:
        return UP

    # don't need all these comparisons as if it isn't the centre child we return UP, only keeping
    # here for now so I remember why I removed it
    # if s_tilde_path[s_star_index] == "L":
    #     s_hat = s_star.left
    # elif s_tilde_path[s_star_index] == "C":
    #     s_hat = s_star.centre
    # elif s_tilde_path[s_star_index] == "R":
    #     s_hat = s_star.right

    # if s_hat != s_star.centre:
    #     return UP

    # find internal vertex on bt that corresponds to s_star
    xi_s_star = s_star.u

    if xi_s_star.get_id() == u.get_id():
        if reduced_s_tilde_prime_path[0] == LEFT:
            return LEFT
        elif reduced_s_tilde_prime_path[0] == RIGHT:
            return RIGHT

    # combined this with the s_hat and s_hat_prime assignments
    # # if the associated internal vertex of s* on J is = to u and s^' and is = to left child of s*?
    # # xi function looks at binary tree J?
    # if xi(s_star) == u:
    #     if s_hat_prime == s_star.left:
    #         return LEFT
    #     elif s_hat_prime == s_star.right:
    #         return RIGHT

    s = s_hat # so s = s_star.centre

    for i in range(len(reduced_s_tilde_path)):
        xi_s = s.u # find internal vertex on bt that corresponds to s
        # The vertex set of D is partitioned into two sets D◦ and D• where each vertex s ∈ D is associated with a 
        # vertex μ(s) ∈ J and every s ∈ D• is also associated with a vertex μ′(s) ∈ ⇓(μ(s))† 
        # E◦ is the set of nodes corresponding to segments without terminal nodes (N nodes in the technical report) 
        # whereas E• is the set of nodes corresponding to segments with terminal nodes (L and R nodes in the technical report)

        # if s.node_label == "N" then in E◦
        if s.node_label is None or s.node_label == TNodeN: # E partitioned into 2 sets, E_empty_circle and E_full_circle, so if it's not in one it's in the other?
            return UP
        elif u.get_id() == xi_s.get_id():
            # if != "N" then in E•
            if s.node_label == TNodeL:
                return LEFT
            elif s.node_label == TNodeR:
                return RIGHT

        # This is encapsulated in the elif u==xi_s and following statements above now
        # elif u == xi_s and s.right in set of E_full_circle:
        #     return LEFT
        # elif u == xi(s) and s.left in set of E_full_circle:
        #     return RIGHT

        # as I have path info, don't need to do this check like this, can just use the if statements below
        # for s_prime in [s.left, s.centre, s.right]:
        #     if s_tilde in subtree of s_prime:
        #         s = s_prime
        direction = reduced_s_tilde_path[i+1] # 1st position in reduced_s_tilde_path == "C" so s is already set to this from above

        if direction == LEFT:
            s = s.left
        elif direction == CENTRE:
            s = s.centre
        elif direction == RIGHT:
            s = s.right

def calculate_phi(epsilon, j):
    # phi_0(epsilon) := 0
    # phi_{j+1}(epsilon) := (1 - epsilon)phi_j(epsilon) + epsilon(1 - phi_j(epsilon))

    phi = 0

    if j < 0:
        # Think this is okay to occur where I know the contraction tree is being formed correctly now? maybe?
        j = -1 * j

    if j == 0:
        return 0
    else:
        phi_j = calculate_phi(epsilon, (j-1))
        phi += (1 - epsilon)*phi_j + epsilon*(1 - phi_j)

        return phi
        
def initialise_tau(u, u_parent, T):

    delta_u = u.d - u_parent.d

    u.tau = np.zeros((2,2))

    epsilon = 1/T
        
    phi_u = calculate_phi(epsilon, delta_u)

    for a in [0, 1]:
        for b in [0, 1]:
            if a == b:
                u.tau[a][b] = 1 - phi_u
            else:
                u.tau[a][b] = phi_u

    return u
    
def marginal(tt_H_J, path):

    Lambda = 0

    w = _marginal(tt_H_J, path)

    Lambda = w[1]

    theta = (1/(4)) * Lambda

    return theta

def _marginal(s, path):

    s.w = update_omega(s)

    if len(path) == 0:
        # at the relevant node
        return s.w

    if path[0] == LEFT:
        # go left
        w = _marginal(s.left, path[1:])
    elif path[0] == CENTRE:
        # go centre
        w = _marginal(s.centre, path[1:])
    elif path[0] == RIGHT:
        # go right
        w = _marginal(s.right, path[1:])

    return w

def evidence(s, path):
    # update all the values of psi and omegas on the path to u?

    if len(path) > 0:
        if path[0] == LEFT:
            evidence(s.left, path[1:])
        elif path[0] == CENTRE:
            evidence(s.centre, path[1:])
        elif path[0] == RIGHT:
            evidence(s.right, path[1:])

    if s.node_label is not None:
        for s_child in [s.left, s.centre, s.right]:
            s_child.psi = update_psi(s_child)
            s_child.big_omega = update_big_omega(s_child)

    s.psi = update_psi(s)
    s.big_omega = update_big_omega(s)

    return s

def grow(model, bt_Z, tt_H_Z, leaf_list, x, nn=None, is_hnn=False):
    # bt is full binary tree of contexts (formed by designating n(x_t) as parent of x_t)
    # tt is balanced ternary tree of Z (bt)
    # u = unique leaf of n(x_t) from Z?

    # tree before grow:
    # 0    u*       
    # 1  a   u

    # tree after grow:
    # 0    u*       
    # 1  a   u'   
    # 2 _ _ u'' u

    # 0    u*       
    # 1  a   n(x)   
    # 2 _ _ x n(x)
    # TODO: work out how to save the tree (keeping the ID values) whilst not breaking this, basically need to make a copy for this tree function? that copies and then sets the bt_id to None in case it's been copied before?

    if bt_Z is None:
        bt_Z = BinaryTreeZ(x)
        leaf_list.add_leaf(bt_Z)

        return bt_Z, None, leaf_list, bt_Z, (None, None)
    elif bt_Z.left is None:
        # Only root node on tree, only need to check left or right as we always add to
        # both to make full tree?
        
        # u* = u_prime in this case
        u = bt_Z

        u_prime = BinaryTreeZ(u.value)
        u_prime2 = BinaryTreeZ(x)

        u_prime.left = u_prime2
        u_prime.right = u

        u_prime.update_height()

        u_prime.d = 0 # where d(u) of root is 0 by definition from paper?
        u.d = 0
        u_prime2.d = 1

        path = []

        leaf_list.add_leaf(u_prime2)

        tt_H_Z, _ = rebalance(tt_H_Z, u_prime, path)
        return u_prime, tt_H_Z, leaf_list, u_prime2, (u, 0)

    # so find nearest neighbour of x_t on Z
    # and set u = to that vertex
    if nn is None:
        u, diff = leaf_list.nearest_neighbour(x)
    else:
        u, diff = nn

    u_star = u.parent # u* = parent of n(x_t) vertex (u)

    # u_prime and u_prime2 are new vertices
    u_prime = BinaryTreeZ(u.value) # new vertex that will go between u and u_star
    u_prime2 = BinaryTreeZ(x) # new vertex with x_t value?

    # u_prime will be set as left or right child of u*, depending on the position n(x_t) was
    if u_star.left.get_id() == u.get_id():
        u_star.left = u_prime
    else:
        u_star.right = u_prime

    u_prime.left = u_prime2
    u_prime.right = u

    # d(u') = d(u)
    # d(u'') = d(u) + 1
    u_prime.d = u.d
    u_prime2.d = u.d + 1

    # u_prime.update_height()

    if not is_hnn: # maybe?
        if diff > 0:
            leaf_list.add_leaf(u_prime2)
        elif diff == 0:
            # then essentially binning will have occurred I think? 
            # so want to update it to point the the newest leaf? to prevent the recursion errors occurring??
            leaf_list.update_leaf(u, u_prime2)

    s_path = u.s.get_path()

    # add in u'' (which is the x_t) and then rebalance TST of Z?
    tt_H_Z, _ = rebalance(tt_H_Z, u_prime, s_path)

    return bt_Z, tt_H_Z, leaf_list, u_prime2, (u, diff)

def insert(model, bt_Z, tt_H_Z, bt_J, tt_H_J, contraction_id, u):
    # J is contraction of Z
    # Z = full binary tree
    # E = TST of Z
    # D = TST of J
    # s = root of D
    # x = new context which needs to be added to contraction (already exists on Z and H_Z)

    # throughout the insert function I use copy_bt_vertex to create a new binary tree
    # vertex for J but links the id part to the Z tree vertex it corresponds to
    # I've done it this way as otherwise if I just point to the same item in memory then changes on 
    # J will affect Z and vice versa, so think this is the best way to do it?

    # could also be other way round (i.e. u^ on right originally)
    # full:
    # binary:
    #     u'   
    #   u*  r
    # u^ u_t 

    # contraction
    # before insert:
    # binary tree:
    #    u'   
    #  u^  r 
    # ternary tree:
    #    N     
    # u^ u' r 
    # post insert:
    # binary tree:
    #     u'   
    #   u*  r
    # u^ u_t 
    # ternary tree:
    #          N     
    #    N     u'    r
    # u^ u* u_t 

    if bt_J is None:
        # initialise the contraction trees
        bt_J = BinaryTreeContraction(bt_Z, contraction_id)

    if bt_J.height == 1: # only root node so need to initialise the tree with left and right children
        # left child should be furthest left child on Z, right child should be furthest right child on Z
        left_child = bt_find_furthest_child(bt_Z, find_left=True)
        right_child = bt_find_furthest_child(bt_Z, find_left=False)

        bt_J.left = BinaryTreeContraction(left_child, contraction_id)
        bt_J.right = BinaryTreeContraction(right_child, contraction_id)

        bt_J = initialise_tau(bt_J, bt_J, model.T)
        bt_J.left = initialise_tau(bt_J.left, bt_J, model.T)
        bt_J.right = initialise_tau(bt_J.right, bt_J, model.T)

        path = []

        tt_H_J, inserted_s = rebalance(tt_H_J, bt_J, path) # this will initialise the ternary tree of J

        bt_J.update_height()

        # TODO: check this is the correct thing to do?
        if bt_J.left.get_id() == u.get_id():
            return bt_J, tt_H_J, tt_H_J.left

        if bt_J.right.get_id() == u.get_id():
            return bt_J, tt_H_J, tt_H_J.right

    # now the contraction trees are initialised we insert the x vertex into the contraction
    E = tt_H_Z
    D = tt_H_J
    s = D # s = root of TST of J (D)

    btZ_u = u
    btJ_u = BinaryTreeContraction(u, contraction_id)

    internal = True

    s_hat_path = []

    while internal:
        # find the associated internal vertex of s on the tree J?
        xi_s = s.u.bt_Z_node # as finding nu on bt_Z
        nu = calculate_nu(E, xi_s, btZ_u)

        if nu == LEFT:
            s = s.left
            s_hat_path.append(LEFT)
        elif nu == UP:
            s = s.centre
            s_hat_path.append(CENTRE)
        elif nu == RIGHT:
            s = s.right
            s_hat_path.append(RIGHT)

        if s.left is None and s.centre is None and s.right is None:
            # s is a leaf node, so finish loop
            internal = False
            break

    # so after the loop, s will be the internal node on the ternary tree D which
    # corresponds to the u^ node on J?
    # each vertex s ∈ D is associated with a vertex μ(s) ∈ J
    btZ_u_hat = s.u.bt_Z_node
    u_hat = btZ_u_hat.get_contraction(contraction_id) # so u_hat is vertex associated with s on J

    s = E # root of full TST of Z (E)
    internal = True
    s_star_path = []

    # now we have u^, need to find the least common ancestor of u^ and u_t on the Z tree, u*
    # loop through all the internal nodes of E (not leaves)
    while internal:
        xi_s = s.u

        nu = calculate_nu(E, xi_s, btZ_u)
        nu_hat = calculate_nu(E, xi_s, btZ_u_hat)

        if nu == nu_hat:
            if nu == LEFT:
                s = s.left
                s_star_path.append(LEFT)
            elif nu == UP:
                s = s.centre
                s_star_path.append(CENTRE)
            elif nu == RIGHT:
                s = s.right
                s_star_path.append(RIGHT)
        else:
            s = s.centre
            s_star_path.append(CENTRE)

        if s.left is None and s.centre is None and s.right is None:
            # s is a leaf node, so finish loop
            internal = False
            break
    # now s is the internal node on the full ternary tree E (which is the ternary of the binary Z)
    # which corresponds to u* on the binary tree
    # where u* is the least common ancestor of u^ and u on Z
    btZ_u_star = s.u # find u_star on Z?
    u_star = BinaryTreeContraction(btZ_u_star, contraction_id) # make a new vertex to use on the contraction binary tree

    u_prime = u_hat.parent # find the current parent of u_hat on J

    # find the position to add in u_hat to u_star, by finding the relative position of u_hat to u_star on the full tree Z
    nu = calculate_nu(E, btZ_u_star, btZ_u_hat)

    bt_J, D, s = update_contractions(bt_J, D, btJ_u, u_hat, u_prime, u_star, nu, model.T, s_hat_path)

    return bt_J, D, s

def update_contractions(bt_J, tt_H_J, u, u_hat, u_prime, u_star, nu, T, s_hat_path):
    # add in u_star to J (by replacing the u^ child of u_prime)
    u_hat_parent_dir = u_hat.get_parent_direction()

    if u_hat_parent_dir == LEFT:
        u_prime.left = u_star
    else:
        u_prime.right = u_star

    # if nu == LEFT then that means u_hat is left relative to u on tree Z
    # so now we know the relative position of u_hat to z, we set the relevant child of u* to u_hat and the remaining child is set to u
    if nu == LEFT:
        u_star.left = u_hat
        u_star.right = u
    else:
        u_star.right = u_hat
        u_star.left = u

    # update the height of u_star now that it has children
    # u_hat height should remain the same and u height is 1
    # u_star.update_height() # this will recursively update the heights up the entire tree path, though I don't even use the heights anywhere? so might not bother updating the heights to save recursing up the full tree

    # should be initialised to 1 already so don't think I need to do this?
    # for i in [0, 1] -> kappa[i] = 1?
    u_star.kappa = np.ones(2)

    # so at this point, we initialise/update the transition probabilities/tau values
    u_star.tau = np.zeros((2, 2))
    u_hat.tau = np.zeros((2, 2))
    u.tau = np.zeros((2, 2))

    u_star = initialise_tau(u_star, u_prime, T)
    u_hat = initialise_tau(u_hat, u_star, T)
    u = initialise_tau(u, u_star, T)

    tt_H_J, inserted_s = rebalance(tt_H_J, u_star, s_hat_path)

    s = inserted_s.right if nu == LEFT else inserted_s.left

    return bt_J, tt_H_J, s

def binning(model, leaf_list, x):

    u, diff = leaf_list.nearest_neighbour(x)

    if diff <= model.binning_radius:
        return u.value, (u, diff)

    return x, (u, diff)

def recreate_cbnn_from_history(model, bt_B, bt_Z, tt_H_Z, leaf_list, history_for_saving):

    nodes_for_nn_lookup = {}

    for entry in history_for_saving:
        x = entry["x"]
        nn_x = entry["nn_value"]
        nn_diff = entry["nn_diff"]
        pi_sum = entry["pi_sum"]
        loss = entry["loss"]
        pi_values = entry["pi_values"]
        path = entry["path_through_bt_B"]

        # need to grow
        if bt_Z is not None and model.binning_radius != 0:
            binned_x = binning(model, leaf_list, x)
        else:
            binned_x = x

        if nn_x is None:
            bt_Z, tt_H_Z, leaf_list, u, nn = grow(model, bt_Z, tt_H_Z, leaf_list, binned_x)
            continue
        else:
            nn_u = nodes_for_nn_lookup[nn_x]
            nn = (nn_u, nn_diff)
            bt_Z, tt_H_Z, leaf_list, u, nn = grow(model, bt_Z, tt_H_Z, leaf_list, binned_x, nn)

        # add node to lookup for use as future nearest neighbours
        nodes_for_nn_lookup[u.value] = u

        if bt_Z.height == 1:
            continue

        psi = calculate_psi(model, loss, pi_sum)
    
        # rebuild and propogate beliefs on model
        bt_B, _ = _cbnn_recreate_model(model, bt_Z, tt_H_Z, bt_B, u, path, pi_values, psi)

    return bt_B, bt_Z, tt_H_Z, leaf_list

def _cbnn_recreate_model(model, bt_Z, tt_H_Z, v, u, path, pi_values, psi):

    if len(path) == 0:
        # print("bottom of tree\n")
        # bottom of tree reached, i.e. action node so begin going back up tree updating the belief values
        return v, psi
    
    pi = pi_values[0]
    pi_values.pop(0)

    left_id = v.left.unique_id
    right_id = v.right.unique_id

    v.left.A_v, v.left.H_A_v, v_left_s = insert(model, bt_Z, tt_H_Z, v.left.A_v, v.left.H_A_v, left_id, u)
    v.right.A_v, v.right.H_A_v, v_right_s = insert(model, bt_Z, tt_H_Z, v.right.A_v, v.right.H_A_v, right_id, u)

    # next node to go to is defined by path[0]
    if path[0] == LEFT:
        # go left
        path.pop(0)

        v.left, psi = _cbnn_recreate_model(model, bt_Z, tt_H_Z, v.left, u, path, pi_values, psi)
        
        v_t = v.left
        v_t_u = v_left_s.u
        v_t_s_path = v_left_s.get_path()
        v_tilde = v.right
        v_tilde_u = v_right_s.u
        v_tilde_s_path = v_right_s.get_path()

    else:
        path.pop(0)

        v.right, psi = _cbnn_recreate_model(model, bt_Z, tt_H_Z, v.right, u, path, pi_values, psi)

        v_t = v.right
        v_t_u = v_right_s.u
        v_t_s_path = v_right_s.get_path()
        v_tilde = v.left
        v_tilde_u = v_left_s
        v_tilde_s_path = v_left_s.get_path()

    psi = update_belief(psi, pi, v_t, v_t_u, v_t_s_path, v_tilde, v_tilde_u, v_tilde_s_path)

    return v, psi

def cbnn_grow_and_find_action(model, bt_B, bt_Z, tt_H_Z, leaf_list, x):
    if bt_Z is not None and model.binning_radius != 0:
        binned_x = binning(model, leaf_list, x)
    else:
        binned_x = x

    bt_Z, tt_H_Z, leaf_list, u, nn = grow(model, bt_Z, tt_H_Z, leaf_list, binned_x)

    # TODO: work out how to initiate better, can I duplicate the first entry maybe?
    if bt_Z.height == 1:
        create_history_for_saving(model, u, (None, None), None, None, None, None)
        return cbnn_grow_and_find_action(model, bt_B, bt_Z, tt_H_Z, leaf_list, x)

    # find the action
    bt_B, action, history_for_saving, pi_sum = cbnn_find_action(model, bt_Z, tt_H_Z, bt_B, u)

    return action, pi_sum, history_for_saving, u, nn, bt_B, bt_Z, tt_H_Z, leaf_list

def calculate_psi(model, loss, pi_sum):
    psi = np.exp((-model.eta*loss)/pi_sum)
    psi = np.clip(psi, MIN_VALUE, MAX_VALUE)

    return psi

def create_history_for_saving(model, u, nn, action, pi_sum, history_for_saving, loss):
    if history_for_saving is not None:
        nn_u, nn_diff = nn
        history_entry_for_saving = {"x": u.value, "nn_value": nn_u.value, "nn_diff": nn_diff, "action": action, "pi_sum": pi_sum, "loss": loss, "pi_values": history_for_saving["pi_values"], "path_through_bt_B": history_for_saving["path"]}
    else:
        history_entry_for_saving = {"x": u.value, "nn_value": None, "nn_diff": None, "action": None, "pi_sum": None, "loss": None, "pi_values": None, "path_through_bt_B": None}
    
    model.history_for_saving_and_loading.append(history_entry_for_saving)

    return

def _cbnn_find_action(model, bt_Z, tt_H_Z, v, u, history_for_saving, pi_sum=1):

    if v is None:
        # tree not passed in?
        print("tree not passed in")
        return None

    if v.left is None and v.right is None:
        # print(v.value)
        # print("bottom of tree\n")
        
        action = v.value

        return v, action, history_for_saving, pi_sum
    else:
        # not sure I actually could do that, because after the insert it will rebalance so will change the direction it came from potentially?
        left_id = v.left.unique_id
        right_id = v.right.unique_id
        # print(f"left: {v.left.unique_id}, right: {v.right.unique_id}")
        v.left.A_v, v.left.H_A_v, v_left_s = insert(model, bt_Z, tt_H_Z, v.left.A_v, v.left.H_A_v, left_id, u)
        v.right.A_v, v.right.H_A_v, v_right_s = insert(model, bt_Z, tt_H_Z, v.right.A_v, v.right.H_A_v, right_id, u)

        v_left_s_path = v_left_s.get_path()
        v_right_s_path = v_right_s.get_path()

        theta_left = marginal(v.left.H_A_v, v_left_s_path)
        theta_right = marginal(v.right.H_A_v, v_right_s_path)

        # TODO: every so often I get division by zero warning here, not sure why?
        z = theta_left + theta_right

        pi_left = theta_left/z
        pi_right = theta_right/z # could only calculate this when needed?

        zeta = np.random.uniform(0, 1)

        if zeta <= pi_left:
            # print("going left down tree")            
            history_for_saving["path"].append(LEFT)
            history_for_saving["pi_values"].append(pi_left)

            pi_sum = pi_sum * pi_left
            v.left, action, _, pi_sum = _cbnn_find_action(model, bt_Z, tt_H_Z, v.left, u, history_for_saving, pi_sum)
            
            v_t = v.left
            v_t_u = v_left_s.u
            v_t_s_path = v_left_s.get_path()
            v_tilde = v.right
            v_tilde_u = v_right_s
            v_tilde_s_path = v_right_s.get_path()

        else:
            # print("going right down tree")
            history_for_saving["path"].append(RIGHT)
            history_for_saving["pi_values"].append(pi_right)

            pi_sum = pi_sum * pi_right
            v.right, action, _, pi_sum = _cbnn_find_action(model, bt_Z, tt_H_Z, v.right, u, history_for_saving, pi_sum)

            # set for history storing
            v_t = v.right
            v_t_u = v_right_s.u
            v_t_s_path = v_right_s.get_path()
            v_tilde = v.left
            v_tilde_u = v_left_s
            v_tilde_s_path = v_left_s.get_path()

        # update with loss = 0 to help keep the tree and memory structures??
        v_t_u.kappa[1] = 1
        v_tilde_u.kappa[1] = 1

        v_t.H_A_v = evidence(v_t.H_A_v, v_t_s_path)
        v_tilde.H_A_v = evidence(v_tilde.H_A_v, v_tilde_s_path)

        return v, action, history_for_saving, pi_sum

def cbnn_find_action(model, bt_Z, tt_H_Z, bt_B, u):

    pi_sum = 1

    history_for_saving = {"pi_values": [], "path": []}
    # print("\nstarting traversal")
    bt_B, action, history_for_saving, pi_sum = _cbnn_find_action(model, bt_Z, tt_H_Z, bt_B, u, history_for_saving, pi_sum)

    return bt_B, action, history_for_saving, pi_sum

def cbnn_learn_from_loss(model, bt_B, u, path, pi_values, loss, pi_sum):

    psi = calculate_psi(model, loss, pi_sum)

    bt_B, _ = _cbnn_learn_from_loss(bt_B, u, path, pi_values, psi)

    return bt_B

def _cbnn_learn_from_loss(v, u, path, pi_values, psi):

    if len(path) == 0:
        # print("bottom of tree\n")
        # bottom of tree reached, i.e. action node so begin going back up tree updating the belief values
        return v, psi
    
    pi = pi_values[0]
    pi_values.pop(0)

    left_id = v.left.unique_id
    right_id = v.right.unique_id

    v_left_u = u.get_contraction(left_id)
    v_right_u = u.get_contraction(right_id)

    # next node to go to is defined by path[0]
    if path[0] == LEFT:
        # print("going left")
        # go left
        path.pop(0)

        v.left, psi = _cbnn_learn_from_loss(v.left, u, path, pi_values, psi)
        
        v_t = v.left
        v_t_u = v_left_u
        v_t_s_path = v_left_u.s.get_path()
        v_tilde = v.right
        v_tilde_u = v_right_u
        v_tilde_s_path = v_right_u.s.get_path()

    else:
        # print("going right")
        path.pop(0)

        v.right, psi = _cbnn_learn_from_loss(v.right, u, path, pi_values, psi)

        # set for history storing
        v_t = v.right
        v_t_u = v_right_u
        v_t_s_path = v_right_u.s.get_path()
        v_tilde = v.left
        v_tilde_u = v_left_u
        v_tilde_s_path = v_left_u.s.get_path()

    # print("updating beliefs")
    # only get here when climbing back up the tree, so can start updating the belief values
    psi = update_belief(psi, pi, v_t, v_t_u, v_t_s_path, v_tilde, v_tilde_u, v_tilde_s_path)
    # psi_parent = 1 - (1 - psi)*pi # calculate psi_(j-1) using psi(j)

    # psi_parent = np.clip(psi_parent, MIN_VALUE, MAX_VALUE)

    # v_t_u.kappa[1] = psi/psi_parent
    # v_tilde_u.kappa[1] = 1/psi_parent

    # psi = psi_parent # set psi to psi_parent for next iteration back up tree?

    # v_t.H_A_v = evidence(v_t.H_A_v, v_t_s_path)
    # v_tilde.H_A_v = evidence(v_tilde.H_A_v, v_tilde_s_path)

    return v, psi

def update_belief(psi, pi, v_t, v_t_u, v_t_s_path, v_tilde, v_tilde_u, v_tilde_s_path):
    psi_parent = 1 - (1 - psi)*pi # calculate psi_(j-1) using psi(j)

    psi_parent = np.clip(psi_parent, MIN_VALUE, MAX_VALUE)

    v_t_u.kappa[1] = psi/psi_parent
    v_tilde_u.kappa[1] = 1/psi_parent

    psi = psi_parent # set psi to psi_parent for next iteration back up tree?

    v_t.H_A_v = evidence(v_t.H_A_v, v_t_s_path)
    v_tilde.H_A_v = evidence(v_tilde.H_A_v, v_tilde_s_path)

    return psi

def cbnn(model, bt_B, bt_Z, tt_H_Z, leaf_list, x, label=None):

    if bt_Z is not None and model.binning_radius != 0:
        binned_x, nn = binning(model, leaf_list, x)
    else:
        binned_x = x
        nn = None

    bt_Z, tt_H_Z, leaf_list, u, nn = grow(model, bt_Z, tt_H_Z, leaf_list, binned_x, nn)

    if bt_Z.height == 1:
        return cbnn(model, bt_B, bt_Z, tt_H_Z, leaf_list, x, label)

    bt_B, _, loss, action = _cbnn(model, bt_Z, tt_H_Z, bt_B, u, x, label)

    return bt_B, bt_Z, tt_H_Z, leaf_list, action, loss

def _cbnn(model, bt_Z, tt_H_Z, v, u, x, label=None, pi_sum=1):
    # initially v is equal to the root of the binary tree B
    loss = None

    if v is None:
        # tree not passed in?
        print("tree not passed in")
        return None

    if v.left is None and v.right is None:
        # leaf node reached?
        a = v.value # leaf nodes of binary tree B are the actions to take
        # perform action and recieve loss
        # loss should be within 0 and 1
        loss = model.action_function(x, a, label)

        psi = np.exp((-model.eta*loss)/pi_sum)
        psi = np.clip(psi, MIN_VALUE, MAX_VALUE)

        return v, psi, loss, a # if I can do the action here, don't need to pass the action back?
    else:
        # not sure I actually could do that, because after the insert it will rebalance so will change the direction it came from potentially?
        left_id = v.left.unique_id
        right_id = v.right.unique_id
        v.left.A_v, v.left.H_A_v, v_left_s = insert(model, bt_Z, tt_H_Z, v.left.A_v, v.left.H_A_v, left_id, u)
        v.right.A_v, v.right.H_A_v, v_right_s = insert(model, bt_Z, tt_H_Z, v.right.A_v, v.right.H_A_v, right_id, u)

        v_left_s_path = v_left_s.get_path()
        v_right_s_path = v_right_s.get_path()

        theta_left = marginal(v.left.H_A_v, v_left_s_path)
        theta_right = marginal(v.right.H_A_v, v_right_s_path)

        # TODO: every so often I get division by zero warning here, not sure why?
        z = theta_left + theta_right

        pi_left = theta_left/z
        pi_right = theta_right/z # could only calculate this when needed?

        zeta = np.random.uniform(0, 1)
        went_left = False

        if zeta <= pi_left:
            pi_sum = pi_sum * pi_left
            v.left, psi, loss, a = _cbnn(model, bt_Z, tt_H_Z, v.left, u, x, label, pi_sum)

            pi = pi_left
            went_left = True
        else:
            pi_sum = pi_sum * pi_right
            v.right, psi, loss, a = _cbnn(model, bt_Z, tt_H_Z, v.right, u, x, label, pi_sum)

            pi = pi_right

        # should only get here once leaf node has been reached and action has been done
        psi_parent = 1 - (1 - psi)*pi # calculate psi_(j-1) using psi(j)

        psi_parent = np.clip(psi_parent, MIN_VALUE, MAX_VALUE)

        v_left_u = v_left_s.u
        v_right_u = v_right_s.u

        if went_left:
            v_t = v.left
            # v_t_s = v_left_s
            v_t_s_path = v_left_s_path
            v_t_u = v_left_u
            v_tilde = v.right
            # v_tilde_s = v_right_s
            v_tilde_s_path = v_right_s_path
            v_tilde_u = v_right_u
        else:
            v_t = v.right
            # v_t_s = v_right_s
            v_t_s_path = v_right_s_path
            v_t_u = v_right_u
            v_tilde = v.left
            # v_tilde_s = v_left_s
            v_tilde_s_path = v_left_s_path
            v_tilde_u = v_left_u

        v_t_u.kappa[1] = psi/psi_parent
        v_tilde_u.kappa[1] = 1/psi_parent

        psi = psi_parent # set psi to psi_parent for next iteration back up tree?

        v_t.H_A_v = evidence(v_t.H_A_v, v_t_s_path)
        v_tilde.H_A_v = evidence(v_tilde.H_A_v, v_tilde_s_path)
        
        return v, psi, loss, a

def cbnn_for_hnn(model, bt_B, bt_Z, tt_H_Z, leaf_list, s, nn, x, label):
    
    bt_Z, tt_H_Z, leaf_list, u, nn = grow(model, bt_Z, tt_H_Z, leaf_list, s, nn, is_hnn=True)

    if bt_Z.height == 1:
        return cbnn_for_hnn(model, bt_B, bt_Z, tt_H_Z, leaf_list, s, nn, x, label)

    bt_B, _, loss, action = _cbnn(model, bt_Z, tt_H_Z, bt_B, u, x, label)

    # info = [loss, action, label]

    return bt_B, bt_Z, tt_H_Z, leaf_list, u, action, loss
