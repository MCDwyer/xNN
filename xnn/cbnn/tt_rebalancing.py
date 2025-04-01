import numpy as np

from .cbnn_belief_prop import update_psi, update_big_omega, update_bp_values_on_segments_leaves
from .trees.tree_classes import TernaryTree

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

DEBUG = False

def centre_to_left(a):
    # b has label "L"
    # Before rotation tree:
    # 0                  a                 
    # 1     a_l          b          a_r     
    # 2 ___ ___ ___ b_l b_c b_r ___ ___ ___  
    # Post rotation tree:
    # 0                  b                 
    # 1      a          b_c         b_r     
    # 2 a_l b_l a_r ___ ___ ___ ___ ___ ___

    # a_l = a.left
    b = a.centre
    # a_r = a.right
    b_l = b.left
    # b_c = b.centre
    # b_r = b.right

    a.centre = b_l
    b.left = a

    a.update_all()
    b.update_all()

    if DEBUG:
        print(a.node_label)
        print(b.node_label)

    return b

def centre_to_right(a):
    # b has label "R"
    # Before rotation tree:
    # 0                  a                 
    # 1     a_l          b          a_r     
    # 2 ___ ___ ___ b_l b_c b_r ___ ___ ___  
    # Post rotation tree:
    # 0                  b                 
    # 1     b_l         b_c          a     
    # 2 ___ ___ ___ ___ ___ ___ a_l b_r a_r 

    # a_l = a.left
    b = a.centre
    # a_r = a.right
    # b_l = b.left
    # b_c = b.centre
    b_r = b.right

    a.centre = b_r
    b.right = a

    a.update_all()
    b.update_all()

    return b

def right_to_centre(a):
    # Before rotation tree:
    # 0                  a                 
    # 1     a_l         a_c           b     
    # 2 ___ ___ ___  ___ ___ ___ b_l b_c b_r
    # Post rotation tree:
    # 0                  b                 
    # 1     b_l          a          b_r     
    # 2 ___ ___ ___ a_l a_c b_c ___ ___ ___  

    # a_l = a.left
    # a_c = a.centre
    b = a.right
    # b_l = b.left
    b_c = b.centre
    # b_r = b.right

    a.right = b_c
    b.centre = a

    a.update_all()
    b.update_all()

    return b

def left_to_centre(a):
    # Before rotation tree:
    # 0                  a                 
    # 1      b          a_c          a_r     
    # 2 b_l b_c b_r ___ ___ ___  ___ ___ ___ 
    # Post rotation tree:
    # 0                  b                 
    # 1     b_l          a          b_r     
    # 2 ___ ___ ___ b_c a_c a_r ___ ___ ___  

    # a_r = a.right
    # a_c = a.centre
    b = a.left
    # b_l = b.left
    b_c = b.centre
    # b_r = b.right

    a.left = b_c
    b.centre = a

    a.update_all()
    b.update_all()

    return b

def tt_dynamic_rebalance(a):
    """ Dynamic rebalance (Algorithm 3) from 'MATHEMATICAL ENGINEERING TECHNICAL REPORTS Balanced 
    Ternary-Tree Representation of Binary Trees and Balancing Algorithms'

    'a' is node we're looking at for the rebalancing to follow paper notation
    'a' is ternary tree object with node_label, leaf_label, height, and height_prime attributes

    At this point the segment of the ternary tree a has been modified in some way further down the 
    tree.
    In CBNN we only ever do insertions into the tree.
    """
    double_rotation = False

    # is this a leaf?
    # if not left_child and not centre_child and not right_child:
    if a.node_label is None:
        # this is a leaf, so can't do any rebalancing here
        # print("leaf")
        return a, double_rotation

    al_height = a.left.get_height_prime()
    ac_height = a.centre.get_height_prime()
    ar_height = a.right.get_height_prime()

    if ac_height > (1 + np.max([al_height, ar_height])):
        # apply single rotation, from centre to left/right

        if a.centre.node_label == TNodeL:
            # centre to left?
            if DEBUG:
                print("single rotation from centre to left")
                # debug_prints(tt=a, print_label=True)

            a = centre_to_left(a)
        elif a.centre.node_label == TNodeR:
            # centre to right?
            if DEBUG:
                print("single rotation from centre to right")
                # debug_prints(tt=a, print_label=True)

            a = centre_to_right(a)
        else:
            print("centre to left/right not happening")
            print(a.centre.node_label)
            # debug_prints(tt=a, print_label=True)

    al_height = a.left.get_height_prime()
    ac_height = a.centre.get_height_prime()
    ar_height = a.right.get_height_prime()
    # other rotations
    if a.node_label != TNodeR and al_height > 1 + np.max([ac_height, ar_height]):
        alc_height = a.left.centre.get_height_prime()
        all_height = a.left.left.get_height_prime()
        alr_height = a.left.right.get_height_prime()

        if alc_height <= np.max([all_height, alr_height]):
            # single rotation from left
            if DEBUG:
                print("single rotation from left")
                # debug_prints(tt=node, print_label=True)

            a = left_to_centre(a)

        else:
        # elif alc_height > np.max([all_height, alr_height]):
            # TODO: this could be an else statement?
            # double rotation from left
            # centre_to_left/right
            double_rotation = True
            if DEBUG:
                print("double rotation from left")
                # debug_prints(tt=node, print_label=True)

            if a.left.centre.node_label == TNodeL:
                a.left = centre_to_left(a.left)
            elif a.left.centre.node_label == TNodeR:
                a.left = centre_to_right(a.left)
            else:
                print("centre to left/right not happening in double from left")
                print(a.centre.node_label)
                # debug_prints(tt=node, print_label=True)

            a = left_to_centre(a)
            
    elif a.node_label != TNodeL and ar_height > 1 + np.max([ac_height, al_height]):

        arc_height = a.right.centre.get_height_prime()
        arl_height = a.right.left.get_height_prime()
        arr_height = a.right.right.get_height_prime()

        if arc_height <= np.max([arl_height, arr_height]):
            # single rotation from right
            if DEBUG:
                print("single rotation from right")
                # debug_prints(tt=a, print_label=True)

            a = right_to_centre(a)
        else:
        # elif arc_height > np.max([arl_height, arr_height]):
            # double rotation from right
            if DEBUG:
                print("double rotation from right")
                # debug_prints(tt=a, print_label=True)
            double_rotation = True
            # centre_to_left/right
            if a.right.centre.node_label == TNodeL:
                a.right = centre_to_left(a.right)
            elif a.right.centre.node_label == TNodeR:
                a.right = centre_to_right(a.right)
            else:
                print("centre to left/right not happening in double from right")
                print(a.centre.node_label)
                # debug_prints(tt=a, print_label=True)

            a = right_to_centre(a)

        if DEBUG:
            if a.centre is not None and a.centre.node_label == TNodeN:
                print("INFO: centre child has N label")

    a.update_heights()

    return a, double_rotation

def rebalance(tt, u, path, is_root=True):
    # is_root is used to track if it's the root vertex during the recursion as we don't run the
    # dynamic rebalancing on the root node
    # When rebalancing the ternary tree only rebalance the left or right subtree of the root
    # (depending on where the added vertex is). So the centre child of the root of the ternary tree
    # should always be a leaf that corresponds to the root of the binary tree.

    # this function isn't actually named very well as it alters the tree and then rebalances?
    # u is parent of the new vertices to add to the tt
    # u is binary tree of what to add
    # 0  u   
    # 1 l r 
    # ternary tree:
    # 0   N     
    # 1 l u r 

    inserted_segment = None
    l_inserted_segment = None
    c_inserted_segment = None
    r_inserted_segment = None
    double_rotation = False

    if tt is None:
        # need to initialise ternary tree
        tt = TernaryTree(u)
        tt.centre = TernaryTree(u, TLeafN)

        left_label = TLeafL if u.left.left is None else TLeafN
        tt.left = TernaryTree(u.left, left_label)

        right_label = TLeafL if u.right.left is None else TLeafN
        tt.right = TernaryTree(u.right, right_label)

        tt.update_all()

        if tt.is_contraction:
            tt = update_bp_values_on_segments_leaves(tt)
            update_psi(tt)

        return tt, tt

    if len(path) == 0:
        # find u on ternary tree (using bt_id value)

        # create ternary node:
        new_vertex = TernaryTree(u)
        new_vertex.node_label = TNodeN

        new_vertex.centre = TernaryTree(u, TLeafN)
        # the u ternary node we just found

        # TODO: finish trying/testing this out? not really sure what I'm doing though...?
        if tt.get_id() == u.left.get_id():
            # u_hat is left
            new_vertex.left = tt

            if u.right.left is not None:
                leaf_label = TLeafN
            else:
                leaf_label = TLeafL

            new_vertex.right = TernaryTree(u.right, leaf_label)

        else:
            new_vertex.right = tt

            if u.left.left is not None:
                leaf_label = TLeafN
            else:
                leaf_label = TLeafL

            new_vertex.left = TernaryTree(u.left, leaf_label)

        new_vertex.update_all()

        if tt.is_contraction:
            new_vertex = update_bp_values_on_segments_leaves(new_vertex)

            if new_vertex.node_label == TNodeN:
                update_psi(new_vertex)
            else:
                update_big_omega(new_vertex)

        return new_vertex, new_vertex

    if path[0] == LEFT:
        tt.left, l_inserted_segment = rebalance(tt.left, u, path[1:], is_root=False)
    elif path[0] == CENTRE:
        tt.centre, c_inserted_segment = rebalance(tt.centre, u, path[1:], is_root=False)
    elif path[0] == RIGHT:
        tt.right, r_inserted_segment = rebalance(tt.right, u, path[1:], is_root=False)

    if l_inserted_segment is not None:
        inserted_segment = l_inserted_segment
    elif c_inserted_segment is not None:
        inserted_segment = c_inserted_segment
    else:
        inserted_segment = r_inserted_segment

    tt.update_all()

    # get here once vertex has been updates, then rebalance as climb back up the tree
    # but don't rebalance at the root
    if not is_root and tt.height > 1:
        tt, double_rotation = tt_dynamic_rebalance(tt)

    if tt.is_contraction:
        if double_rotation:
            # if double rotation has happened need to re-calculate beliefs on the child segments of the tree?
            for s in [tt.left, tt.centre, tt.right]:
                if s.node_label is not None:
                    for s_child in [s.left, s.centre, s.right]:
                        s_child.psi = update_psi(s)
                        s_child.big_omega = update_big_omega(s)

            if tt.node_label is not None:
                for s in [tt.left, tt.centre, tt.right]:
                    s.psi = update_psi(s)
                    s.big_omega = update_big_omega(s)

            if is_root:
                tt.psi = update_psi(tt)
                tt.big_omega = update_big_omega(tt)

    return tt, inserted_segment
