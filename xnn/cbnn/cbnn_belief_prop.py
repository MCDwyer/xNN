import numpy as np

# constants needed
TLeafL = "L"
TLeafN = "N"
TNodeN = "N"
TNodeL = "L"
TNodeR = "R"
DTYPE = np.float128
MAX_VALUE = 1e50
MIN_VALUE = 1e-50

def bp_full_update(s):

    # updating big omega and psi
    update_climbing_tree(s)

    # updating omega
    update_descending_tree(s)

    return

def update_climbing_tree(s):

    if s.left is not None:
        bp_full_update(s.left)
    if s.centre is not None:
        bp_full_update(s.centre)
    if s.right is not None:
        bp_full_update(s.right)

    if s.node_label is not None:
        if s.node_label == TNodeN:
            update_psi(s)
        else:
            update_big_omega(s)
    else:
        if s.leaf_label == TLeafL:
            update_psi(s)
        else:
            update_big_omega(s)

    return

def update_descending_tree(s):

    s.w = update_omega(s)

    if s.left is not None:
        update_descending_tree(s.left)
    
    if s.centre is not None:
        update_descending_tree(s.centre)

    if s.right is not None:
        update_descending_tree(s.right)

    return

def update_psi(s):
    # Given a leaf... If it is a centre child then it has an omega. If it is the left child of an L
    # node it has an omega. If it is the right child of an R node it has an omega. Otherwise it has
    # a psi -> essentially has a psi if it's a TLeafL node? update the value of psi on the vertex s,
    # updating the left and right psi's as necessary?

    if s.node_label is not None and s.node_label != TNodeN:
        s.psi = None
        return None

    if s.node_label is None:
        # leaf node
        if s.leaf_label == TLeafL:
            psi = np.matmul(s.tau, s.kappa, dtype=DTYPE)
        else:
            psi = None
            return None

    else:
        left_psi = get_psi(s.left)
        right_psi = get_psi(s.right)
        centre_big_omega = get_big_omega(s.centre)

        psi = np.multiply(left_psi, right_psi, dtype=DTYPE)
        psi = np.matmul(centre_big_omega, psi, dtype=DTYPE)
    
    psi = np.clip(psi, MIN_VALUE, MAX_VALUE)

    s.psi = psi

    return psi

def update_big_omega(s):
    # Given a leaf... If it is a centre child then it has an omega.
    # If it is the left child of an L node it has an omega.
    # If it is the right child of an R node it has an omega.
    # Otherwise it has a psi -> essentially has a big omega if it's a TLeafN node?

    if s.node_label is not None and s.node_label == TNodeN:
        s.big_omega = None
        return None

    if s.node_label is None:
        # leaf node so don't do anything here
        if s.leaf_label == TLeafL:
            return None
        else:
            big_omega = np.multiply(s.tau, s.kappa, dtype=DTYPE)
    else:
        if s.node_label == TNodeL:
            s_prime = s.left
            s_prime2 = s.right
        else:
            s_prime = s.right
            s_prime2 = s.left

        centre_big_omega = get_big_omega(s.centre)
        s_prime_big_omega = get_big_omega(s_prime)
        s_prime2_psi = get_psi(s_prime2)

        big_omega = np.multiply(centre_big_omega, s_prime2_psi, dtype=DTYPE)
        big_omega = np.matmul(big_omega, s_prime_big_omega, dtype=DTYPE)

    big_omega = np.clip(big_omega, MIN_VALUE, MAX_VALUE)

    s.big_omega = big_omega

    return big_omega

def get_big_omega(s):
    big_omega = s.big_omega if s.big_omega is not None else update_big_omega(s)

    if big_omega is None:
        print("big_omega is none")

    return big_omega

def get_psi(s):
    psi = s.psi if s.psi is not None else update_psi(s)

    if psi is None:
        print("psi is none")

    return psi

def update_omega(s):

    if s.node_label is None:
        # leaf so should be?
        return s.w

    s.centre.w = s.w

    if s.node_label is not None and s.node_label != TNodeN:
        # then s is in D•
        # if s.left.node_label is not None and s.left.node_label != TNodeN:
        if s.node_label == TNodeL:
            s_prime = s.left
            s_prime2 = s.right
        else:
            s_prime = s.right
            s_prime2 = s.left

        s_prime_big_omega = get_big_omega(s_prime)
        s_prime2_psi = get_psi(s_prime2)

        tmp_s_centre_w_prime = np.matmul(s_prime_big_omega, s.w_prime, dtype=DTYPE)
        s.centre.w_prime = np.multiply(s_prime2_psi, tmp_s_centre_w_prime, dtype=DTYPE)
        s.centre.w_prime = np.clip(s.centre.w_prime, MIN_VALUE, MAX_VALUE)

        s_big_omega = get_big_omega(s)

        s_prime.w = np.matmul(s.w, s_big_omega, dtype=DTYPE)
        s_prime.w = np.multiply(s_prime.w, s_prime2_psi, dtype=DTYPE)
        s_prime.w = np.clip(s_prime.w, MIN_VALUE, MAX_VALUE)

        s_centre_big_omega = get_big_omega(s.centre)

        s_prime.w_prime = s.w_prime

        s_prime2.w = np.matmul(s.w, s_centre_big_omega, dtype=DTYPE)
        tmp_s_prime2 = np.matmul(s.w_prime, s_prime_big_omega.T, dtype=DTYPE)

        s_prime2.w = np.multiply(s_prime2.w, tmp_s_prime2, dtype=DTYPE)
        s_prime2.w = np.clip(s_prime2.w, MIN_VALUE, MAX_VALUE)

    else:
        s_left_psi = get_psi(s.left)
        s_right_psi = get_psi(s.right)
        s_centre_big_omega = get_big_omega(s.centre)

        s.centre.w_prime = np.multiply(s_left_psi, s_right_psi, dtype=DTYPE)
        s.centre.w_prime = np.clip(s.centre.w_prime, MIN_VALUE, MAX_VALUE)

        sum_value = np.matmul(s.w, s_centre_big_omega, dtype=DTYPE)

        s.left.w = np.multiply(s_right_psi, sum_value, dtype=DTYPE)
        s.left.w = np.clip(s.left.w, MIN_VALUE, MAX_VALUE)

        s.right.w = np.multiply(s_left_psi, sum_value, dtype=DTYPE)
        s.right.w = np.clip(s.right.w, MIN_VALUE, MAX_VALUE)

    return s.w

def update_bp_values_on_segments_leaves(segment_vertex):

    for vertex in [segment_vertex.left, segment_vertex.centre, segment_vertex.right]:
        has_psi = False
        has_omega = False

        if vertex.node_label is not None and vertex.node_label == TNodeN:
            has_psi = True
        elif vertex.node_label is not None:
            has_omega = True
        elif vertex.leaf_label is not None and vertex.leaf_label == TNodeL:
            has_psi = True
        else:
            has_omega = True

        if has_omega:
            vertex.big_omega = update_big_omega(vertex)
        if has_psi:
            vertex.psi = update_psi(vertex)

    return segment_vertex

def check_bp(tt):

    if tt.node_label is None:
        return

    if tt.node_label == TNodeN:
        # An N node should have psi_i=\sum_j omegacentre_{i,j}psileft_{j}\psiright_{j}

        psi = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            for j in [0, 1]:
                psi[i] += tt.centre.big_omega[i][j]*tt.left.psi[j]*tt.right.psi[j]


        diff = np.sum(tt.psi - psi, dtype=np.float16)

        print("checking N node")
        print("tree psi: ")
        print("dtype: ", tt.psi.dtype)
        print(tt.psi)
        print("calculated here psi: ")
        print("dtype: ", psi.dtype)
        print(psi)

        print(diff)

        assert diff == 0
    else:
        big_omega = np.zeros((2,2), dtype=DTYPE)

        if tt.node_label == TNodeL:
            # An L node should have omega_{i,j}=\sum_k omegacentre_{i,k}\omegaleft_{k,j}\psiright_k
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        big_omega[i][j] += tt.centre.big_omega[i][k]*tt.left.big_omega[k][j]*tt.right.psi[k]

            print("checking L node")

        else:
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        big_omega[i][j] += tt.centre.big_omega[i][k]*tt.right.big_omega[k][j]*tt.left.psi[k]

            print("checking R node")

        diff = np.sum(tt.big_omega - big_omega, dtype=np.float16)

        print("tree big omega: ")
        print("dtype: ", tt.big_omega.dtype)
        print(tt.big_omega)
        print("calculated here big omega: ")
        print("dtype: ", big_omega.dtype)
        print(big_omega)
        print(diff)

        assert diff == 0

    # check w
    if tt.node_label is not None and tt.node_label == TNodeN:
        # check centre w = w
        diff = np.sum(tt.centre.w - tt.w, dtype=np.float16)

        print("centre w:")
        print(tt.centre.w)
        print("w:")
        print(tt.w)

        assert diff == 0

        centre_w_prime = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            centre_w_prime[i] = tt.left.psi[i]*tt.right.psi[i]

        diff = np.sum(centre_w_prime - tt.centre.w_prime, dtype=np.float16)

        print("centre w' from tree:")
        print(tt.centre.w_prime)
        print("centre w':")
        print(centre_w_prime)

        assert diff == 0

        left_w = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            for j in [0, 1]:
                left_w[i] += tt.right.psi[i]*tt.w[j]*tt.centre.big_omega[j][i]

        diff = np.sum(left_w - tt.left.w, dtype=np.float16)

        print("left w from tree:")
        print(tt.left.w)
        print("left w:")
        print(left_w)

        assert diff == 0

        right_w = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            for j in [0, 1]:
                right_w[i] += tt.left.psi[i]*tt.w[j]*tt.centre.big_omega[j][i]

        diff = np.sum(right_w - tt.right.w, dtype=np.float16)

        print("right w from tree:")
        print(tt.right.w)
        print("right w:")
        print(right_w)

        assert diff == 0

    elif tt.node_label is not None:
        diff = np.sum(tt.centre.w - tt.w, dtype=np.float16)

        print("centre w:")
        print(tt.centre.w)
        print("w:")
        print(tt.w)

        assert diff == 0

        centre_w_prime = np.zeros(2, dtype=DTYPE)

        s = tt

        if s.left.node_label is not None and s.left.node_label != TNodeN:
            # then s.left contains a terminal node
            s_prime = s.left
            s_prime_2 = s.right
            print("s' = left")
        elif s.right.node_label is not None and s.right.node_label != TNodeN:
            s_prime = s.right
            s_prime_2 = s.left
            print("s' = right")
        elif s.left.leaf_label is TLeafN:
            s_prime = s.left
            s_prime_2 = s.right
            print("s' = left")
        elif s.right.leaf_label is TLeafN:
            s_prime = s.right
            s_prime_2 = s.left
            print("s' = right")

        for i in [0, 1]:
            for j in [0, 1]:
                centre_w_prime[i] += s_prime_2.psi[i]*s_prime.big_omega[i][j]*tt.w_prime[j]

        diff = np.sum(centre_w_prime - tt.centre.w_prime, dtype=np.float16)

        print("centre w' from tree:")
        print(tt.centre.w_prime)
        print("centre w':")
        print(centre_w_prime)

        assert diff == 0

        s_prime_w = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            for j in [0, 1]:
                s_prime_w[i] += tt.w[j]*tt.big_omega[j][i]*s_prime_2.psi[i]

        diff = np.sum(s_prime_w - s_prime.w, dtype=np.float16)

        print("s' w from tree:")
        print(s_prime.w)
        print("s' w:")
        print(s_prime_w)

        assert diff == 0

        diff = np.sum(s_prime.w_prime - tt.w_prime, dtype=np.float16)

        print("s' w' from tree:")
        print(s_prime.w)
        print("tt. w' from tree:")
        print(tt.w_prime)

        assert diff == 0

        s_prime_2_w = np.zeros(2, dtype=DTYPE)

        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    s_prime_2_w[i] += tt.w[j]*tt.centre.big_omega[j][i]*tt.w_prime[k]*s_prime.big_omega[i][k]

        diff = np.sum(s_prime_2_w - s_prime_2.w, dtype=np.float16)

        print("s'' w from tree:")
        print(s_prime_2.w)
        print("s'' w:")
        print(s_prime_2_w)

        assert diff == 0

    if tt.left is not None:
        check_bp(tt.left)

    if tt.centre is not None:
        check_bp(tt.centre)

    if tt.right is not None:
        check_bp(tt.right)

    return
