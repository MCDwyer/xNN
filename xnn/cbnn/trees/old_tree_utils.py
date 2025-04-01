"""Various utility functions for use on the binary and ternary trees for the CBNN implementation"""
import copy
import pickle
import zipfile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .tree_classes import BinaryTreeZ, TernaryTree

# constants needed
TLeafL = "L"
TLeafN = "N"
TNodeN = "N"
TNodeL = "L"
TNodeR = "R"
LEFT = "L"
CENTRE = "C"
RIGHT = "R"
DX = 5
DY = -5
OFFSET = 5
MAX_PLOT_WIDTH = 15

VECTOR_INPUT = False

def _print_aux(node, level, lowest_level, whitespace_amount, one_string_format, current_level=0, is_binary=False,
               print_height=False, print_id=False, print_d=False, print_label=False):
    """recursive print function for binary and ternary trees, with optional booleans to 
    instead print the height, id or d value of the vertex"""

    if current_level == level:
        # then print stuff?
        # whitespace = " "*(2*(num_children**(lowest_level - current_level - 1)) - 1)

        if node is None:
            value = one_string_format
        else:
            if print_d:
                value = str(node.d)
            elif print_height:
                value = str(node.height)
            elif print_id:
                value = str(node.get_id())
            elif print_label:
                value = str(node.node_label) if node.leaf_label is None else str(node.leaf_label).lower()
            else:
                # if isinstance(node.value, str):
                #     value = node.value
                if is_binary:
                    if VECTOR_INPUT:
                        value = f"[{node.value[0]:+.5g},{node.value[1]:+.5g}]"
                    else:
                        value = f"{node.value}"
                else:
                    if node.node_label is None:
                        if VECTOR_INPUT:
                            value = f"[{node.value[0]:+.5g},{node.value[1]:+.5g}]"
                        else:
                            value = f"{node.value}"
                    else:
                        value = " "*int(len(one_string_format)/2) + str(node.node_label) + " "*int(len(one_string_format)/2)

        return value + " "*whitespace_amount
    else:
        # get to that level?
        current_level += 1

        left_node = node.left if node is not None else None
        if not is_binary:
            centre_node = node.centre if node is not None else None
        right_node = node.right if node is not None else None

        left_value = _print_aux(left_node, level, lowest_level, whitespace_amount,
                                one_string_format, current_level, is_binary, print_height,
                                print_id, print_d, print_label)
        if not is_binary:
            centre_value = _print_aux(centre_node, level, lowest_level, whitespace_amount,
                                      one_string_format, current_level, is_binary, print_height,
                                      print_id, print_d, print_label)
        right_value = _print_aux(right_node, level, lowest_level, whitespace_amount,
                                 one_string_format, current_level, is_binary, print_height,
                                 print_id, print_d, print_label)

        if is_binary:
            return left_value + right_value

        return left_value + centre_value + right_value

def print_tree(node, print_height=False, print_id=False, print_d=False, print_label=False):
    """function to print binary and ternary trees utilising the _print_aux function, 
    this only formats it well if the node value is 1 space long atm"""

    is_binary = False if isinstance(node, TernaryTree) else True

    global VECTOR_INPUT

    if is_binary:
        VECTOR_INPUT = True if node.value.ndim > 1 else False
    else:
        VECTOR_INPUT = True if node.u.value.ndim > 1 else False

    height = node.height

    num_children = 2 if is_binary else 3

    if print_height or print_d or print_label:
        one_string_format = "_"
    elif print_id:
        one_string_format = "__________"
    else:
        if not VECTOR_INPUT:
            one_string_format = "+_.__"
        else:
            one_string_format = "[___.___,___.___]"
            one_string_format = "[__,__]"

    length_of_one_string = len(one_string_format)
    lowest_level = height # as 0 indexed
    for level in range(lowest_level):
        whitespace_amount = (num_children**(lowest_level - level) - 1)*(length_of_one_string)
        whitespace_amount += (num_children**(lowest_level - level))
        left_whitespace = " "*int(whitespace_amount/2)
        
        node_values = _print_aux(node, level, lowest_level, whitespace_amount, one_string_format,
                                 is_binary=is_binary,
                                 print_height=print_height, print_id=print_id, print_d=print_d,
                                 print_label=print_label)
        print(str(level) + left_whitespace + node_values)

def _print_bp_aux(node, level, lowest_level, whitespace_amount, one_string_format,
                  print_big_omega=True, print_psi=False, print_omega=False, current_level=0):
    """gets the big omega/psi/omega and omega prime values of a ternary tree to print them out"""

    length_of_one_string = len(one_string_format)

    if current_level == level:
        # print stuff?
        whitespace = " "*whitespace_amount
        if whitespace_amount == 0:
            whitespace = " "

        values = []

        if node is None:
            # empty_whitespace = " "*length_of_one_string
            # values.append(empty_whitespace + whitespace)
            values.append(one_string_format + whitespace)
            values.append(one_string_format + whitespace)
        else:
            # tmp_value = " "*int(length_of_one_string/2)
            # tmp_value += str(node.node_label) if node.node_label is not None else f"[{node.value[0]:+.2f},{node.value[1]:+.2f}]"
            # tmp_value += " "*(int(length_of_one_string/2))
            # values.append(tmp_value + whitespace)

            appended = False

            if print_big_omega:
                big_omega = node.big_omega
                if big_omega is not None:
                    appended = True
                    values.append(f"|{big_omega[0][0]:+.2f},{big_omega[0][1]:+.2f}|" + whitespace)
                    values.append(f"|{big_omega[1][0]:+.2f},{big_omega[1][1]:+.2f}|" + whitespace)
            elif print_psi:
                psi = node.psi
                if psi is not None:
                    appended = True
                    values.append(f"|{psi[0]:+.2f}|" + whitespace)
                    values.append(f"|{psi[1]:+.2f}|" + whitespace)
            elif print_omega:
                w = node.w
                w_prime = node.w_prime
                if w is not None:
                    appended = True
                    to_append_0 = f"|{w[0]:+.2f}|"
                    to_append_1 = f"|{w[1]:+.2f}|"

                    if w_prime is not None:
                        to_append_0 += f"{w_prime[0]:+.2f}|" + whitespace
                        to_append_1 += f"{w_prime[1]:+.2f}|" + whitespace
                    else:
                        to_append_0 += "__.__|" + whitespace
                        to_append_1 += "__.__|" + whitespace

                    values.append(to_append_0)
                    values.append(to_append_1)

            if not appended:
                values.append(one_string_format + whitespace)
                values.append(one_string_format + whitespace)

        return values
    else:
        # get to that level?
        current_level += 1

        left_node = node.left if node is not None else None
        centre_node = node.centre if node is not None else None
        right_node = node.right if node is not None else None

        left_value = _print_bp_aux(left_node, level, lowest_level, whitespace_amount,
                                   one_string_format, print_big_omega, print_psi, print_omega,
                                   current_level)
        centre_value = _print_bp_aux(centre_node, level, lowest_level, whitespace_amount,
                                     one_string_format, print_big_omega, print_psi, print_omega,
                                     current_level)
        right_value = _print_bp_aux(right_node, level, lowest_level, whitespace_amount,
                                    one_string_format, print_big_omega, print_psi, print_omega,
                                    current_level)

        new_values = []
        new_values.append(left_value[0] + centre_value[0] + right_value[0])
        new_values.append(left_value[1] + centre_value[1] + right_value[1])
        # new_values.append(left_value[2] + centre_value[2] + right_value[2])

        return new_values

def print_bp_tree(node, print_big_omega=False, print_psi=False, print_omega=False):
    """prints out the big omega/psi/omega and omega prime values for a ternary tree"""
    height = node.height

    num_children = 3
    one_string_format = None

    if print_big_omega:
        one_string_format = "|__.__,__.__|"
    elif print_psi:
        one_string_format = "|__.__|"
    elif print_omega:
        one_string_format = "|__.__|__.__|"

    if one_string_format is None:
        print("no info on what to print given")
        return

    lowest_level = height - 1 # as 0 indexed
    length_of_one_string = len(one_string_format)

    for level in range(height):
        whitespace_amount = (num_children**(lowest_level - level) - 1)*(length_of_one_string)
        whitespace_amount += (num_children**(lowest_level - level))
        # print(whitespace_amount)
        left_whitespace = " "*int(whitespace_amount/2)
        node_values = _print_bp_aux(node, level, lowest_level, whitespace_amount, one_string_format,
                                    print_big_omega=print_big_omega, print_psi=print_psi,
                                    print_omega=print_omega)

        print(str(level) + left_whitespace + node_values[0])
        print(" " + left_whitespace + node_values[1])
        # print(" " + left_whitespace + node_values[2])

def bt_find_vertex(node, node_to_find, internal_only=False):
    # TODO: could probably combine this with the tt find vertex, as it's essentially the same code?

    if internal_only:
        found_node, is_found, path = _bt_find_vertex_internal(node, node_to_find)
    else:
        found_node, is_found, path = _bt_find_vertex(node, node_to_find)

    # path needs to be reversed
    if is_found:
        path.reverse()

    return found_node, is_found, path

def _bt_find_vertex_internal(node, node_to_find, is_root=True):
    # TODO: fix this, pretty sure this is just doing a full search of the tree?
    if node is None:
        return None, False, None
    elif node.get_id() == node_to_find.get_id():
        path = []
        return node, True, path

    is_root = False

    if node.left is not None:
        found_node, is_found, path = _bt_find_vertex_internal(node.left, node_to_find, is_root)

        if is_found:
            path.append(LEFT)
            return found_node, is_found, path

    if node.right is not None:
        found_node, is_found, path = _bt_find_vertex_internal(node.right, node_to_find, is_root)

        if is_found:
            path.append(RIGHT)
            return found_node, is_found, path

    # if I get here means I've gotten to a leaf, and this is internal only search?
    return None, False, None

def _bt_find_vertex(node, node_to_find):
    if node is None:
        return None, False, None

    if node.left is not None:
        found_node, is_found, path = _bt_find_vertex(node.left, node_to_find)

        if is_found:
            path.append(LEFT)
            return found_node, is_found, path

    if node.right is not None:
        found_node, is_found, path = _bt_find_vertex(node.right, node_to_find)

        if is_found:
            path.append(RIGHT)
            return found_node, is_found, path

    if node.get_id() == node_to_find.get_id():
        path = []
        return node, True, path

    return None, False, None

def bt_find_furthest_child(node, find_left):
    """finds and returns the node for the furthest left or right child on the binary tree given"""
    if node.left is None:
        # where we always create full trees checking node.left alone confirms it is a leaf node
        return node

    if find_left:
        child_node = bt_find_furthest_child(node.left, find_left)
    else:
        child_node = bt_find_furthest_child(node.right, find_left)

    return child_node

def follow_path(node, path):
    """follows the given path through the tree and returns the node at the end of that path"""

    if len(path) == 0:
        return node

    if node is None:
        return None

    if path[0] == LEFT:
        found_node = follow_path(node.left, path[1:])

    elif path[0] == CENTRE:
        found_node = follow_path(node.centre, path[1:])

    elif path[0] == RIGHT:
        found_node = follow_path(node.right, path[1:])

    return found_node

def find_least_common_ancestor(tree, node_1_path, node_2_path):
    """given two paths finds the least common ancestor on those paths 
    and returns that node in the tree"""
    # compare lists to get shared portion
    common_path = []
    common_node_index = -1 # return -1 if root?

    if node_1_path is None or node_2_path is None:
        # root of tree is the least common ancestor?
        return tree, common_path, common_node_index

    for i in range(len(node_1_path)):
        if node_1_path[i] == node_2_path[i]:
            common_path.append(node_1_path[i])
            common_node_index += 1
        else:
            # common path finished
            break

    common_ancestor = follow_path(tree, common_path)

    return common_ancestor, common_path, common_node_index

def prep_tree_for_save(tree):

    is_binary = True if isinstance(tree, BinaryTree) else False

    tree = _prep_tree_for_save(tree, is_binary)

    return tree

def _prep_tree_for_save(tree, is_binary):
    # tt is root
    if tree is None:
        return tree

    if tree.bt_id is None:
        tree.bt_id = tree.get_id()

    if tree.left is not None:
        tree.left = _prep_tree_for_save(tree.left, is_binary)

    if not is_binary:
        if tree.centre is not None:
            tree.centre = _prep_tree_for_save(tree.centre, is_binary)

    if tree.right is not None:
        tree.right = _prep_tree_for_save(tree.right, is_binary)

    return tree

def bt_to_tt(node):
    loop_break_limit = node.height
    # don't think I actually should need this though but don't trust the code to not just run forever

    # while this isn't a ternary tree only? i.e. one node
    only_ternary = False
    i = 0
    while not only_ternary:
        i += 1

        print("Iterating through tree count: ", i)
        node, _ = _merge_bt_to_tt(node)
        if node.height == 1:
            only_ternary = True

        # temp code as otherwise it could go on forever
        if i > loop_break_limit:
            print("breaking while loop as iterating too long")
            only_ternary = True

    # node is a binary node, so return the object stored in that binary node as that should be the full ternary tree
    return node.value

def _merge_bt_to_tt(node):
    # Algorithm 1 from paper

    # adj_node_merged boolean is to indicate if a directly attached node has just merged, so that we don't merge connecting branches straight away
    # this is from step 2.1 in algo 1 in the paper (mark internal nodes such that no two adjacent nodes are marked)
    # these nodes not being merged this time will be merged on the next iteration through the tree?

    adj_node_merged = False

    if node is None:
        return node, adj_node_merged

    left_merged = False
    right_merged = False

    if node.left is not None:
        node.left, left_merged = _merge_bt_to_tt(node.left)

    if node.right is not None:
        node.right, right_merged = _merge_bt_to_tt(node.right)

    # is it a leaf node?
    has_left_child = False if node.left is None else True
    has_right_child = False if node.right is None else True

    if not has_left_child and not has_right_child:
        # has no children so leaf node, so can pass back without having to do any merging
        return node, adj_node_merged

    if not left_merged and not right_merged:
        # at this point it means it hasn't been merged on any immediately connected nodes, and it has at least one child

        has_left_leaf = False
        has_right_leaf = False
        # has it got at least one leaf child?
        if has_left_child:
            if node.left.left is None and node.left.right is None:
                # checking the left child of this node has no children
                has_left_leaf = True

        if has_right_child:
            if node.right.left is None and node.right.right is None:
                # checking the right child of this node has no children
                has_right_leaf = True

        new_ternary = TernaryTree(node.value)
        # if binary nodes children are ternary trees then directly add those,
        # otherwise should be a single value binary node which will need to be made into a ternary node
        if isinstance(node.left.value, TernaryTree):
            new_ternary.left = node.left.value
        else:
            new_ternary.left = TernaryTree(node.left.value)

        if isinstance(node.right.value, TernaryTree):
            new_ternary.right = node.right.value
        else:
            new_ternary.right = TernaryTree(node.right.value)

        if isinstance(node.value, TernaryTree):
            new_ternary.centre = node.value
        else:
            new_ternary.centre = TernaryTree(node.value)

        new_ternary.height = np.max([new_ternary.left.height, new_ternary.centre.height, new_ternary.right.height]) + 1

        new_binary = BinaryTreeZ(" ")

        if has_left_leaf and has_right_leaf:
            print("merging - N")
            # new_ternary.value = node.value
            new_ternary.node_label = 'N'

        elif has_left_leaf:
            print("merging - R")
            # only has left leaf
            # new_ternary.value = node.value
            new_ternary.node_label = 'R'

            new_binary.left = node.right.left
            new_binary.right = node.right.right

        elif has_right_leaf:
            print("merging - L")
            # only has right leaf
            # new_ternary.value = node.value
            new_ternary.node_label = 'L'

            new_binary.left = node.left.left
            new_binary.right = node.left.right

        else:
            # this node is an internal node with two non-leaf children so need to handle it later once it's children have been merged
            return node, adj_node_merged

        adj_node_merged = True

        new_binary.value = new_ternary
        left_height = 0 if new_binary.left is None else new_binary.left.height
        right_height = 0 if new_binary.right is None else new_binary.right.height

        new_binary.height = np.max([left_height, right_height]) + 1

        return new_binary, adj_node_merged

    # merged on directly attached branch in this loop, so need to skip this node for now
    return node, adj_node_merged

def create_bt_graph(bt):

    graph = nx.Graph()

    parent = bt
    parent_node = create_unique_graph_node_key(parent, {}) #parent.get_id()

    graph.add_node(parent_node)
    positions = {parent_node: [0, 0]}

    height = bt.height

    if bt.left is not None:
        child = bt.left
        graph, positions = _create_bt_graph(graph, parent, parent_node, child, positions, height, left=True)

    if bt.right is not None:
        child = bt.right
        graph, positions = _create_bt_graph(graph, parent, parent_node, child, positions, height)

    return graph, positions

def _create_bt_graph(graph, parent, parent_node_key, child, positions, height, level=0, left=False):

    if child is None:
        return graph, positions

    child_node_key = create_unique_graph_node_key(child, positions)

    graph.add_node(child_node_key)
    graph.add_edge(parent_node_key, child_node_key)

    level += 1

    horizontal_offset = OFFSET * (height - level)

    x_position = (DX * 2**(height-level)) + horizontal_offset
    x_position = -x_position if left else x_position

    position = copy.deepcopy(positions[parent_node_key])
    position[0] += x_position
    position[1] += DY

    horizontal_offset = OFFSET*(height - level)
    x_position = -(DX + horizontal_offset) if left else DX + horizontal_offset
    position = copy.deepcopy(positions[parent_node_key])
    position[0] += x_position
    position[1] += DY

    positions[child_node_key] = position
    
    if parent.left is not None:
        grandchild = child.left
        graph, positions = _create_bt_graph(graph, child, child_node_key, grandchild, positions, height, level, left=True)

    if parent.right is not None:
        grandchild = child.right
        graph, positions = _create_bt_graph(graph, child, child_node_key, grandchild, positions, height, level)

    return graph, positions

def create_unique_graph_node_key(vertex, positions, is_binary=True):
    id_string = str(vertex.get_id())
    id_string_index = len(id_string) - 1 # index so we can add the final digit of the id to the string

    if not is_binary:
        if vertex.node_label is not None:
            node_key = vertex.node_label + "_" + id_string[id_string_index]
        else:
            try:
                node_key = f"[{vertex.value[0]:2f},{vertex.value[1]:2f}]_{id_string[id_string_index]}" #child.get_id()
            except IndexError:
                node_key = f"[{vertex.value[0]}]_{id_string[id_string_index]}" 
    else:
        # TODO: fix this as should be able to do this in a nicer/better if I think?
        try:
            node_key = f"[{vertex.value[0]:2f},{vertex.value[1]:2f}]_{id_string[id_string_index]}" #child.get_id()
        except IndexError:
            node_key = f"[{vertex.value[0]}]_{id_string[id_string_index]}"

    # if this string for this node already exists, then keep adding in the id digits until it doesn't so it's unique
    while node_key in positions:
        id_string_index -= 1
        if id_string_index < 0:
            print("don't think this should happen though??")
            id_string = "0" + id_string
        node_key = node_key + id_string[id_string_index]

    return node_key

def create_tt_graph(tt):

    graph = nx.Graph()

    height = tt.height
    parent = tt

    parent_node = create_unique_graph_node_key(parent, {}, False)

    # if height == 1:
    #     parent_node = str(parent.value)
    # else:
    #     parent_node = str(parent.node_label) #parent.get_id()

    graph.add_node(parent_node)
    positions = {parent_node: [0, 0]}

    if tt.left is not None:
        child = tt.left
        graph, positions = _create_tt_graph(graph, parent, parent_node, child, positions, height, direction=LEFT)

    if tt.centre is not None:
        child = tt.centre
        graph, positions = _create_tt_graph(graph, parent, parent_node, child, positions, height, direction=CENTRE)

    if tt.right is not None:
        child = tt.right
        graph, positions = _create_tt_graph(graph, parent, parent_node, child, positions, height, direction=RIGHT)

    return graph, positions

def _create_tt_graph(graph, parent, parent_node_key, child, positions, height, direction, level=0):

    if child is None:
        return graph, positions

    child_node_key = create_unique_graph_node_key(child, positions, is_binary=False)

    graph.add_node(child_node_key)
    graph.add_edge(parent_node_key, child_node_key)

    level += 1

    horizontal_offset = OFFSET * (height - level)

    if direction == CENTRE:
        x_position = 0
    else:
        x_position = (DX * 3**(height-level)) + horizontal_offset
        x_position = -x_position if direction == LEFT else x_position

    position = copy.deepcopy(positions[parent_node_key])
    position[0] += x_position
    position[1] += DY

    positions[child_node_key] = position
    
    if parent.left is not None:
        grandchild = child.left
        graph, positions = _create_tt_graph(graph, child, child_node_key, grandchild, positions, height, LEFT, level)

    if parent.centre is not None:
        grandchild = child.centre
        graph, positions = _create_tt_graph(graph, child, child_node_key, grandchild, positions, height, CENTRE, level)

    if parent.right is not None:
        grandchild = child.right
        graph, positions = _create_tt_graph(graph, child, child_node_key, grandchild, positions, height, RIGHT, level)

    return graph, positions

def create_and_save_graph_plot(tree, file_name, colour="#cbf3f0"):

    height = tree.height

    is_binary = False if isinstance(tree, TernaryTree) else True

    num_children = 2 if is_binary else 3

    x_size = (num_children**height)

    x_scaling = MAX_PLOT_WIDTH #np.ceil(MAX_PLOT_WIDTH/height)
    y_scaling = np.floor(MAX_PLOT_WIDTH/height)

    x_size = 1 * x_scaling
    y_size = 1 * y_scaling

    plt.figure(figsize=(x_size, y_size))

    if is_binary:
        graph, positions = create_bt_graph(tree)
    else:
        graph, positions = create_tt_graph(tree)

    nx.draw(graph, positions, with_labels=True, font_size=10, node_color=colour)

    plt.savefig(file_name)
    plt.close('all')

# TODO: work out how to save and load the trees when the recursion is too large for pickle
def save_trees(bt_B, bt_Z, tt_H_Z, filepath, training_data=None):
    
    bt_Z = prep_tree_for_save(bt_Z)

    bt_Z_filename = "bt_Z.pkl"
    tt_H_Z_filename = "tt_H_Z.pkl"
    bt_B_filename = "bt_B.pkl"

    # Name of the zip file
    zip_filename = filepath + "_CBNN_Trees.zip"

    # Serialize the object and store it in a zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as myzip:
        # Serialize the object to a bytes object
        pickled_bt_Z = pickle.dumps(bt_Z)
        pickled_tt_H_Z = pickle.dumps(tt_H_Z)
        pickled_bt_B = pickle.dumps(bt_B)
        
        # Use writestr to create a file in the zip archive directly from the serialized data
        myzip.writestr(bt_Z_filename, pickled_bt_Z)
        myzip.writestr(tt_H_Z_filename, pickled_tt_H_Z)
        myzip.writestr(bt_B_filename, pickled_bt_B)

        if training_data is not None:
            with open("training_info.txt", "w") as file:
                file.write(training_data)

    print("Trees saved in: " + zip_filename)

    return zip_filename

def load_trees(zip_filename):
    # The name of the pickled file inside the zip archive
    bt_Z_filename = "bt_Z.pkl"
    tt_H_Z_filename = "tt_H_Z.pkl"
    bt_B_filename = "bt_B.pkl"

    # Open the zip file
    with zipfile.ZipFile(zip_filename, 'r') as myzip:
        # Read the pickled data from the file inside the zip
        with myzip.open(bt_Z_filename) as file:
            # Deserialize the object
            bt_Z = pickle.load(file)
        
        with myzip.open(tt_H_Z_filename) as file:
            # Deserialize the object
            tt_H_Z = pickle.load(file)
        
        with myzip.open(bt_B_filename) as file:
            # Deserialize the object
            bt_B = pickle.load(file)
            
    return bt_B, bt_Z, tt_H_Z
