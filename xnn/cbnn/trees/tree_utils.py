"""Various utility functions for use on the binary and ternary trees for the CBNN implementation"""

# constants needed
LEFT = "L"
CENTRE = "C"
RIGHT = "R"

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

    for i, path_segment in enumerate(node_1_path):
        if path_segment == node_2_path[i]:
            common_path.append(path_segment)
            common_node_index += 1
        else:
            # common path finished
            break

    common_ancestor = follow_path(tree, common_path)

    return common_ancestor, common_path, common_node_index