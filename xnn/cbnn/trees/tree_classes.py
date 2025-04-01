import numpy as np
import uuid

# constants needed
TLeafL = "L"
TLeafN = "N"
TNodeN = "N"
TNodeL = "L"
TNodeR = "R"
LEFT = "L"
CENTRE = "C"
RIGHT = "R"
DTYPE = np.float128

class TreeSkeleton:
    def __init__(self) -> None:
        self._left = None
        self._right = None
        self._parent = None
        self.height = 1 # TODO: update height stuff
        self.unique_id = uuid.uuid4()

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, child):
        self._left = child
        self._left.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, child):
        self._right = child
        self._right.parent = self

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    def update_height(self):
        left_height = 0 if self.left is None else self.left.height
        right_height = 0 if self.right is None else self.right.height

        self.height = 1 + np.max([left_height, right_height])

        if self.parent is not None:
            self.parent.update_height()

    def get_path(self, is_binary):
        if self.parent is None:
            return []

        if is_binary:
            path = _get_path_binary(self.parent, self)
        else:
            path = _get_path_ternary(self.parent, self)

        path.reverse() # as travelled up the tree from node

        return path
    
    def get_parent_direction(self):
        if self.parent is None:
            return None
        
        return self._get_parent_direction

    @staticmethod
    def _get_parent_direction(parent, child):
        if parent.left == child:
            return LEFT
        elif parent.right == child:
            return RIGHT
        else:
            return CENTRE

class BinaryTree(TreeSkeleton):
    def __init__(self) -> None:
        super().__init__()
        self._s = None
        self.s_unique_id = None
        self._d = None

    @property
    def s(self):
        # ternary tree node link
        return self._s

    @s.setter
    def s(self, node):
        self._s = node
        if node is not None:
            self.s_unique_id = node.unique_id

    @property
    def d(self):
        return self._d
    
    @d.setter
    def d(self, value):
        self._d = value

    def get_path(self):
        return super().get_path(is_binary=True)

class Test:
    def __init__(self, id, pointer):
        self.id = id
        self.pointer = pointer

class BinaryTreeZ(BinaryTree):
    def __init__(self, value) -> None:
        self.value = value
        self.contraction_nodes = {}#set()
        self.contraction_unique_ids = {}
        super().__init__()

    def get_id(self):
        return self.unique_id #id(self)
    
    def add_contraction(self, contraction_id, node):
        # self.contraction_nodes.add(Test(contraction_id, node))
        self.contraction_nodes[contraction_id] = node
        # print(f"Contraction added {contraction_id}: {node.unique_id}")
        self.contraction_unique_ids[contraction_id] = node.unique_id

    def get_contraction(self, contraction_id):
        # for item in self.contraction_nodes:
        #     if item.id == contraction_id:
        #         return item.pointer
        return self.contraction_nodes[contraction_id]

class BinaryTreeContraction(BinaryTree):
    def __init__(self, bt_Z_node, contraction_id) -> None:
        super().__init__()
        self._bt_Z_node = None
        self.bt_Z_unique_id = None
        self.contraction_id = contraction_id

        self.bt_Z_node = bt_Z_node

        self.kappa = np.ones(2, dtype=DTYPE)
        self.tau = np.zeros((2, 2), dtype=DTYPE)

    def get_id(self):
        return self.bt_Z_node.get_id()
    
    @property
    def bt_Z_node(self):
        return self._bt_Z_node

    @bt_Z_node.setter
    def bt_Z_node(self, node):
        self._bt_Z_node = node

        if node is not None:
            self.bt_Z_unique_id = node.unique_id

            # add link to bt_Z_node
            node.add_contraction(self.contraction_id, self)

    def relink_bt_Z_node(self):
        if self.bt_Z_node is not None:
            # print("relinking nodes")
            self.bt_Z_node.add_contraction(self.contraction_id, self)

    @property
    def d(self):
        return self.bt_Z_node.d
    
    @property
    def value(self):
        return self.bt_Z_node.value

class TernaryTree(TreeSkeleton):
    def __init__(self, u, leaf_label=None) -> None:
        super().__init__()
        self.node_label = None
        self.leaf_label = leaf_label
        self._centre = None
        self._u = None
        self.u_unique_id = None
        self.u = u # binary tree node

        # u.s = self # link binary node to this ternary node

        self.big_omega = None
        self.psi = None
        self.w = np.ones(2, dtype=DTYPE)
        self.w_prime = np.ones(2, dtype=DTYPE)
        self.height_prime = None
        self.is_contraction = True if isinstance(u, BinaryTreeContraction) else False

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, node):
        self._u = node

        if node is not None:
            self.u_unique_id = node.unique_id

            # link binary node to this ternary node
            # set ternary link to self on this node
            node.s = self

    @property
    def centre(self):
        return self._centre

    @centre.setter
    def centre(self, child):
        self._centre = child
        self._centre.parent = self

    def update_height(self):
        left_height = 0 if self.left is None else self.left.height
        centre_height = 0 if self.centre is None else self.centre.height
        right_height = 0 if self.right is None else self.right.height

        self.height = 1 + np.max([left_height, right_height, centre_height])

    def get_id(self):
        return self.u.get_id()
    
    @property
    def tau(self):
        return self.u.tau

    @property
    def kappa(self):
        return self.u.kappa
    
    @property
    def value(self):
        return self.u.value
    
    def get_height_prime(self):
        if self.height_prime is None:
            self.update_height_prime()

        return self.height_prime

    def update_height_prime(self):

        if self.node_label is None:
            # this is a leaf
            self.height_prime = 1
        else:
            left_height = self.left.get_height_prime()
            centre_height = self.centre.get_height_prime()
            right_height = self.right.get_height_prime()

            if self.node_label == TNodeN:
                max_height = np.max([left_height, centre_height, right_height])
                self.height_prime = 1 + max_height

            elif self.node_label == TNodeL:
                max_height = np.max([left_height, centre_height])

                if right_height > max_height:
                    self.height_prime = right_height
                else:
                    self.height_prime = 1 + max_height

            elif self.node_label == TNodeR:
                max_height = np.max([right_height, centre_height])

                if left_height > max_height:
                    self.height_prime = left_height
                else:
                    self.height_prime = 1 + max_height

        return

    def update_heights(self):
        self.update_height()
        self.update_height_prime()

    def update_all(self):
        self.update_label()
        self.update_heights()
    
    def update_label(self):
        """updates the node label on the given node"""

        if self.left is None:
            # leaf node
            self.node_label = None
        else:
            left_leaf = True if self.left.leaf_label is not None else False
            right_leaf = True if self.right.leaf_label is not None else False

            left_terminal = True if self.left.leaf_label == TLeafN else False
            right_terminal = True if self.right.leaf_label == TLeafN else False

            if left_terminal:
                self.node_label = TNodeL
            elif right_terminal:
                self.node_label = TNodeR
            elif not left_leaf and self.left.node_label != TNodeN:
                self.node_label = TNodeL
            elif not right_leaf and self.right.node_label != TNodeN:
                self.node_label = TNodeR
            else:
                self.node_label = TNodeN

    def get_path(self):
        return super().get_path(is_binary=False)

def _get_path_binary(node, child_node, path=None):

    if path is None:
        path = []

    if node.left == child_node:
        path.append(LEFT)
    else:
        path.append(RIGHT)

    if node.parent is None:
        return path
    else:
        path = _get_path_binary(node.parent, node, path)

    return path

def _get_path_ternary(node, child_node, path=None):

    if path is None:
        path = []

    if node.left == child_node:
        path.append(LEFT)
    elif node.centre == child_node:
        path.append(CENTRE)
    else:
        path.append(RIGHT)

    if node.parent is None:
        return path
    else:
        path = _get_path_ternary(node.parent, node, path)

    return path
