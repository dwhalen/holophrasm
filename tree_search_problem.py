''' This builds up a database of trees and allows for us to efficiently search through
them.  In practice, this may actually be slow for small numbers of trees, but whatever '''




# this allows us to feed in a tree and will return the tree if it's in the database.
# otherwise it will return None
# it accepts arbitrary objects as long as they have an object.tree property
class ExactSearchProblemNode:
    def __init__(self, next_position):
        self.terminal = False
        self.next_position = next_position
        self.dict = {}
        self.value = None  # only true if terminal node

    def add(self, value):
        if self.value is None and len(self.dict)==0:
            # empty
            self.terminal = True
            self.value = value
            return

        if self.terminal:
            assert not value == self.value
            self.terminal = False
            self.add_to_dictionary(value)
            self.add_to_dictionary(self.value)
            self.value = None
            return

        self.add_to_dictionary(value)

    def add_to_dictionary(self, value):
        value_tree_value = value.tree.value_at_position(self.next_position)
        even_nexter_position = value.tree.next_breadth_position(self.next_position)
        # add it to the dictionary if it's not already there.
        if not value_tree_value in self.dict:
            self.dict[value_tree_value] = ExactSearchProblemNode(even_nexter_position)

        self.dict[value_tree_value].add(value)

    def search(self, tree):
        if not (len(self.dict)>0 or not self.value is None): # we should never be searching an empty thingy
            return None

        # the terminal case is easy
        if self.terminal:
            if tree == self.value.tree:
                return self.value
            else:
                return None

        # recurse for the nonterminal case
        value_tree_value = tree.value_at_position(self.next_position)
        if not value_tree_value in self.dict:
            return None

        return self.dict[value_tree_value].search(tree)

class ExactSearchProblem:
    def __init__(self):
        self.start = ExactSearchProblemNode(())
        self.size = 0
        self.searches = 0

    def add(self, value):
        #print 'Searcher: adding',value.tree.stringify()
        self.start.add(value)
        self.size += 1

    def search(self, tree): # search by tree: otherwise this defeats the point
        self.searches += 1
        return self.start.search(tree)
