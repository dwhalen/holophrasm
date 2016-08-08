''' This builds up a database of trees and allows for us to efficiently search through
them.  This should work well for small numbers of trees, but poorly for larger numbers. '''

class ExactSearchProblem:
    def __init__(self):
        self.objects = []
        self.size = 0
        self.searches = 0

    def add(self, value):
        #print 'Searcher: adding',value.tree.stringify()
        if self.search(value.tree) is not None:
            return
        self.objects.append(value)
        self.size += 1

    def search(self, tree): # search by tree: otherwise this defeats the point
        self.searches += 1
        for ob in self.objects:
            if tree==ob.tree:
                return ob
        return None
