'''
Performs a beam search with given parameters.

Takes as input a
interface: a BeamSearchInterface object
params: a set of parameters
n: the number of things to return
k: the number of symbols to expand at each step.
'''

''' the following actions are needed for instance:
instance.next_child(self, must_be_better_than): returns the next child and presteps
instance.step(): prep for the next step.
instance.finalize(): returns the corresponding trees
instance.__cmp__(other): compares the current state of the two things

instance.value: the current value of the state
instance.complete: whether the state is finished
'''

import heapq

class BeamSearcher:
    # an object that allows us to perform beam searches to fetch additional
    # searches (although I don't think we'll ever need that functionality)

    def __init__(self, interface, params):
        self.instance = interface.instance(params)

    def best(self, width, k, num_out):
        # performs the simple nonextensible beam search
        assert width>0
        assert k>0
        assert num_out>0
        out = []

        heap = [self.instance]

        while len(out) < num_out and len(heap)>0:
            # print heap
            old_heap = [heapq.heappop(heap) for i in range(len(heap))]
            old_heap.reverse()
            heap = []

            for old in old_heap:
                #print 'old', old
                # if len(heap)>0:print heap[0].value
                for _ in range(k): # at most k times.
                    if len(heap)>=width:
                        child = old.next_child(heap[0])
                        #print 'child', child
                        if child is None: break
                        heapq.heappushpop(heap, child)
                    else:
                        child = old.next_child(None)
                        #print 'child', child
                        if child is None: break
                        heapq.heappush(heap, child)

            # now step everything
            for h in heap:
                assert not h.complete
                h.step()
                if h.complete:
                    out.append(h)
            heap = [x for x in heap if not x.complete]
            #print heap

        out.sort()
        out.reverse()
        return [x.finalize() for x in out]
