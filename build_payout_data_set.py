import pickle as pickle
import interface
import withpool
import data_utils5 as data
import heapq
import numpy as np

import time

'''
Figure out the data set.
For each validation proposition, do the following:
    list all the entails proof steps
    for each proof step:
        find the ten best next steps using pred and gen
        list all of the above tree, save these as 'wrong' trees
        find the correct next step, save as 'correct' tree.
    remove all of the correct trees from the wrong tree list if they appear
    save the context expressions, the correct expressions, and the wrong expressions.
'''

BEAM_SIZE = 1
WRONG_SAMPLES = 2

global_interface = None
global_lm = None

def slow_delete_duplicates(inlist):
    out = []
    for i in inlist:
        if i not in out:
            out.append(i)
    return out

def initialize_interface(lm, directory):
    global global_interface
    global global_lm
    global_interface = interface.ProofInterface(lm, directory=directory, skip_payout=True)
    global_lm = lm

class PropositionsData:
    def __init__(self, prop):
        '''
        This creates the data for a single proposition.
        '''

        start = time.time()
        lm = global_lm
        assert global_interface is not None
        context = lm.standardize_context(prop)

        self.wrong = []
        self.correct = []
        self.hyps = [h.tree for h in context.hyps if h.type == 'e']
        # self.f = context.f

        proc_trees = []
        for t in prop.entails_proof_steps:
            if t.tree in proc_trees:
                continue # don't do duplicate trees
            else:
                proc_trees.append(t.tree)

            # add the wrong steps
            this_tree = t.tree.copy().replace_values(context.replacement_dict)
            new_wrong = get_wrong(context, this_tree)
            self.wrong += new_wrong
            # # the proof step is just an e-hypothesis
            # if t.prop.type == 'e':
            #     print 'e-hypothesis'
            #     print t.tree
            #     print t.prop.tree
            #     continue

            # add the correct steps
            correct_pre_sub = get_correct(prop, t)
            self.correct += [tree.replace_values(context.replacement_dict) for tree in correct_pre_sub]

        # slow, but whatever.
        self.wrong = [tree for tree in self.wrong if tree not in self.correct]
        self.wrong = slow_delete_duplicates(self.wrong)
        self.correct = slow_delete_duplicates(self.correct)

        print(prop.label, time.time()-start, 'with hyps/correct/wrong', len(self.hyps),len(self.correct),len(self.wrong))
        # if any('wps' in tree for tree in self.correct):
        #     print 'WPSWPSWPSWPSWPSWPSWPSWPSWPSWPSWPSWPS'
        #     print self.correct, prop.f
        #     for t in prop.entails_proof_steps:
        #         print t.tree
        #         print t.unconstrained


def get_correct(context, t):
    '''
    gets the correct hypotheses from this context.
    context is the un-standardized context
    '''
    prop = t.prop
    fit = data.prop_applies_to_statement(t.tree, prop, context)
    #print fit
    assert fit is not None # the correct fit needs to work

    for var, tree in zip(prop.unconstrained_variables, t.unconstrained):
        fit[var] = tree

    hyps = [h.tree.copy().replace(fit) for h in prop.hyps if h.type == 'e']
    #print 'hyps', hyps
    return hyps

def get_wrong(context, tree):
    '''
    context is the standardized context
    '''

    # generate the prop list
    labels, log_probs = global_interface.props(tree, context)
    log_probs -= np.max(log_probs)
    heap = []
    for l, p in zip(labels, log_probs):
        heapq.heappush(heap,  (-1.0*p, l, None) )

    # and now extract elements from the heap until we're done
    out = []
    while len(out)<WRONG_SAMPLES and len(heap)>0:
        child = process_child(tree, context, heap)
        if child is not None:
            out+=child
    return out

# we ignore the parent tree restriction: too annoying to keep track of
def process_child(selftree, context,  heap):
    child_params = heapq.heappop(heap)
    nlp, label, tree = child_params
    lp = -nlp

    if tree is None:
        lptrees = global_interface.apply_prop(selftree, context, label, n=BEAM_SIZE)
    else:
        # we've already expanded this one
        lptrees = [(lp, tree)]

    child = None
    while child is None and len(lptrees)>0:
        lp_new, trees = lptrees.pop(0)
        # print child_params[1], trees
        child = trees # we just need the list of trees

    # and now (possibly) add things back to the heap
    if len(lptrees)>0:
        # add the rest of the items back onto the heap
        for lptree in lptrees:
            this_lp, this_tree = lptree
            this_lp = this_lp+lp-lp_new #
            heapq.heappush(heap, (-1.0*this_lp, label, this_tree))
    return child
