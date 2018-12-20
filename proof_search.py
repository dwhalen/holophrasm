'''
This is the implementation of the proof search algorithm.  I'm pretty sure that
I know this well enough to just flat out implement it.

The tricky part is going to be merging of the threading and multiprocessing
libraries.  I predict that this will be ugly.

We're going to make all of the interface calls properties of the threads.

I assume that the payouts range from 0 to 1, with proven nodes returning 1.0
'''

# we make these global variables because we want them to
# be accessible after forking.
global_interface = None
global_context = None
global_problem = None
global_using_threads = None

inf = float('inf')

import threading
import multiprocessing
import signal
from interface import *
import heapq
import time
import naive_tree_search_problem as tsp
import tree_parser
import traceback
import write_proof
import last_step



from IPython.display import clear_output

BEAM_SIZE = 10
VERBOSE = False
def printv(*x):
    if VERBOSE:
        print(x)

# value for UCT is calculated as:
# c.value/(c.visits + GAMMA * c.visiting_threads)
# + BETA * c.prob/(1.0 + c.visits)
# +ALPHA * np.sqrt(np.log(self.visits)/(1.0+c.visits))
# we want to try everything with p > 0.05 when starting with a value of 0.5
# so we should set BETA = 10
CHECK_TAUTOLOGIES = True
CHECK_LAST_STEP = True
APPLY_EASY_PROPS_FIRST = False  # with this, all of the constrained propositions are applied immediately at the first (second) visit
REDUCED_TREE_VALUE = True  # whether the prover uses the reduced-tree formalism (only considering the least-promising child)
HYP_BONUS = 3.0

ALPHA = 1.0
BETA = 0.5
GAMMA = 3.0 # penalty to currently considered paths
DELTA = 4.0 # the depth at which the value is halved
def valuation_function(child_value, child_visits, visits, child_prob, visiting_threads, fix_payout=None):
    score = fix_payout if fix_payout is not None else child_value/(child_visits + GAMMA * visiting_threads)
    return (score #* DELTA/(DELTA+np.log(child_visits))
            + BETA * child_prob/(1.0 + child_visits)
            +ALPHA * np.sqrt(np.log(visits)/(1.0+child_visits)))
def depth_cost(value, depth):
    # return value * DELTA / (DELTA + depth)
    return value

def desired_children(num_visits):
    return 0.01 + num_visits/6.0  # maybe this will be better
    #return (1.0+num_visits) ** 0.75

''' some auxiliary functions for the printing of proof trees '''
# def print_tree(tree, instance):
#     string = tree_parser.tree_to_string(tree, instance.language_model.database, instance.context)
#     return ' '.join(string)

def print_pp(tree, depth):
    string = tree_parser.tree_to_string(tree, global_problem.lm.database, global_context)
    string =  ' '.join(string)
    #string = string.replace(" ", "") # so that the display fits on one line
    return string


''' copies of the interface functions rewritten to
include the global variables '''
def global_get_payout(tree):
    try:
        return global_interface.get_payout(tree, global_context)
    except:
        print('ERROR IN GET PAYOUT')
        print(tree)
        print(('%s: %s' % ('test', traceback.format_exc())))
def global_apply_prop(tree, prop_name):
    try:
        return global_interface.apply_prop(tree, global_context, prop_name, n=BEAM_SIZE)
    except:
        print('ERROR IN APPLY PROP')
        print(tree, prop_name)
        print(('%s: %s' % ('test', traceback.format_exc())))
def global_props(tree):
    try:
        return global_interface.props(tree, global_context)
    except:
        print('ERROR IN PROPS')
        print(tree)
        print(('%s: %s' % ('test', traceback.format_exc())))

''' some stuff for multithreading.
I would have expected Pool to work with with.  Maybe
I'm missing something?'''
def init_func():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class withPool:
    def __init__(self, procs):
        self.p = multiprocessing.Pool(procs, init_func)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        #print 'closing pool'
        self.p.close()
        self.p.terminate()  # I have no idea why the fuck this needs to be here, but otherwise everything has a 50% chance of breaking
        #print 'requested close'
        self.p.join()
        #print 'finished join'
        self.p = None
    def apply(self, *args, **kwargs):
        return self.p.apply(*args, **kwargs)



''' the threading stuff '''
class myThread (threading.Thread):
    def __init__(self, name, problem, multi=False):
        threading.Thread.__init__(self)
        self.finished = False
        self.name = name
        self.multi = multi
        self.problem = problem
        if multi:
            #self.p = multiprocessing.Pool(1,init_func)
            #self.p = withPool(1)
            # pool.apply(time.sleep, (10,))
            # self.p.start()
            pass
        else:
            self.p=None

    ''' these functions are all defined again to make the reference to the
    thread's pool cleaner '''
    def get_payout(self, tree):
        #print ' '*0+str(self.name)+' starting payout'
        out = self.p.apply(global_get_payout, (tree,)) if self.multi else global_get_payout(tree)
        #print ' '*0+str(self.name)+' stopping payout'
        return out

    def apply_prop(self, tree, prop_name):
        #print ' '*30+str(self.name)+' starting gen'
        out = self.p.apply(global_apply_prop, (tree,prop_name)) if self.multi else global_apply_prop(tree, prop_name)
        #print ' '*30+str(self.name)+' stopping gen'
        return out

    def props(self, tree):
        #print ' '*60+str(self.name)+' starting prop'
        out = self.p.apply(global_props, (tree,)) if self.multi else global_props(tree)
        #print ' '*60+str(self.name)+' stopping prop'
        return out

    def run(self):
        if global_problem.done():
            #print 'Should not be running.  Problem already done', self.name
            return
        # print "Starting " + self.name, time.time()

        if self.multi:
            #print  "is multi, about to call pool " + self.name

            #with multiprocessing.Pool(1,init_func) as self.p:
            with withPool(1) as self.p:
                #print 'Created process for'+self.name
                while not self.problem.done():
                    #print 'stepping'+self.name
                    self.problem.visit()
                    #print 'end stepping'+self.name
                #print
                #print 'Terminating process for'+self.name
            #print 'Terminated process for'+self.name
            self.p = None
        else:
            while not self.problem.done():
                self.problem.visit()

        # print "Exiting " + self.name
        self.finished = True

class TypeA:
    ''' a type A node is a tree.
    It has children that are type B nodes'''
    def __init__(self, tree, depth, proven=False,label=None):
        self.dead = False
        if proven:
            # this Type A node has already been proven, probably because
            # it was one of the original hypotheses
            self.depth = depth
            self.tree = tree
            self.value = 1.0
            self.visits = 1
            self.children = []
            self.initial_payout = 1.0
            self.proven = True
            self.label = label
            self.is_hypothesis = True

            # these should never be used
            self.modification_lock = threading.Lock()
            self.children_lock = threading.Lock()
            return
        self.is_hypothesis = False
        self.label = None
        self.depth = depth
        self.tree = tree
        if global_using_threads:
            self.initial_payout = threading.current_thread().get_payout(self.tree)
        else:
            self.initial_payout = global_get_payout(self.tree)
        self.initial_payout = depth_cost(self.initial_payout, self.depth)
        self.value = self.initial_payout
        self.visits = 1
        self.modified_visits = 1
        self.children = []
        self.proven = False
        self.modification_lock = threading.Lock()
        self.children_lock = threading.Lock()
        self.childless_visits = 0

        # controlled by heap_lock
        self.in_queue = 0 # the number of things from the heap that are being processed

        ''' self.heap stores the potential new propositions to apply.  Entries
        are of the form (-log probability, prop_label, tree or None) '''
        self.heap_lock = threading.Lock()
        self.heap = None

        # do the tautology checking now.
        if CHECK_TAUTOLOGIES and not CHECK_LAST_STEP:
            taut = global_interface.is_tautology(self.tree, global_context)
            if taut is None:
                self.tautology = False
            else:
                self.tautology = True
                # add the blue child immediately.
                b = TypeB([], np.exp(0.0), self, taut)
                self.children.append(b)
                self.update_proven()
                printv('added tautology:', taut," ", print_pp(self.tree, None)   )

        elif CHECK_LAST_STEP:
            out = last_step.is_easy(self.tree, global_context, global_problem.lm)
            if out is not None:
                label, hyps = out
                b = TypeB(hyps, np.exp(0.0), self, label)
                self.children.append(b)
                self.update_proven()
                assert self.proven
                printv('added last_step:', label," ", print_pp(self.tree, None)   )

    def update_proven(self):
        if any(c.proven for c in self.children):
            self.proven = True
            # check whether we already knew about it
            # global_problem.tsp.add(self)

            # self.prune()

    def prune(self):
        # this prunes the tree down, remove unproven children
        with self.children_lock:
            for c in self.children:
                if c.proven:
                    self.children = [c]
                    return

    def create_child(self, child_params, parent_trees=None):
        nlp, label, tree = child_params
        lp = -nlp
        # print self.heap
        # print 'creating child from', child_params, 'avoiding', parent_trees

        if tree is None:
            #lptrees = threading.current_thread().apply_prop(self.tree, label)
            if global_using_threads:
                lptrees = threading.current_thread().apply_prop(self.tree, label)
            else:
                lptrees = global_apply_prop(self.tree, label)
            #lptrees = [(x+lp, y) for x, y in lptrees]
        else:
            # we've already expanded this one
            lptrees = [(lp, tree)]

        child = None
        while child is None and len(lptrees)>0:
            assert len(lptrees)>0
            lp_new, trees = lptrees.pop(0)
            if not any(t in parent_trees for t in trees):
                child = TypeB(trees, np.exp(lp), self, label)
            else:
                printv('FAILED TO CREATE CHILD: CIRCULARITY WHEN APPLYING', label, 'TO', print_pp(self.tree, None))
        if child is None:
            # We're still going to count this as a visit with value 0, just to discourage continued exploration
            # around this node.
            printv('FAILED TO CREATE CHILD: NO TREES WHEN APPLYING OR CIRCULAR', label, 'TO', print_pp(self.tree, None))
            self.childless_visits += 1
        # print 'child', [c.tree for c in child.children]
            # else:
            #     print 'abandoned child', trees
            # with self.children_lock:
            #     self.children.append(child)

        if len(lptrees)>0:
            with self.heap_lock:
                # add the rest of the items back onto the heap
                for lptree in lptrees:
                    this_lp, this_tree = lptree
                    if any(t in parent_trees for t in this_tree):
                        continue
                    this_lp = this_lp+lp-lp_new #
                    heapq.heappush(self.heap, (-1.0*this_lp, label, this_tree))
        return child

    def attempt_to_add_child(self, next_child, parent_trees):
        child = self.create_child(next_child, parent_trees=parent_trees+[self.tree])
        #print 'child', child
        if child is not None:
            with self.children_lock:
                self.children.append(child)
            with self.heap_lock:
                self.in_queue -= 1
            return (child.value, child.visits)
        else:
            #print 'Caught child but it was None'
            with self.heap_lock:
                self.in_queue -= 1
            # TODO: it's possible that I should keep trying things until they work
            return None

    def apply_easy_props(self, parent_trees):
        with self.heap_lock:
            # this figures out all the propositions that are easy and adds them
            # immediately.  This will hopefully give us a performance boost.  Maybe.
            old_heap = self.heap
            self.heap = []
            children_to_add = []
            for child_params in old_heap:
                nlp, label, tree = child_params
                if label in global_problem.lm.constrained_propositions:
                    children_to_add.append(child_params)
                else:
                    heapq.heappush(self.heap, child_params)
            self.in_queue += len(children_to_add)
            self.modified_visits = len(children_to_add)

        #print 'children to add: ', children_to_add
        for next_child in children_to_add:
            self.attempt_to_add_child(next_child, parent_trees)
            self.update_proven()
            self.update_value()
            if self.proven:
                break

    def visit_next_child(self, parent_trees):
        '''
        let's try something different: keep track of how many
        children we want to have as a function of the number of visits.
        '''

        #if desired_children(self.visits) > len(self.children) and len(self.heap) > 0:
        with self.children_lock:
            self.remove_dead_children()
            #min_child_visits = min(c.visits for c in self.children) if len(self.children)>0 else 1000

        if APPLY_EASY_PROPS_FIRST and self.visits == 1:
            self.apply_easy_props(parent_trees)
            if len(self.children) > 0:
                return True


        with self.heap_lock:
            #if min_child_visits > 1 and len(self.heap) > 0:
            if (desired_children(self.visits) > len(self.children)+self.childless_visits or len(self.children)==0) and len(self.heap) > 0:
            # pull a new child from the heap
                next_child = heapq.heappop(self.heap)
                self.in_queue += 1
                has_child = True
            else:
                has_child = False

        # if we managed to catch a child:
        if has_child:
            return self.attempt_to_add_child(next_child, parent_trees)

        if len(self.children)>0:
            old_scores = np.array(
                    [valuation_function(c.value, c.visits, self.visits, c.prob, c.visiting_threads)
                    for c in self.children])
            best_old_score_index = np.argmax(old_scores)
            next_child = self.children[best_old_score_index]
            best_old_score = old_scores[best_old_score_index]
            exists_children = True
            return next_child.visit(parent_trees+[self.tree])
        else:
            self.check_death()
            printv('NODE HAS NO CHILDREN, DEAD?', self.dead)
            return None

    def check_death(self):
        # is the thing really really dead?
        with self.children_lock:
            with self.heap_lock:
                if len(self.heap) == 0 and all(c.dead for c in self.children) and self.in_queue == 0 and not self.proven:
                    self.dead = True

    def remove_dead_children(self):
        # this should be in the children_lock
        for c in self.children:
            if c.dead:
                self.children.remove(c)

    def can_be_visited(self):
        # checks a bunch of things to determine whether this can be visited
        # mostly this avoids visiting nodes where the only child is being considered
        if self.dead: return False
        if self.proven: return False
        if self.heap is None: return True  # hasn't been visited twice
        if len(self.heap) == 0 and len(self.children) == 0: return False
        return True

    def get_props(self):
        # lists all the propositions, and sorts them into a heap
        if global_using_threads:
            labels, log_probs = threading.current_thread().props(self.tree)
        else:
            labels, log_probs = global_props(self.tree)
        log_probs -= np.max(log_probs)
        # print log_probs
        self.heap = []
        for l, p in zip(labels, log_probs):
            heapq.heappush(self.heap,  (-1.0*p, l, None) )

    def visit(self, tree_stack=[]):
        #print 'visiting node with', self.tree, self.proven

        # if we haven't expanded yet, do so.
        with self.heap_lock:
            if self.heap is None:
                self.get_props()

        # figure out what child to visit via UCT.  Possibly expand one
        # of the children
        out = self.visit_next_child(tree_stack)

        # update my parameters based off of the the returned value
        with self.modification_lock:
            self.update_value()
            # if out is not None:
            #     self.value += out[0]
            #     self.visits += out[1]
            #else:
                #print 'failed to create child blue node'
            self.update_proven()
    def update_value(self):
        self.value = self.initial_payout + sum(c.value for c in self.children)
        self.visits = 1 + sum(c.visits for c in self.children)+self.childless_visits
        self.modified_visits = self.visits

    def print_proof(self, prefix, depth):
        if len(self.children)==0:
            if self.proven:
                #print '{1:6.2f}% {0:4.2f} {2:4.2f} {3:4} '.format(uct_score, self.prob*100.0, self.value/(self.visits+0.00001), self.visits)
                #print ' '*(8+10) + '{1:4.2f} ! {0:9}'.format('HYP', self.initial_payout)+prefix+str(depth)+' '+print_pp(self.tree, depth)
                print(' '*(8+5) + '{1:4.2f}    1 ! {0:9}'.format('HYP', self.initial_payout)+prefix+str(depth)+' '+print_pp(self.tree, depth))
            else:
                #print ' '*(8+10) + '{1:4.2f}   {0:9}'.format('????', self.initial_payout)+prefix+str(depth)+' '+print_pp(self.tree, depth)
                print(' '*(8+5) + '{1:4.2f}    1   {0:9}'.format('????', self.initial_payout)+prefix+str(depth)+' '+print_pp(self.tree, depth))
            return

        #sorted_children = sorted(self.children)
        #print self.children
        unsorted = [(c.visits + c.value/c.visits, c) for c in self.children]
        unsorted.sort()
        unsorted.reverse()
        _, sorted_children = list(zip(*unsorted))
        # sorted_children = self.children
        sorted_children[0].print_proof(prefix, depth)
        for c in sorted_children[1:]:
            string = ''
            print(' '*(8+10) +'       {0:9}'.format('')+prefix+'or')
            c.print_proof(prefix, depth)

    def generate_mm_format_proof(self):
        if self.label is not None:
            return [self.label] # this is a hypothesis
        xlist = [x for x in self.children if x.proven]
        assert len(xlist)>0
        x = xlist[0]
        assert x.proven
        return x.generate_mm_format_proof()

class TypeB:
    ''' a type B is the application of a proposition to a tree.
    It has children that are type A nodes'''
    def __init__(self, child_trees, prob, parent, label):
        self.parent = parent
        self.label = label
        self.proven = False

        # some locks
        self.modification_lock = threading.Lock()
        self.visits_lock = threading.Lock()
        self.prob = prob

        self.children = [self.create_child(t) for t in child_trees]
        self.visiting_threads = 0 # this adjusts the value for UCT

        self.value = 1.0
        self.visits = 1

        self.dead = False

        with self.modification_lock:
            self.update_proven()
            updated_value = self.update_value()

    def check_death(self):
        # really really dead.
        #with self.modification_lock:
        if any(c.dead for c in self.children):
            self.dead = True

    def create_child(self, tree):
        # check if the child has already been proven.
        child = global_problem.tsp.search(tree)
        if child is None: child = TypeA(tree, self.parent.depth+1)
        return child

    def update_value(self):
        if REDUCED_TREE_VALUE:
            self.update_value_reduced_tree()
        else:
            self.update_value_full_tree()

    def update_value_reduced_tree(self):
        # lock the values and then update them
        self.update_proven()
        self.check_death()
        if self.proven or self.dead:
            return None

        # checks whether the child node with dominent value has changed
        # and if so, potentially propagates things up
        unproven_children = [c for c in self.children if c.can_be_visited()]
        child_values = [c.value/c.visits for c in unproven_children]
        if len(child_values) == 0: return None
        best_child = unproven_children[np.argmin(child_values)]
        #child_values = [c.value/c.visits for c in self.children]
        #best_child = self.children[np.argmin(child_values)]

        # check whether any children are proven
        proven_children = len([c for c in self.children if c.proven])
        bonus_value = proven_children * HYP_BONUS
        #if proven_children > 0: print 'PROVEN CHILDREN BONUS', bonus_value

        # calculate the changes from the current condition
        delta_visits = best_child.visits-self.visits
        delta_value = best_child.value + bonus_value -self.value

        self.value = best_child.value + bonus_value
        self.visits = best_child.visits
        return (delta_value, delta_visits)

    def update_value_full_tree(self):
        self.update_proven()
        self.check_death()
        if self.proven or self.dead:
            return None

        if len(self.children) == 0:
            self.value = 1.0
            self.visits = 1
        else:
            self.visits = sum(c.visits for c in self.children)
            self.value = sum(c.value for c in self.children)

    def visit(self, parent_trees):
        # always visit the child with the lowest *true* value
        child_values = [c.value/c.visits for c in self.children if c.can_be_visited()]
        uproven_children = [c for c in self.children if not c.proven]

        if len(child_values) > 0:
            # actually the worst child.  *that* was an annoying bug.
            best_child = uproven_children[np.argmin(child_values)]

            with self.visits_lock:
                self.visiting_threads += 1

            # visit the child
            best_child.visit(parent_trees)

            with self.visits_lock:
                self.visiting_threads -= 1

        # updates the current node and returns the updated value so that
        # we can propagate up.
        with self.modification_lock:
            self.update_proven()
            updated_value = self.update_value()
        return updated_value

    def update_proven(self):
        if all(c.proven for c in self.children):
            self.proven = True

    def print_proof(self, prefix, depth):
        if self.proven:
            uct_score = 9.99
        else:
            uct_score =valuation_function(self.value, self.visits, self.parent.visits, self.prob, 0)

        string = '{1:6.2f}% {0:4.2f} {2:4.2f} {3:4} '.format(self.value/(self.visits+0.00001), self.prob*100.0, self.parent.initial_payout, self.visits)
        if self.proven:
            string += '!'
        else:
            string += ' '
        if global_problem.lm.database.propositions[self.label].unconstrained_arity() > 0:
            string += '*'
        else:
            string +=' '
        print(string+'{0:9}'.format(self.label[:9])+prefix+str(depth)+' '+print_pp(self.parent.tree, global_context))

        for c in self.children:
            c.print_proof(prefix + '| ', depth+1)

    def generate_mm_format_proof(self):
        prop = global_problem.lm.database.propositions[self.label]
        child_trees = [c.tree for c in self.children]
        fit = global_problem.lm.reconstruct_fit(self.parent.tree, child_trees, self.label)
        assert fit is not None  # this worked the first time

        out = []
        next_out = 0

        for h in prop.hyps:
            if h.type == 'e':
                x = self.children[next_out]
                next_out+=1
                out += x.generate_mm_format_proof()
            else:
                var = h.label
                assert var in fit  # fit should have all mandatory variables
                out+=fit[var].right_list()

        out.append(self.label)
        return out






class ProofSearcher:
    def __init__(self, prop, lm, tree=None, directory='searcher', timeout=None):
        # timeout is in minutes
        self.start_time = time.time()
        self.timeout = timeout

        self.lm = lm
        self.directory = directory

        # the number of passes
        self.passes = 0
        self.max_passes = None
        self.pass_lock = threading.Lock()

        # set up the threading lock for printing
        self.print_lock = threading.Lock()
        self.last_print_time = time.time()
        self.print_frequency = 10.0

        # set up the globals
        global global_interface
        global global_context
        global global_problem
        global global_using_threads
        global_using_threads = False

        if global_interface is None:
            print(global_interface)
            global_interface = ProofInterface(lm, directory=directory)
        global_context = lm.standardize_context(prop)
        global_problem = self
        self.context = global_context

        # define the tree in terms of the context
        if tree is None: tree=global_context.tree
        global_context.tree = tree
        global_interface.initialize_payout(global_context)

        # build the search database
        self.tsp = tsp.ExactSearchProblem()
        for hyp in global_context.hyps:
            if hyp.type == 'f':
                continue
            node = TypeA(hyp.tree, None, proven=True, label=hyp.label)
            self.tsp.add(node)
            if hyp.tree == tree:
                # Oh, look.  We're already done.
                self.root = node

        # build the root node
        self.root = TypeA(tree, 0)

    def run(self, passes, multi=False, threads=None, print_output = True, clear_output=True):
        self.print_output = print_output
        self.clear_output = clear_output

        global global_using_threads
        global_using_threads = not (threads is None)

        # set the ending condition
        self.max_passes = self.passes + passes

        # start by printing the current tree
        self.print_proof(force=True)

        if global_using_threads:
            #threaded
            # build the threads
            self.threads = []
            for i in range(threads):
                t = myThread(i, self, multi=multi)
                t.start()
                self.threads.append(t)
            #print 'threads:', len(self.threads), self.threads,

            # now wait for the threads to finish
            for t in self.threads:
                #print 'joined', t.name
                t.join()
        else:
            # unthreaded
            while not self.done():
                self.visit()

        self.print_proof(force=True)

    def print_proof(self, force=False):
        # skip this if something is already printing
        if time.time()-self.last_print_time < self.print_frequency and not force:
            return
        if self.print_lock.locked()  and not force:
            return
        with self.print_lock:
            # iPython only
            if self.clear_output:
                clear_output()
            if self.root.proven:
                print('PROVEN')
            self.last_print_time = time.time()
            if self.print_output:
                print('Current proof after {0} / {1} passes'.format(self.passes, self.max_passes))
                self.root.print_proof('', 0)

    def visit(self):
        with self.pass_lock:
            self.passes += 1

        self.root.visit()


        self.print_proof()

    def done(self):
        elapsed_time = time.time()-self.start_time
        if self.timeout is not None and elapsed_time > self.timeout * 60:
            print('search ended: reached timeout of {0} minutes'.format(self.timeout))
            return True
        return (self.passes >= self.max_passes) or self.root.proven or self.root.dead

    def proven(self):
        return self.root.proven

    def generate_mm_format_proof(self):
        self.root.prune()
        string = self.root.generate_mm_format_proof()
        #print 'string', string
        # now we substitute the variable constructors back in.
        dereplace = {v: k for k, v in self.context.replacement_dict.items() if k in self.context.mandatory}
        #print dereplace
        string = [label if label not in dereplace else dereplace[label]
            for label in string]

        string = ' '.join(string)
        print(string)
        return string

    def write(self):
        # writes to the modified set.mm file.
        write_proof.write({global_context.label:self.generate_mm_format_proof()})

    def proof_object(self):
        assert self.proven()
        out = write_proof.Proof(self.context.label, self.generate_mm_format_proof(), self.passes)

        # write the proof.  Why not?
        out.save(directory=self.directory)
        return out
