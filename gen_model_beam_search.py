'''
This is a bunch of stuff to start a beam search using gen_model_train as a
baseline.  I'll try to implement augmentation and structure_data as well.
I'll additionally try to see how feasible it is to do an extendable beam search,
where we remember certain vectors for later.

We'll additionally incorporate ensembling, letting the calling function make this
an autoensemble if it wants.
'''

import gen_model_train as model
import numpy as np
import data_utils5 as data_utils
from models import *
import nnlibrary as nn
from tree import *

Config = model.Config
Variables = model.Variables

# parameters for what we'll permit when looking at new variables
PERMIT_NEW_CLASSES = False
PERMIT_NEW_WFFS = False
PERMIT_NEW_SETS = True

MAX_BEAM_SEARCH_LENGTH = 75

class BeamSearchInterface:
    ''' this is an object that acts as an interface between the beam search
    algorithm and the problem-specific stuff.'''
    def __init__(self, vs):
        # load the variables
        self.config = vs[0].config  # mostly for the language_model
        # self.config.load(file_path+'/train.parameters')
        # self.v = model.Variables(self.config)
        # self.v.load(file_path+'/train.weights')
        self.vs = vs

    def instance(self, params):
        ''' in this case, params = tree, context, prop_name '''
        tree, context, prop_name, allowed, return_replacement_dict = params
        prop = self.config.lm.database.propositions[prop_name]

        # verify that we've standardized the context
        assert all(label in self.config.lm.new_names for label in context.f)

        return BeamSearchState(tree, context, prop, self.config, self.vs, allowed_constructors=allowed, return_replacement_dict=return_replacement_dict)

# BeamSearchState stores all of the information neccessary for a single partially
# completed state in the beam.
''' the following actions are needed:
BeamSearchState.get_children(min_cost, max_num): returns a list of children after prestepping
BeamSearchState.next_child(self, must_be_better_than)
BeamSearchState.step(): prep for the next step.
BeamSearchState.finalize(): returns the corresponding trees
BeamSearchState.__cmp__(other): compares the current state of the two things
'''
class BeamSearchState:
    def __init__(self, tree, context, prop, config, vs,
            allowed_constructors=None, beam_search_state=None,
            return_replacement_dict=False):
        '''
        init just builds the beam search state, and we'll use it for copying

        inputs:
            vs: a set of variable objects, for all the models we
                    are ensembling over
            config: the config object
            beam_search_state: a BeamSearchState to copy
        '''
        self.total_symbols = 0 if beam_search_state is None else beam_search_state.total_symbols
        self.return_replacement_dict = return_replacement_dict if beam_search_state is None else beam_search_state.return_replacement_dict

        # basic configuration stuff
        self.vs = vs if beam_search_state is None else beam_search_state.vs
        self.config = config if beam_search_state is None else beam_search_state.config
        self.lm = self.config.lm
        self.tree = tree if beam_search_state is None else beam_search_state.tree
        self.context = context if beam_search_state is None else beam_search_state.context
        self.prop = prop if beam_search_state is None else beam_search_state.prop
        self.value = 0 if beam_search_state is None else beam_search_state.value
        self.complete = False if beam_search_state is None else beam_search_state.complete
        self.allowed_constructors = (self.get_all_allowed_symbols(allowed_constructors)
                if beam_search_state is None
                else beam_search_state.allowed_constructors)

        # the current state (non-vector)
        self.string = [] if beam_search_state is None else beam_search_state.string[:]
        self.position_into_arity_stack = [0] if beam_search_state is None else beam_search_state.position_into_arity_stack[:]
        self.parent_arity_stack = [-1] if beam_search_state is None else beam_search_state.parent_arity_stack[:]
        self.complete = False if beam_search_state is None else beam_search_state.complete
        self.next_symbol = None if beam_search_state is None else beam_search_state.next_symbol

        # information about the current state vectors
        self.h = None if beam_search_state is None else beam_search_state.h
        self.parent_stack = [None] if beam_search_state is None else beam_search_state.parent_stack[:]
        self.left_sibling_stack = [None] if beam_search_state is None else beam_search_state.left_sibling_stack[:]
        self.logits = None if beam_search_state is None else beam_search_state.logits
        self.logit_order = None if beam_search_state is None else beam_search_state.logit_order
        self.parent_symbol_stack = None if beam_search_state is None else beam_search_state.parent_symbol_stack[:]

        # the information for returning symbols
        # this is immediately forgotten
        self.returned_symbols = 0 # this state information dies when we copy.
        self.exhausted = False
        self.new_wff_added = not PERMIT_NEW_WFFS
        self.new_set_added = not PERMIT_NEW_SETS
        self.new_class_added = not PERMIT_NEW_CLASSES
        self.next_vclass = None

        # information about the current model
        # this is kept until we move to the next unconstrained variable
        self.current_models = None if beam_search_state is None else beam_search_state.current_models
        self.disallowed_symbols = None if beam_search_state is None else beam_search_state.disallowed_symbols
        #self.used_symbols = context.hyp_symbols.copy() if beam_search_state is None else beam_search_state.used_symbols.copy()
        self.used_symbols = context.hyp_symbols if beam_search_state is None else beam_search_state.used_symbols.copy()
        self.this_ua = None if beam_search_state is None else beam_search_state.this_ua
        self.model_complete = True if beam_search_state is None else beam_search_state.model_complete
        if beam_search_state is None:
            self.fit = data_utils.prop_applies_to_statement(tree, prop, context)
            self.remaining_uas = [x.label for x in prop.hyps if x.type=='f' and x.label not in self.fit]
            np.random.shuffle(self.remaining_uas)
            self.set_up_next_model()
            if not self.complete:
                self.determine_next_vclass()
        else:
            self.remaining_uas = beam_search_state.remaining_uas[:]
            self.fit = beam_search_state.fit.copy()

    def get_all_allowed_symbols(self, allowed_symbols):
        if allowed_symbols is None: return None
        hyp_symbols = [s.tree.set() for s in self.context.hyps if s.type == 'e']
        hyp_symbols = [symbol for sset in hyp_symbols for symbol in sset]
        tree_symbols = self.tree.set()
        all_symbols = set()
        all_symbols = all_symbols.union(hyp_symbols)
        all_symbols = all_symbols.union(tree_symbols)
        all_symbols = all_symbols.union(self.lm.new_names)
        all_symbols = all_symbols.union(allowed_symbols)
        #print 'gotten all', all_symbols
        return all_symbols

    def finish_model(self):
        # the only thing we need to do is add our new tree
        # to the current fit.
        tree = self.generate_tree()
        self.fit[self.this_ua] = tree

    def set_up_next_model(self):
        '''this sets up the next unconstrained variable to be run,
        going in the predetermined variable order.'''

        # check to see whether we are just done
        if self.complete:
            return
        if len(self.remaining_uas)==0:
            self.complete = True
            return

        self.this_ua = self.remaining_uas.pop()
        # print 'about to generate_disallowed_symbols_list'
        self.generate_disallowed_symbols_list()
        self.model_complete = False
        #self.vclass = self.prop.f[this_ua].vclass # the vclass of the corresponding variable in prop
        self.current_models = [Model(v, self.config, self.tree, self.context, self.fit, self.this_ua, self.prop) for v in self.vs]
        self.h = [model.initial_h for model in self.current_models]

        # feed in the start symbol
        left = [None]*len(self.vs)
        parent = [None]*len(self.vs)
        self.next_symbol = 'START_OUTPUT'
        self.get_logits_and_next_state(left, parent, [-1,-1,-1,-1])

        # and assemble the stacks
        self.parent_symbol_stack = [None]
        self.parent_stack = [None]
        self.left_sibling_stack = [None]

        self.string = []
        self.position_into_arity_stack = [0]
        self.parent_arity_stack = [-1]
        self.complete = False
        self.next_symbol = None

    def copy(self):
        return BeamSearchState(None, None, None, None, None, beam_search_state=self)

    def pre_step(self, symbol, additional_value):
        self.string.append(symbol)
        self.total_symbols += 1
        self.next_symbol = symbol
        self.value += additional_value
        if symbol in self.lm.new_names:
            self.used_symbols.add(symbol)

    def step(self):
        if self.complete: return
        assert self.next_symbol is not None
        symbol = self.next_symbol

        arity = self.config.label_to_arity[symbol]
        left = [None]*len(self.vs) if self.left_sibling_stack[-1] is None else self.left_sibling_stack[-1]
        parent = [None]*len(self.vs) if self.parent_stack[-1] is None else self.parent_stack[-1]
        #structure_data = [depth, parent_arity, leaf_position, arity]
        structure_data = [len(self.parent_arity_stack)-1, self.parent_arity_stack[-1],
                self.position_into_arity_stack[-1], arity]

        self.get_logits_and_next_state(left, parent, structure_data)
        self.parent_stack.append(self.h)
        self.parent_symbol_stack.append(symbol)
        self.left_sibling_stack.append(self.h)

        # update the stacks
        self.position_into_arity_stack[-1]+=1
        self.position_into_arity_stack.append(0)
        self.parent_arity_stack.append(arity)
        self.left_sibling_stack.append(None)

        # Now kill the dead layers in the stack
        while len(self.position_into_arity_stack)>0 and self.position_into_arity_stack[-1]==self.parent_arity_stack[-1]:
            #print len(self.remaining_arity_stack), len(self.left_sibling_stack), len(self.parent_stack)
            self.position_into_arity_stack.pop()
            self.parent_arity_stack.pop()
            self.left_sibling_stack.pop()
            self.parent_stack.pop()
            self.parent_symbol_stack.pop()

        if self.parent_arity_stack[-1] == -1:
            self.model_complete = True

        self.next_symbol = None

        if self.model_complete:
            self.finish_model()         # add the current stuff to fit
            self.set_up_next_model()    # initialize the next ua

        if not self.complete:
            self.determine_next_vclass()

        return

    def determine_next_vclass(self):
        symbol = self.parent_symbol_stack[-1]
        if symbol is None:
            self.next_vclass = self.prop.f[self.this_ua].vclass
        else:
            hyps = self.lm.database.propositions[symbol].hyps
            vclasses = [x.vclass for x in hyps if x.type=='f']
            # self.print_summary()
            # print symbol, vclasses, self.position_into_arity_stack[-1], self.string, self.model_complete, self.complete
            self.next_vclass = vclasses[self.position_into_arity_stack[-1]]

    def __cmp__(self, other):
        if self.value < other.value: return -1
        if self.value == other.value: return 0
        return 1
    def __eq__(self, other):
        return self.value == other.value
    def __neq__(self, other):
        not self.value != other.value
    def __lt__(self, other):
        return self.value < other.value
    def __le__(self, other):
        return self.value <= other.value
    def __gt__(self, other):
        return self.value > other.value
    def __ge__(self, other):
        return self.value >= other.value

    def get_logits_and_next_state(self, left, parent, structure_data):
        # return logits, an actual np array and hs, a list of lists of nodes
        all_out_hs = []
        x=0.0
        #print 'feeding in ', self.next_symbol
        for model, h, l, p in zip(self.current_models, self.h, left, parent):
            out_hs, logits = model.logits_and_next_state(h, p, l, self.next_symbol, structure_data)
            all_out_hs.append(out_hs)
            x = x + logits
        self.h = all_out_hs
        self.logits = x/len(self.vs)
        self.logits = -1 * nn.negative_log_softmax(self.logits) # log probability
        self.logit_order =np.argsort(self.logits)[::-1]  # highest prop to lowest

        # print self.string
        # print self.logit_order
        # print self.logits
        # print [self.config.decode[i] for i in self.logit_order[0:50]]
        # print

    def generate_tree(self):
        assert self.model_complete

        assert len(self.string)>0

        root = Tree(value = self.string[0], leaves=[])
        #print root, root.leaves
        #print root.stringify()
        root.gen_tree_degree =self.config.label_to_arity[self.string[0]]
        tree_stack = [root]
        while len(tree_stack)>0 and len(tree_stack[-1].leaves) == tree_stack[-1].gen_tree_degree:
            tree_stack.pop()
        #print root.stringify()

        # fill in the rest.
        for label in self.string[1:]:
            child = Tree(value=label, leaves=[])
            child.gen_tree_degree = self.config.label_to_arity[label]
            #print 'stackstring', tree_stack, self.string, label
            tree_stack[-1].leaves.append(child)
            tree_stack.append(child)


            while len(tree_stack)>0 and len(tree_stack[-1].leaves) == tree_stack[-1].gen_tree_degree:
                tree_stack.pop()

        #print self.string, root, tree_stack

        assert len(tree_stack) == 0
        assert root.size() == len(self.string)

        self.tree = root
        return root

    def finalize(self):
        # at this point, all of the trees should have already been generated
        # and added to fit.
        # returns a tuple (value, list_of_trees)
        assert self.complete
        if self.return_replacement_dict:
            return self.fit
        else:
            return self.value, [h.tree.copy().replace(self.fit) for h in self.prop.hyps if h.type=='e']

    def next_child(self, must_be_better_than):
        if MAX_BEAM_SEARCH_LENGTH is not None and self.total_symbols >= MAX_BEAM_SEARCH_LENGTH:
            return None   # too long, abort!
        min_logit = must_be_better_than.value-self.value if must_be_better_than is not None else None
        assert self.logits is not None

        while self.returned_symbols < len(self.logits):
            index = self.logit_order[self.returned_symbols]
            if must_be_better_than is not None and self.logits[index] < min_logit:
                return None

            # at this point, we're going to return a symbol no matter what
            self.returned_symbols += 1

            symbol = self.config.decode[index]
            if self.is_valid(symbol):
                out = self.copy()
                out.pre_step(symbol, self.logits[index])
                return out
        return None

    def generate_disallowed_symbols_list(self):
        # note that we only need to do this once per
        # new fit we are trying to create

        # the symbol we are finding a substitution for
        assert self.this_ua is not None

        '''now we're going to consider a bunch of variables.
        for (x, this_symbol) in prop_d, if x in fit, then for
        all t appearing in fit[x], y, a variable, add y unless
        (t,y) is in context_d'''

        # I wish there were a better way to do this.
        context_variables_names = self.lm.new_names
        prop_variable_names = list(self.prop.f.keys())

        distincts = [] # the variables that this needs to be distinct from
        # note that this is larger than the list of variable it can't include
        for prop_v in self.fit.keys():
            # d_labels already considers both directions
            if (prop_v, self.this_ua) in self.prop.d_labels:
                distincts += self.fit[prop_v].list()
        distincts = set(distincts).intersection(context_variables_names)

        # print self.fit
        # print self.prop.d_labels
        # print 'context.d_labels'
        # print sorted(list(self.context.d_labels))
        # print distincts

        self.disallowed_symbols = set(self.config.p.special_symbols).union(self.context.main_but_not_hyp_symbols)
        for context_v in distincts:
            for context_v2 in context_variables_names:
                if (context_v, context_v2) not in self.context.d_labels:
                    # not known to be distinct
                    self.disallowed_symbols.add(context_v2)

        # print self.disallowed_symbols

    def is_valid(self, symbol):
        '''checks whether the symbols can be added
        to the current tree'''

        '''at this point, we should be keeping track of a number
        of things: the context, the current tree (after substitutions),
        all of the current subtitutions (both constrained and
        unconstrained), the list of variables we still need to
        find substitutions for.

        Thing we are not allowed to do:
        violate a d statement
        be a new set/class/wff if we've already added one
        be a special symbol
        have the wrong v-class
        '''

        if self.allowed_constructors is not None and symbol not in self.allowed_constructors:
            # it's a new symbol that's insufficiently common
            return False

        # print symbol, self.used_symbols
        if symbol in self.disallowed_symbols:
            # symbol conflicts with a d-statement
            # print 'INVALID: symbol disallowed'
            return False

        vclass = self.config.lm.symbol_to_vclass[symbol]
        if vclass!=self.next_vclass:
            # symbol has the wrong vclass
            # print 'INVALID: had vclass', vclass, 'but expected', self.next_vclass
            return False

        # print symbol, self.lm.label_to_number[symbol], self.context.number
        if self.lm.label_to_number[symbol] >= self.context.number:
            # this symbol isn't introduced until later, since
            # we won't have access to the axioms to deal with it
            return False

        if symbol not in self.lm.new_names:
            # print 'VALID'
            return True

        if vclass == 'set' and symbol not in self.used_symbols:
            if self.new_set_added:
                # print 'INVAlID: Is new set, but new set already added'
                return False
            else:
                self.new_set_added = True
                return True

        if vclass == 'class' and symbol not in self.used_symbols:
            if self.new_class_added:
                # print 'INVAlID: Is new class, but new class already added'
                return False
            else:
                self.new_class_added = True
                return True

        if vclass == 'wff' and symbol not in self.used_symbols:
            if self.new_wff_added:
                # print 'INVAlID: Is new wff, but new wff already added'
                return False
            else:
                self.new_wff_added = True
                return True

        # print 'VALID'
        return True

    def print_predictions(self, n=20, printfalse=True):
        best = self.logit_order[:n]
        #print best

        probs = np.exp(self.logits[best])
        symbols = [self.config.decode[k] for k in best]
        for p, s in zip(probs, symbols):
            if s in self.lm.database.propositions:
                statement = self.lm.database.propositions[s].statement
                statement = ' '.join(statement)
                valid = self.is_valid(s)
                if not valid and not printfalse:continue
            else:
                statement = ""
                valid = ""
            print('{0:6.4f}/{1:6.4f}: {2:10} {3:20} {4}'.format(p,np.log(p),s, statement, valid))
        # print probs
        # print symbols


    def print_summary(self):
        print('State summary:')
        print('fit', self.fit)
        if self.logits is not None:
            print('predictions')
            self.print_predictions()
        print('parent_arity_stack', self.parent_arity_stack)
        print('left_sibling_stack', self.left_sibling_stack)
        print('parent_stack', self.parent_stack)
        print('position_into_arity_stack', self.position_into_arity_stack)
        print('model_complete', self.model_complete)
        print('complete', self.complete)
        print('string', self.string)
        print('this_ua', self.this_ua)
        print('remaining_uas', self.remaining_uas)
        print('value', self.value)
        print('parent_symbol_stack', self.parent_symbol_stack)
        print()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '<BeamSearchState with value {0:.2f}, {1}>'.format(self.value, self.string)










class Model(model.Model):
    '''This sets up everything that is used for all of the BeamSearchStates,
    that is, the left and the middle part of the model.  We'll create one
    of these for each set of variables.

    This is mostly just copied from model.Model with the right side stuff
    removed

    We're going to keep everything as nnLibrary nodes, because I'm lazy.
    We'll just extract the values exactly when we need them.
    '''
    def __init__(self, variables, config, tree, context, fit, this_ua, prop):
        train = False

        DefaultModel.__init__(self, config, variables, train=train)
        self.g = None  # don't remember all of the stuff in the graph.

        # we don't know the output, so don't try to determine it
        self.parse_and_augment_proof_step(tree, context, fit, this_ua, prop)

        # merge the inputs together so that we can bidirection it
        in_string, in_parents, in_left, in_right, in_params, depths, parent_arity, leaf_position, arity = merge_graph_structures(
                [self.known_graph_structure, self.to_prove_graph_structure],
                [self.v.known_gru_block, self.v.to_prove_gru_block])

        # do the left side gru blocks
        to_middle = self.gru_block(self.v.forward_start, in_string, in_params,
                hs_backward=self.v.backward_start, parents=in_parents,
                left_siblings=in_left, right_siblings=in_right,
                bidirectional=self.config.p.bidirectional,
                structure_data = list(zip(depths, parent_arity, leaf_position, arity)),
                feed_to_attention=self.config.p.attention)

        # set up the attentional model
        if self.config.p.attention:
            self.set_up_attention()

        # process the middle
        from_middle = [nn.RELUDotAdd(x, W, b, self.g)
                for x, W, b in zip(to_middle, self.v.middle_W, self.v.middle_b)]

        self.initial_h = from_middle
        # # and feed in the start token
        # self.initial_h, self.initial_logits = self.logits_and_next_state(
        #     from_middle, [x.no_parent for x in self.v.out_gru_block],
        #     [x.no_left for x in self.v.out_gru_block], 'START_OUTPUT',
        #     [-1,-1,-1,-1]
        # )

    def logits_and_next_state(self, hs, parent, left, input_token, structure_data):
        # structure data = [depth, parent_arity, leaf_position, arity]

        out_hs, x = self.forward_vertical_slice(hs, parent, left, input_token,
                self.v.out_gru_block.forward, structure_data,
                takes_attention=self.config.p.attention)

        logits = self.x_to_predictions(x)

        return out_hs, self.rearrange_logits(logits.value)


    def rearrange_logits(self, logits):
        x = 1*logits

        # DEBUG
        # print self.random_replacement_dict
        # best = np.argsort(x)[::-1]
        # best = best[:20]
        # print best
        #
        # probs = np.exp(x[best])
        # symbols = [self.config.decode[k] for k in best]
        # print probs
        # print symbols
        # print
        # DEBUG

        for key, value in self.random_replacement_dict.items():
            key_location = self.config.encode[key]
            value_location = self.config.encode[value]
            x[key_location] = logits[value_location]

        return x

        # WARNING: THIS MAY BE THE WRONG DIRECTION.  VERIFY IT.

    def parse_and_augment_proof_step(self, tree, context, fit, this_ua, prop):
        ''' this gets the data for the proof step, but also does the
        randomization and the augmentation.  There are a number of
        strange choices in the augmentation, particularly with keeping
        track of the distinct unconstrained variables but we'll leave
        them as is for now. '''

        # figure out the unconstrained variables in prop, create
        # the corresponding replacement rules and such.
        # Eventually I'll distinguish the different non-target variables
        # but that is a task for another day.
        unconstrained_variables = prop.unconstrained_variables
        unconstrained_variables = [uv for uv in unconstrained_variables if uv not in fit]
        uv_dict = {var:'UC' for var in unconstrained_variables}
        uv_dict[this_ua] = 'TARGET_UC'

        # figure out the fit of prop to the proof step.
        # we already know the fit
        self.to_prove_trees = [hyp.tree.copy().replace(fit).replace_values(uv_dict)
            for hyp in prop.hyps if hyp.type == 'e']
        self.known_trees = [hyp.tree.copy()
            for hyp in context.hyps if hyp.type == 'e']

        # now generate a random replacement dictionary
        # note that we need to standardize
        random_replacement_dict = self.config.lm.random_replacement_dict_f(f=None)
        self.random_replacement_dict = random_replacement_dict

        # perform the replacements
        for tree in self.to_prove_trees:
            tree.replace_values(random_replacement_dict)
        for tree in self.known_trees:
            tree.replace_values(random_replacement_dict)

        # and get the graph structures
        self.known_graph_structure = TreeInformation(self.known_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        self.to_prove_graph_structure = TreeInformation(self.to_prove_trees,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
