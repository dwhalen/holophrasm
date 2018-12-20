from tree_parser import *
#from model import *
import numpy as np

import random

"""
The data utilities take a statement in a proof, assigns each
node a number from 0 to number_of_nodes_placeholder-1 with 0
being the root node:

    number_of_nodes_placeholder:    shape=()
            the number of nodes in the tree
    children_placeholder:           shape=(None,max_axiom_arity)
            a list of the number associated to the children of the node
    prop_number_placeholder:        shape=(None)
            the number associated to the proposition at that node with that arity
    arity_placeholder:              shape=(None)
            the arity of the corresponding node
"""
language_model_extra_variables_of_each_type = 0  # Chosen arbitrarily

# a minimal class, which we can use to build objects.
class Container:
    def __init__(self):
        pass

class LanguageModel:
    def __init__(self, database):
        """Builds a number of dictionaries equal to max_axiom_arity"""
        self.max_axiom_arity = max([p.arity() for p in database.non_entails_axioms.values()]) + 1  # one more than the max

        self.database = database

        self.max_unconstrained_arity = 10000000
        self.searcher = SearchProblem(database, max_unconstrained_arity = self.max_unconstrained_arity)

        self.tautologies = set()
        for p in self.database.propositions.values():
            e_hyps = [h for h in p.hyps if h.type == 'e']
            if p.vclass=='|-' and len(e_hyps) == 0:
                self.tautologies.add(p.label)
        print('tautologies:', len(self.tautologies))

        # the propositions with trivial unconstrained arity.  That is, the ones
        # that are really easy to apply.
        self.constrained_propositions = set(
                p.label for p in self.database.propositions.values()
                if p.vclass == '|-' and p.unconstrained_arity() == 0
                )

        # figure out the names of the read variables
        # self.real_wff_names = set()
        # self.real_set_names = set()
        # self.real_class_names = set()
        # real_name_dict = {'wff': self.real_wff_names, 'set': self.real_set_names, 'class': self.real_class_names}
        #
        # for p in self.database.propositions.itervalues():
        #     for label in p.f:
        #         vclass = p.f[label].vclass
        #         real_name_dict[vclass].add(label)
        # print real_name_dict

        self.constructor_dictionary = [{} for _ in range(self.max_axiom_arity)]

        # we need to define some extra variables, which we'll randomly assign when we read in a statement
        # this is a reasonable amount of data augmentation.
        self.extra_wffs     = language_model_extra_variables_of_each_type+max(len([f for f in p.f.values() if f.vclass=='wff']) for p in database.propositions.values() )
        self.extra_classes  = language_model_extra_variables_of_each_type+max(len([f for f in p.f.values() if f.vclass=='class']) for p in database.propositions.values() )
        self.extra_sets     = language_model_extra_variables_of_each_type+max(len([f for f in p.f.values() if f.vclass=='set']) for p in database.propositions.values() )

        # hand code these in.
        self.extra_sets = 20
        self.extra_wffs = 18
        self.extra_classes = 27

        self.wff_names = ['WFFVar'+str(i) for i in range(self.extra_wffs)]
        self.set_names = ['SetVar'+str(i) for i in range(self.extra_sets)]
        self.class_names = ['ClassVar'+str(i) for i in range(self.extra_classes)]

        self.num_extra_variable_names = len(self.wff_names)+len(self.set_names)+len(self.class_names)
        self.extra_variable_dict = {}

        # the names for the unconstrained variables
        #self.ua_names = ['UA'+str(i) for i in range(self.max_unconstrained_arity)]

        # add them to the dictionary
        arityzerodict = self.constructor_dictionary[0]
        for i in range(self.extra_wffs):
            arityzerodict['WFFVar'+str(i)]=len(arityzerodict)
            self.extra_variable_dict['WFFVar'+str(i)]=len(self.extra_variable_dict)
        for i in range(self.extra_classes):
            arityzerodict['ClassVar'+str(i)]=len(arityzerodict)
            self.extra_variable_dict['ClassVar'+str(i)]=len(self.extra_variable_dict)
        for i in range(self.extra_sets):
            arityzerodict['SetVar'+str(i)]=len(arityzerodict)
            self.extra_variable_dict['SetVar'+str(i)]=len(self.extra_variable_dict)
        # for i in range(len(self.ua_names)):
        #     arityzerodict['UA'+str(i)]=len(arityzerodict)
        #     self.extra_variable_dict['UA'+str(i)]=len(self.extra_variable_dict)

        # a block to create a dictionary that takes a symbol to its vclass
        self.symbol_to_vclass = {label:database.propositions[label].vclass for label in database.non_entails_axioms}
        for symbol in self.wff_names:
            self.symbol_to_vclass[symbol] = 'wff'
        for symbol in self.set_names:
            self.symbol_to_vclass[symbol] = 'set'
        for symbol in self.class_names:
            self.symbol_to_vclass[symbol] = 'class'

        # a list of all of the extra variables, for use later
        self.new_names = self.wff_names+self.set_names+self.class_names

        # describe the number of variables we've used
        print('wff variables:',self.extra_wffs)
        print('class variables:',self.extra_classes)
        print('set variables:',self.extra_sets)
        #print 'ua variables:', self.ua_names

        # now add the actual constructor axioms to our dictionary
        for p in database.non_entails_axioms.values():
            c_dict = self.constructor_dictionary[p.arity()]
            c_dict[p.label] = len(c_dict)

        for i in range(self.max_axiom_arity):
            print(len(self.constructor_dictionary[i]),'constructor axioms with arity',i)

        # build a pair of dictionaries that convert (arity,num) to total_num
        # and vice versa.  This is ugly.  Whatever
        self.arity_num_to_global_index = {}
        self.global_index_to_arity_num=[]
        global_index = 0
        for arity in range(self.max_axiom_arity):
            for num in range(len(self.constructor_dictionary[arity])):
                self.global_index_to_arity_num.append((arity,num))
                self.arity_num_to_global_index[(arity,num)]=global_index
                global_index+=1

        """sets up the data sets.  We divide the propositions into training/validation/test and
        then compile the corresponding list of statements"""
        list_of_propositions = self.database.propositions_list[:]  # database.propositions.values()
        np.random.seed(seed=121451345)
        list_of_propositions = np.random.permutation(list_of_propositions)

        num_validation = len(list_of_propositions)//10
        num_test = num_validation
        num_training = len(list_of_propositions)-num_test-num_validation
        self.training_propositions = list_of_propositions[:num_training]
        self.training_propositions = [_ for _ in self.training_propositions if _.type=='p']
        self.validation_propositions = list_of_propositions[num_training:num_training+num_validation]
        self.validation_propositions = [_ for _ in self.validation_propositions if _.type=='p']
        self.test_propositions = list_of_propositions[num_training+num_validation:]
        self.test_propositions = [_ for _ in self.test_propositions if _.type=='p']

        if self.database.remember_proof_steps:
            self.all_proof_steps = [] # except those that refer to e or f-type hypotheses
            for p in self.database.propositions.values():
                self.all_proof_steps += [step for step in p.entails_proof_steps if not (step.prop.type=='f' or step.prop.type == 'e')]


            self.training_proof_steps = []
            for p in self.training_propositions:
                self.training_proof_steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]

            self.validation_proof_steps = []
            for p in self.validation_propositions:
                self.validation_proof_steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]

            self.test_proof_steps = []
            for p in self.test_propositions:
                self.test_proof_steps += [step for step in p.entails_proof_steps
                        if not (step.prop.type=='f' or step.prop.type == 'e')]

            print()
            print('training steps:', len(self.training_proof_steps))
            print('validation steps:', len(self.validation_proof_steps))
            print('test steps:', len(self.test_proof_steps))


            # figure out how frequenly each proposition is used
            self.prop_usage = [0 for p in self.database.propositions]
            for s in self.all_proof_steps:
                self.prop_usage[s.prop.number]+=1

            # figure out what the most difficult proof step is
            self.max_depth = max([s.height for s in self.all_proof_steps]) + 1
            print('max proof step depth:', self.max_depth-1)


        # figure out the number of times each proposition is used.
        # self.prop_uses = [0.1] * len(self.database.propositions) # for numberical stability
        # for step in self.all_proof_steps:
        #     self.prop_uses[step.prop.number] += 1
        # self.initial_b = np.log(1.0*np.array(self.prop_uses)/sum(self.prop_uses))


        # build up a database of propositions by unconstrained arity
        # that is, total_unconstrained_arity is the total
        # of all of the unconstrained arities of all of the propositions.
        # and unconstrained_arity_indices is a list of p.unconstrained_arity()
        # unique indices for each proposition p.
        self.total_unconstrained_arity = 0
        self.unconstrained_arity_indices = {}
        self.unconstrained_label_to_number = {}
        for p in self.database.propositions_list:  # in order of proposition number
            u_arity = p.unconstrained_arity()
            self.unconstrained_arity_indices[p.label]=list(range(self.total_unconstrained_arity, self.total_unconstrained_arity + u_arity))
            self.total_unconstrained_arity += u_arity
            self.unconstrained_label_to_number[p.label]=len(self.unconstrained_label_to_number)
        #self.max_unconstrained_arity = max([p.unconstrained_arity() for p in self.database.propositions.itervalues()])

        self.total_constructor_arity = 0
        self.constructor_arity_indices = {}
        self.constructor_label_to_number = {}
        self.constructor_labels = []
        for p in database.non_entails_axioms.values():
            u_arity = p.arity()
            self.constructor_arity_indices[p.label]=list(range(self.total_constructor_arity, self.total_constructor_arity + u_arity))
            self.total_constructor_arity += u_arity
            self.constructor_label_to_number[p.label]=len(self.constructor_label_to_number)
            self.constructor_labels.append(p.label)
        for name in self.wff_names+self.set_names+self.class_names: #+self.ua_names:
            self.constructor_arity_indices[name] = []  # the extra arity 0 constructors
            self.constructor_label_to_number[name]=len(self.constructor_label_to_number)
            self.constructor_labels.append(name)

        # a lookup table for the index into all the propositions of the label
        self.label_to_number = {x.label:x.number for x in self.database.propositions.values()}
        for x in self.new_names:
            self.label_to_number[x] = -1  # all variables should always be included

    def training_set(self):  # DEFUNCT
        assert self.database.remember_proof_steps
        return np.random.permutation(self.training_proof_steps)

    def random_proof_step(self, source=None):
        assert self.database.remember_proof_steps
        if source is None:
            return random.choice(self.all_proof_steps)
        if source == "test":
            return random.choice(self.test_proof_steps)
        if source == "train":
            return random.choice(self.training_proof_steps)
        if source == "validation":
            return random.choice(self.validation_proof_steps)


    def axiom_counts(self):
        max_axiom_arity = max([p.arity() for p in self.database.non_entails_axioms.values()])+1
        # out = [0 for x in range(max_axiom_arity)]
        # for p in self.database.non_entails_axioms.itervalues():
        #     out[p.arity()]+=1
        out = [len(d) for d in self.constructor_dictionary]
        return max_axiom_arity, out

    # this lists all the free variables in context and generates a dictionary replacing them with my own variable types.
    def random_replacement_dict_f(self, f = None):
        if f is None: return self.random_permutation_dict()

        statement_wffs = [l for l in f if f[l].vclass=='wff']
        statement_classes = [l for l in f if f[l].vclass=='class']
        statement_sets = [l for l in f if f[l].vclass=='set']
        wff_random = np.random.choice(self.extra_wffs, len(statement_wffs), replace=False) if self.extra_wffs>0 else {}
        class_random = np.random.choice(self.extra_classes, len(statement_classes), replace=False) if self.extra_classes>0 else {}
        set_random = np.random.choice(self.extra_sets, len(statement_sets), replace=False) if self.extra_sets>0 else {}

        replacement_dict = {}
        for i in range(len(wff_random)):
            replacement_dict[statement_wffs[i]] = self.wff_names[wff_random[i]]
        for i in range(len(class_random)):
            replacement_dict[statement_classes[i]] = self.class_names[class_random[i]]
        for i in range(len(set_random)):
            replacement_dict[statement_sets[i]] = self.set_names[set_random[i]]

        return replacement_dict

    def deterministic_replacement_dict_f(self, f = None):
        assert f is not None

        statement_wffs = [l for l in f if f[l].vclass=='wff']
        statement_classes = [l for l in f if f[l].vclass=='class']
        statement_sets = [l for l in f if f[l].vclass=='set']
        # wff_random = np.random.choice(self.extra_wffs, len(statement_wffs), replace=False) if self.extra_wffs>0 else {}
        # class_random = np.random.choice(self.extra_classes, len(statement_classes), replace=False) if self.extra_classes>0 else {}
        # set_random = np.random.choice(self.extra_sets, len(statement_sets), replace=False) if self.extra_sets>0 else {}

        replacement_dict = {}
        for i in range(len(statement_wffs)):
            replacement_dict[statement_wffs[i]] = self.wff_names[i]
        for i in range(len(statement_classes)):
            replacement_dict[statement_classes[i]] = self.class_names[i]
        for i in range(len(statement_sets)):
            replacement_dict[statement_sets[i]] = self.set_names[i]

        return replacement_dict

    def random_permutation_dict(self):
        replacement_dict={}

        out_vars = self.wff_names[:]
        np.random.shuffle(out_vars)
        for i in range(len(out_vars)):
            replacement_dict[self.wff_names[i]] = out_vars[i]

        out_vars = self.set_names[:]
        np.random.shuffle(out_vars)
        for i in range(len(out_vars)):
            replacement_dict[self.set_names[i]] = out_vars[i]

        out_vars = self.class_names[:]
        np.random.shuffle(out_vars)
        for i in range(len(out_vars)):
            replacement_dict[self.class_names[i]] = out_vars[i]
        return replacement_dict

    def random_replacement_dict(self, context):
        statement_wffs = [l for l in context.f if context.f[l].vclass=='wff']
        statement_classes = [l for l in context.f if context.f[l].vclass=='class']
        statement_sets = [l for l in context.f if context.f[l].vclass=='set']
        wff_random = np.random.choice(self.extra_wffs, len(statement_wffs), replace=False)
        class_random = np.random.choice(self.extra_classes, len(statement_classes), replace=False)
        set_random = np.random.choice(self.extra_sets, len(statement_sets), replace=False)

        replacement_dict = {}
        for i in range(len(wff_random)):
            replacement_dict[statement_wffs[i]] = self.wff_names[wff_random[i]]
        for i in range(len(class_random)):
            replacement_dict[statement_classes[i]] = self.class_names[class_random[i]]
        for i in range(len(set_random)):
            replacement_dict[statement_sets[i]] = self.set_names[set_random[i]]

        return replacement_dict

    def standardize_context(self, prop, tree=None):
        '''
        takes in a proposition and return it in a standardized form,
        in particular, keeping the hypotheses, variables, d statements,
        and tree.

        prop: a proposition
        tree: an optional tree that will replace the prop's tree

        this retuns an object with the following properties:
        number, tree, hyps, d_labels

        This uses a fixed replacement dictionary because it will make it
        easier to interpret on consequative runs.
        '''

        # use the default tree
        if tree is None: tree = prop.tree

        context = Container()
        context.label = prop.label

        context.number = prop.number
        replacement_dict = self.deterministic_replacement_dict_f(f=prop.f)
        mandatory = [h.label for h in prop.hyps if h.type == 'f']

        # might as well keep the tree
        context.tree = tree.copy().replace_values(replacement_dict)

        # I think that this is everything we need
        context.hyps = []
        for h in prop.hyps:
            newh = Container()
            newh.type = h.type
            if h.type == 'e':
                newh.tree = h.tree.copy().replace_values(replacement_dict)
                newh.label = h.label
            else:
                newh.label = replacement_dict[h.label]
                newh.old_label = h.label
            context.hyps.append(newh)

        # fuck d, we'll just consider d_labels
        replaced_mandatory = [replacement_dict[label] for label in mandatory]
        context.mandatory = mandatory
        context.d_labels = set()
        for (i,j) in prop.d_labels:
            if i in mandatory and j in mandatory:
                context.d_labels.add((replacement_dict[i], replacement_dict[j]))
        for i in self.new_names:
            for j in self.new_names:
                if i == j:
                    continue
                if i not in replaced_mandatory or j not in replaced_mandatory:
                    context.d_labels.add((i,j))

        # the list (err... dictionary) of variables
        context.f = {k:Container() for k in self.new_names}
        for k in self.new_names:
            context.f[k].statement = [k]

        # list all the variables that appear in the hypotheses
        context.hyp_symbols = set()
        for h in context.hyps:
            if h.type=='e':
                context.hyp_symbols |= set(h.tree.list())
        context.hyp_symbols &= set(self.new_names)
        context.main_but_not_hyp_symbols = set(replaced_mandatory).difference(context.hyp_symbols)

        # the hyp symbols are the ones we're allowed to use.
        # the main_but_not_hyp_symbols symbols are explicitly excluded

        # I'm not sure that we should actually include this, but meh
        context.replacement_dict = replacement_dict
        return context

    def simple_apply_prop(self, tree, prop, context, vclass=None):
        # assert prop.unconstrained_arity() == 0
        fit = prop_applies_to_statement(tree, prop, context, vclass=vclass)
        assert fit is not None
        return [h.tree.copy().replace(fit) for h in prop.hyps if h.type=='e']

    '''
    Iterated fit for the tree and hyps.
    This doesn't check for disjointness, just matches the trees
    '''
    def reconstruct_fit(self, tree, hyps, prop_label):
        prop = self.database.propositions[prop_label]
        prop_variables = [f for f in prop.f]

        current = tree.fit(prop.tree, prop_variables)

        for h, ph in zip(hyps, [h.tree for h in prop.hyps if h.type=='e']):
            next_fit = h.fit(ph, prop_variables)
            if dictionary_merge(current, next_fit) is None: return None

        return current

    def prop_applies_to_statement(self, tree, prop, context, vclass=None):
        return prop_applies_to_statement(tree, prop, context, vclass=vclass)


def dictionary_merge(current, new):
    # merges new into current
    for x in new:
        if x in current:
            if current[x] != new[x]:
                return None
        else:
            current[x] = new[x]
    return current

"""
Now we use generate some utilities that we'll use for fitting propositions and
axioms to statements, or just which ones apply in general.
"""

# attempts to fit a proposition to a statement.  If it does fit, it returns the fit.
def prop_applies_to_statement(tree, prop, context, vclass=None):
    # verify first that the vclass is okay.
    if vclass is not None and prop.vclass != vclass: return None

    #context_variables = set(f.label for f in context.f.itervalues())
    #prop_variables = set(f.label for f in prop.f.itervalues())
    prop_variables = [f for f in prop.f]

    #print prop_variables
    # attempt to fit to the tree
    fit = tree.fit(prop.tree,prop_variables)

    if fit is None: return None # it doesn't work. Sadness.

    # we only need the mandatory variables
    context_variables = [x.label for x in context.hyps if x.type=='f']
    prop_variables = [x.label for x in prop.hyps if x.type=='f']
    vars_that_appear = {v:fit[v].set().intersection(context_variables) for v in fit}

    # print 'context_variables', context_variables
    # print 'prop_variables', prop_variables
    # print 'vars_that_appear', vars_that_appear

    for (xvar,yvar) in prop.d_labels:
        if xvar not in fit or yvar not in fit: continue
        for i in vars_that_appear[xvar]:
            for j in vars_that_appear[yvar]:
                if (i,j) not in context.d_labels:
                    return None


    return fit




# How the fuck do we do this search?
# I'm just going to insert an ugly hack here, so that I have something at least.
# Given that the brute force approach isn't terrible, this should be good, right?

"""
This is going to be my algorithm:  It's a lame algorithm but it should be good enough
We store some number of objects, which are a list of pairs of the following form:
(location in tree, dictionary)
dictionary takes a proposition or "*" and returns one of two things:
1. A list of valid labels of matching propositions (if the number is small or if we're out of nodes)
2. Another pair of the above form.

To search for valid labels, we start at the first location, and look up two values in the
dictionary: "*" and whatever the label at that position is. If it's another location pair,
follow it and then union everything.

To build up the tree, we start with a node and add things to it.  If the number of items in
the list is sufficiently large (greater than 100 or so), we expand that node.  To do so,
we take the next location that must exist in the corresponding tree in a breadth-first manner
and build a new dictionary node thingy based off of that.

This actually works very well with the expand_threshold of somewhere between 10 and 100.
That is, it has 10 percent of the time of subsequent verification as inefficency.
"""

"""
# the following code was used to test it in an ipython notebook
searchproblem = SearchProblem(database)


p = random.choice(database.propositions.values())
while len(p.entails_proof_steps)==0: p = random.choice(database.propositions.values())
statement = random.choice(p.entails_proof_steps)
print statement.tree.stringify()
print statement.prop.label, statement.prop.tree.stringify()
print prop_applies_to_statement(statement.tree, statement.prop, p)
print
def test_all_props(statement,proplist):
    labels = set()
    for prop in proplist:
        if not prop_applies_to_statement(statement.tree, prop, p)==None:
            labels.add(prop.label)
            #print prop.label, prop.tree.stringify(), prop_applies_to_statement(statement.tree, prop, p)
    return labels

labels = test_all_props(statement,database.propositions.itervalues())
%timeit searchproblem.search(statement.tree,p)
%timeit test_all_props(statement,database.propositions.itervalues())
%timeit test_all_props(statement,[database.propositions[l] for l in labels])
#print labels
print getsizeof(labels)
labels2 = searchproblem.search(statement.tree,p)
#print labels2
print list(labels).sort()==labels2.sort()
print statement.prop.label in labels
"""

class SearchProblem:
    def __init__(self,database, max_unconstrained_arity = None):
        self.start = ((),{}) # root node, empty dictionary
        self.wildcard = "VAR" #the placeholder for the wildcard value
        self.expand_threshold = 10;
        self.database = database

        for p in database.propositions.values():
            # if the unconstrained_arity is too hight, skip it
            # if not max_unconstrained_arity is None and p.unconstrained_arity()>max_unconstrained_arity: continue
            self.add(p)

    def add(self,p):
        current = self.start
        observed_positions = []

        while True:
            position, dictionary = current
            observed_positions.append(position)
            value = p.tree.value_at_position(position)
            if value in p.f: value = self.wildcard #variables count as wildcards

            # if the value isn't in the dictionary, add it
            if value not in dictionary:
                dictionary[value] = [p.label]
                return

            next_pair = dictionary[value]
            if type(next_pair) is list:
                next_pair.append(p.label)
                if len(next_pair) > self.expand_threshold:
                    self.expand(current,observed_positions, value,p)
                return

            # otherwise continue to track along the search nodes until you find it.
            current = next_pair

    # I use the fact that the degree of any wildcard node is 0
    def expand(self,current, observed_positions, value,p):
        # expand the node at current[1][value]
        all_positions = p.tree.breadth_first_position_list()

        next_position = None
        for pos in all_positions:
            if pos not in observed_positions:
                next_position = pos
                break
        if next_position == None: #we've exhausted the entire tree
            return

        labels = current[1][value]

        current[1][value] = (next_position,{})

        for l in labels:
            self.add(self.database.propositions[l])

    # Performs the search in the tree.  I'll use this as a untility function
    # for dealing with the actual matching: it's a pretty good filter, I hope
    def tree_match(self, tree, current = None):
        if current == None: current = self.start

        # if we ended up at a list, return everything
        if type(current) is list: return current

        position, dictionary = current
        value = tree.value_at_position(position)
        out = []
        if value in dictionary: out+= self.tree_match(tree,current=dictionary[value] )
        if self.wildcard in dictionary: out+= self.tree_match(tree,current=dictionary[self.wildcard])
        return out

    # given a statement, this will return all of the consistant propositions.
    # if max_proposition (=context.number) is set, the search
    # will only return propositions numbered *less* than that.
    def search(self,tree,context, max_proposition = None, vclass=None):
        restricted_labels = self.tree_match(tree)
        out =  [l for l in restricted_labels if not prop_applies_to_statement(tree, self.database.propositions[l], context, vclass=vclass) is None]
        if not max_proposition == None:
            out = [l for l in out if self.database.propositions[l].number<max_proposition]
        return out

    def search_dictionary(self,tree,context, max_proposition = None, vclass=None):
        restricted_labels = self.tree_match(tree)
        out =  {}
        for l in restricted_labels:
            if max_proposition is not None and self.database.propositions[l].number>=max_proposition:
                continue
            fit = prop_applies_to_statement(tree, self.database.propositions[l], context, vclass=vclass)
            if fit is not None:
                out[l]=fit
        return out
