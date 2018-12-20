'''
This is a basic paser for the metamath language.

NOTE: THIS IS NOT A PROOF VERIFIER.  IT CATCHES MOST PROBLEMS WITH METAMATH
PROOFS, BUT MAY NOT CATCH ALL OF THEM.
'''


import sys
import itertools
import collections
import os.path
import copy
from tree import *
from statement_to_tree import *

import json  # for debugging

# The path to set.mm
initial_file_name = "set.mm"


def concatonate_to_string(list):
    if len(list) == 0: return ''
    string = list[0]
    for item in list[1:]:
        string += ' '+item
    return string


# splits a list that looks like ['(', 'thing1', 'thing2', ')','proofpart1','proofpart2']
def split_compressed_proof(list):
    assert list[0] == '('
    index = 1
    while list[index] != ')': index+=1
    labels = list[1:index]  # not including index
    proof = ''
    for part in list[index+1:]: proof+=part
    return labels, proof


class file_contents:
    def __init__(self):
        self.included_files = set()
        self.tokens = []
        self.input_file(initial_file_name)
        self.current_index = 0

    # read the contents of filename, splits into space-separated list,
    # and appends to self.tokens.
    def input_file(self, filename):
        full_path = os.path.realpath(filename)
        if full_path in self.included_files:
            return
        f = open(full_path, 'r')
        f_contents = f.read().split()
        f.close()
        print("included", len(f_contents), "tokens from", filename)
        f_contents.reverse()
        self.tokens += f_contents
        self.included_files.add(full_path)

    def read_token(self):
        if len(self.tokens) == 0: return None
        return self.tokens.pop()

    def read_until(self,end_token):
        out = []
        in_comment = False
        while True:
            next_token = self.read_token()
            assert next_token != None
            if next_token==end_token: return out
            if next_token=='$(': in_comment=True
            if not in_comment: out.append(next_token)
            if next_token=='$)': in_comment=False

class f_hypothesis:
    def __init__(self,label,statement):
        self.label = label
        self.type = "f"
        self.vclass = statement[0] # the class of the statement
        self.statement = [statement[1]]
        self.variable = statement[1]
        self.tree = Tree(value=self.label)
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return '<f_hypothesis '+self.label+': '+self.vclass + ' '+''.join(self.statement)+'>'

class e_hypothesis:
    def __init__(self, label, statement, tree):
        self.label = label
        self.string = statement
        self.type = "e"
        self.tree = tree
        self.statement=statement
        self.vclass = statement[0] # the class of the statement --- here must be entails
        assert self.vclass == '|-'
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return '<e_hypothesis '+self.label+': '+self.vclass + ' '+''.join(self.statement)+'>'

class block:
    def __init__(self):
        self.c = set() # the constants.  In theory these apply to the entire database
        self.v = set()
        self.d = set() # a set of tuples with distinct variables
        self.f = {} # a dictionary mapping labels to the corresponding f_hypothesis, which define a variable type
        self.e = {} # a dictionary mapping labels to the corresponding e_hypothesis
        self.hyps = [] # a list of hypotheses, which are f_hypotheses or e_hypotheses

# Converts a tree to a string.  Context is the proposition that this is happening in.
# this doesn't include the class that the thing is in.   Meh.
def tree_to_string(tree, database, context):
    if tree.value in context.f: return context.f[tree.value].statement
    prop = database.propositions[tree.value]
    assert len(tree.leaves)==len(prop.hyps)
    replacement_dict = {prop.hyps[i].variable:tree_to_string(tree.leaves[i], database, context) for i in range(len(prop.hyps)) if prop.hyps[i].type == "f"} #find the replacement rules for the variables
    #print 'replacement_dict', replacement_dict, 'applied to', prop.statement
    return string_replace(prop.statement[1:],replacement_dict)

def string_replace(list,dic):
    out = []
    for l in list:
        if l in dic:
            out += dic[l]
        else:
            out += [l]
    return out


# I assume that these are all entails statements.
class proof_step:
    def __init__(self, tree, context, prop, prior_statements):
        # fundamental properites
        self.tree = tree  # the parse tree of the statement
        self.context = context  # the proposition that this is being used to prove
        self.vclass = prop.vclass  # wff, class, set, or |-, since this isn't remembered by the tree.

        # properties for generative nn
        self.prop = prop  # the proposition/axiom that was used to derive this step from the previous one(s)
        self.unconstrained = []  # is the trees of the statements corresponding to the unconstrained_f of self.prop

        if len(prior_statements) > 0:  # don't consider this for the f and e type statements
            for i in prop.unconstrained_indices:
                self.unconstrained.append(prior_statements[i].tree) # only keep the tree

        # properties for difficulty predictor network
        prior_entails = [p for p in prior_statements if p.vclass == '|-']
        if len(prior_entails) == 0:
            self.height = 0  # the depth of the proof tree of the statement
        else:
            self.height = 1 + max([p.height for p in prior_entails])
        self.descendants = 1 + sum([p.height for p in prior_entails])  # the number of descendants in the proof tree, including this node

        self.applicable_propositions = None

    def summarize(self):
        print("proof step summary:")
        print("  statement", self.tree.stringify())
        print("  involved in proof of", self.context.label)
        print("  uses", self.prop.label)
        print("    with new steps",[tree.stringify() for tree in self.unconstrained])
        print("  height", self.height, "descendants",self.descendants)

class proposition:
    # this is more-or-less the same as a block, but with some additional information
    # statement is a string
    def __init__(self,a_or_p,label,block,statement, proof, number=None, tree=None):
        self.type = a_or_p
        self.statement = statement

        self.number = number  # the count in the list of propositions

        self.label = label
        self.vclass = statement[0]

        self.proof = proof  # the expanded proof string

        self.c = block.c.copy()  # the constants.  In theory these apply to the entire database
        self.v = block.v.copy()
        self.d = block.d.copy()

        # reduce the list of f and e statements
        self.e = block.e.copy()  # all active e-statements are mandatory

        # include only the mandatory f statements.  We'll extend this later after we decompress the proof.
        self.f = {label:block.f[label] for label in block.f if block.f[label].variable in statement or any(block.f[label].variable in e.statement for e in block.e.values())}

        # all the mandatory hypotheses
        self.hyps = [hyp for hyp in block.hyps if hyp.type == "e" or hyp.label in self.f]
        self.tree = tree

        # rewrite d, using the labels instead of variable names
        replacement_dict = {self.f[label].variable:label for label in self.f}
        self.d_labels = set()
        for a, b in self.d:
            if a not in replacement_dict or b not in replacement_dict:
                continue
            self.d_labels.add((replacement_dict[a], replacement_dict[b]))
            self.d_labels.add((replacement_dict[b], replacement_dict[a]))

        # self.unconstrained_f gives the list of labels of *mandatory* f-statements if the corresponding variable
        # does not appear in the statement.  This is notable because it's the part that we are going to
        # need to predict in the automated prover.
        self.unconstrained_f = set(label for label in self.f if self.f[label].variable not in statement)
        # the indices of the hypotheses that we'll need to keep track of
        self.unconstrained_indices = [i for i in range(len(self.hyps)) if self.hyps[i].type=='f' and self.hyps[i].label in self.unconstrained_f]
        self.unconstrained_variables = [self.hyps[x].label for x in self.unconstrained_indices]
        self.entails_proof_steps = []

        # the list of numbers of applicable propositions
        # we don't do anything with this now, but we will when we
        # load them into the datum
        self.applicable_propositions = None

    def print_details(self):
        print()
        print('('+self.type+')',self.label+': arity',self.arity())
        if len(self.d)>0: print( ' ',self.d)
        print( 'f:',self.f)
        for hyp in self.hyps:
            print( '  '+hyp.label+':',concatonate_to_string(hyp.statement))
        print('statement:',concatonate_to_string(self.statement))

    # the arity of the statement
    def arity(self):
        return len(self.hyps)

    def unconstrained_arity(self):
        return len(self.unconstrained_indices)

    # unify takes a list of trees which are correspond to self.hyps
    # f-hypotheses need a thing of the correct class (but I'm too lazy to actually check)
    # e-statements need an entails
    # this is an exercise in only passing trees
    # variables is a list of variables for the
    # context is the proposition that this is happening in
    def unify(self,trees,context):
        assert len(trees) == len(self.hyps)
        replacement_dict = {self.hyps[i].label:trees[i] for i in range(len(trees)) if self.hyps[i].type == "f"}
        # print 'proposition', self.statement, self.tree.stringify()
        # print 'starting trees', [tree.stringify() for tree in trees]
        # print 'unified to',self.tree.copy().replace(replacement_dict).stringify()
        # build the replacement dictionary out of the trees

        # verify that we actually have proofs of all the e-hypotheses
        for i in range(len(trees)):
            if self.hyps[i].type == "e":
                #print '  start with' ,self.hyps[i].tree.stringify()
                test_tree = self.hyps[i].tree.copy().replace(replacement_dict)
                #print '  replacements:',{label:replacement_dict[label].stringify() for label in replacement_dict}
                #print '  e-hyps', test_tree.stringify(), 'should be', trees[i].stringify()
                assert test_tree == trees[i]

        # verify that all the d statements are satisfied

        # that's all wrong.  Very wrong.  What we actually want to do is
        # verify that for every pair (v1,v2) in self.d, for every pair of variables x in v1, y in v2
        # we have (x,y) in context.d
        context_variables = set(f.variable for f in context.f.values())
        var_sets =  {self.hyps[i].variable:trees[i].set().intersection(context_variables) for i in range(len(trees)) if self.hyps[i].type == "f"}
        for v1,v2 in self.d:
            if v1 not in var_sets or v2 not in var_sets: continue # one of these is an optional variable, used only in the proof
            for x1 in var_sets[v1]:
                for x2 in var_sets[v2]:
                    assert (x1,x2) in context.d or (x2,x1) in context.d


        # Yay!  These are valid contributions
        # now we can return the parse tree for the actual statement with
        # the substitutions added in.
        return self.tree.copy().replace(replacement_dict)

    # Reads though the proof and finds all the optionals that are referenced and includes them
    # in the f statements for the proposition
    def update_optional_hypotheses(self,block):
        for f in block.f.values():
            if (not f.label in self.f) and (f.label in self.proof):
                self.f[f.label] = f
        self.update_d()

    def update_d(self):
        included_vars = set(f.variable for f in self.f.values())
        self.d = set(x for x in self.d if x[0] in included_vars and x[1] in included_vars)

class meta_math_database:  # the database is just a collection of blocks
    def __init__(self, file_contents, n=None, remember_proof_steps=True):
        self.remember_proof_steps = remember_proof_steps
        self.block_stack = [block()]
        self.propositions = {}
        self.propositions_list = []  # a list of all the propositions in numerical order
        self.label = None
        self.non_entails_axioms = {}
        self.parser = StatementParser(self)
        self.read_file_contents(file_contents, n=n)


    def read_file_contents(self, file_contents, n=None):
        # read in the database
        while n==None or len(self.propositions) < n:
            next_token = file_contents.read_token()
            current_block = self.block_stack[-1]

            if next_token == None: return
            elif next_token == '$(':
                file_contents.read_until('$)')
            elif next_token == '${':
                # start a new block.  I assume all sub-blocks are started after
                # all the inherited statements
                self.block_stack.append(copy.deepcopy(self.block_stack[-1]))
            elif next_token == '$}':
                # end the block: remove it from the block stack.
                assert len(self.block_stack)>0
                self.block_stack.pop()

            # new constants
            elif next_token == '$c':
                new_tokens = file_contents.read_until('$.')
                for token in new_tokens:
                    if token in current_block.c: raise Exception('token already defined in scope')
                    if token in current_block.v: raise Exception('token already defined in scope')
                    current_block.c.add(token)
                assert self.label == None #check for stray tokens

            # new distinct variables.  Add them pairwise to the current block
            elif next_token == '$d':
                new_tokens = file_contents.read_until('$.')
                for i in range(len(new_tokens)):
                    for j in range(i+1,len(new_tokens)):
                        current_block.d.add((new_tokens[i],new_tokens[j]))
                assert self.label == None #check for stray tokens

            # new variables
            elif next_token == '$v':
                new_tokens = file_contents.read_until('$.')
                for token in new_tokens:
                    if token in current_block.c: raise Exception('token already defined in scope')
                    if token in current_block.v: raise Exception('token already defined in scope')
                    current_block.v.add(token)
                assert self.label == None #check for stray tokens

            # class specifications for variables
            elif next_token == '$f':
                #constant = file_contents.read_token()
                variables = file_contents.read_until("$.")
                assert len(variables)==2 # [constant, variable]
                new_hypothesis=f_hypothesis(self.label,variables)
                current_block.hyps.append(new_hypothesis)
                current_block.f[self.label]=new_hypothesis
                self.label = None

            # assumptions for the theorems in the block
            elif next_token == '$e':
                #constant = file_contents.read_token()
                math_symbols = file_contents.read_until("$.")

                # deal with the parse tree
                _, tree = self.parser.parse_new_statement(["wff"]+math_symbols[1:],current_block)
                if not tree:
                    # Fuck: the parser failed
                    raise Exception('Parsing of', math_symbols,'failed')
                #replacement_dict={"VAR"+f.variable:f.label for f in current_block.f.itervalues()}
                #tree.replace_values(replacement_dict)


                new_hypothesis=e_hypothesis(self.label,math_symbols,tree)
                current_block.hyps.append(new_hypothesis)
                current_block.e[self.label]=new_hypothesis
                self.label = None

            # an axiom --- I don't check uniqueness of labels like I should.
            elif next_token == '$a':
                #constant = file_contents.read_token()
                statement = file_contents.read_until("$.")
                if statement[0]=="|-":
                    _, tree = self.parser.parse_new_statement(["wff"]+statement[1:],current_block)
                    #replacement_dict={"VAR"+f.variable:f.label for f in current_block.f.itervalues()}
                    #tree.replace_values(replacement_dict)
                    #print '\nparsed axiom', self.label, statement, tree.stringify()
                    #assert ["wff"]+statement[1:] == tree_to_string(tree,self,current_block)
                else:
                    tree = Tree(value=self.label,leaves=[f.tree for f in current_block.hyps if f.variable in statement])
                    #print '\nunparsed axiom', self.label, statement, tree.stringify()
                prop = proposition('a',self.label,current_block,statement,None,number=len(self.propositions), tree=tree)

                if statement[0]!="|-": self.non_entails_axioms[prop.label]=prop

                self.propositions[self.label]=prop
                self.propositions_list.append(prop)
                if len(self.propositions) % 10 ==0:#print len(self.propositions)#print prop.label #prop.print_details()
                    sys.stdout.write('\rproposition: '+str(len(self.propositions)))
                    sys.stdout.flush()
                self.label = None


            # a provable assertion
            elif next_token == '$p':
                if self.remember_proof_steps:
                    #constant = file_contents.read_token()
                    statement = file_contents.read_until("$=")
                    proof = file_contents.read_until("$.")
                    prop = proposition('p',self.label,current_block,statement,proof,number=len(self.propositions),)
                    self.propositions[self.label]=prop
                    self.propositions_list.append(prop)
                    self.uncompress(prop,current_block)  # uncompress the proof
                    prop.update_optional_hypotheses(current_block)
                    if len(self.propositions) % 10 ==0:#print len(self.propositions)#print prop.label #prop.print_details()
                        sys.stdout.write('\rproposition: '+str(len(self.propositions)))
                        sys.stdout.flush()
                    self.label = None


                    #if statement[0]!="|-":
                    #    self.non_entails_axioms[prop.label]=prop # these are all provable from non $p statements, so...
                    # try to verify the proposition
                    self.verify(prop)

                # don't remember steps.  don't verify.
                else:
                    #constant = file_contents.read_token()
                    statement = file_contents.read_until("$.")
                    if statement[0]=="|-":
                        _, tree = self.parser.parse_new_statement(["wff"]+statement[1:],current_block)
                    else:
                        tree = Tree(value=self.label,leaves=[f.tree for f in current_block.hyps if f.variable in statement])
                        #print '\nunparsed axiom', self.label, statement, tree.stringify()
                    prop = proposition('p',self.label,current_block,statement,None,number=len(self.propositions), tree=tree)
                    self.propositions[self.label]=prop
                    self.propositions_list.append(prop)
                    if len(self.propositions) % 10 ==0:#print len(self.propositions)#print prop.label #prop.print_details()
                        sys.stdout.write('\rproposition: '+str(len(self.propositions)))
                        sys.stdout.flush()
                    self.label = None

            elif next_token == '$=':
                raise NotImplementedError

            elif next_token == '$[': #untested
                filename = concatonate_to_string(file_contents.read_until('$]'))
                file_contents.input_file(filename)

            else: # it's part of a label
                if self.label == None:
                    self.label = next_token
                else:
                    self.label += ' '+next_token


    def uncompress(self, prop,context):
        #print 'uncompressing'
        #print prop.proof[-1], concatonate_to_string(prop.statement)
        #for hyp in prop.block.hypotheses:
        #    print hyp.label+':',hyp.statement

        compressed_proof=prop.proof
        if compressed_proof[0]!='(': return # don't do anything if not compressed

        label_part, proof = split_compressed_proof(compressed_proof)

        # construct a list of all of the valid labels and their arities
        # we'll need this when expanding Z statements in the compressed proof.
        arities = {}
        hypothesis_labels = [hyp.label for hyp in prop.hyps]
        for label in hypothesis_labels:
            arities[label]=0

        token_list = hypothesis_labels+label_part
        for label in label_part:
            if label in self.propositions:
                arities[label] = self.propositions[label].arity()
            elif label in context.f:
                arities[label] = 0 # hypotheses have 0 arity
            else:
                print(prop.print_details())
                print(token_list)
                print (proof)
                print (label)
                raise Exception('not defined')
        zpositions = [];

        proof = proof.upper()
        out = [];
        index = -1
        current = 0
        for char in proof:
            index+=1 # not needed.  kept for reference
            num = ord(char)
            if num==90: # Z
                zpositions.append(len(out)-1) #save the last one as a z-position
            elif (85<=num<=89): # Ord('U')=85, Ord('Y')=89
                current *= 5
                current += num-84
            else:  # Ord('A')=65
                current = current *20 + num-64

                # append to the out list the corresponding stuff
                if current<=len(token_list):
                    out.append(token_list[current-1])
                elif current - len(token_list)<=len(zpositions):
                    # pull a bunch of stuff from the previous position
                    source_position = zpositions[current - len(token_list)-1]
                    to_append = [] # the new stuff to append, written backwards
                    remaining_arity = 1;
                    while remaining_arity>0:
                        if source_position<0:
                            print (token_list,zpositions,compressed_proof)
                            print (to_append,source_position)
                            print (remaining_arity)
                            prop.print_details()
                            raise Exception('reached start of proof before finishing Z substitution')
                        new_token = out[source_position]
                        remaining_arity += -1 + arities[new_token]
                        to_append.append(new_token)
                        source_position-=1
                    to_append.reverse()
                    out += to_append
                else:
                    print (len(token_list),len(zpositions), 'current=',current)
                    print( token_list,proof[index-5:index+1])
                    raise Exception('Attemped to refer to non-existant token position')

                # now reset the counter
                current = 0
        prop.proof = out # replace the compressed proof with the uncompressed one

    def verify(self,prop):
        # attempt to verify a proof
        statement = prop.statement
        proof = prop.proof
        #print
        #print 'proof:',proof

        # print some extra stuff for the theorem
        #for hyp in prop.hyps:
        #    print 'hypothesis', hyp.label, concatonate_to_string(hyp.statement), hyp.tree.stringify()
        #print prop.label

        proof_steps_stack = [] # this is equivalent to the statement stack, more or less,
        statement_stack = []
        entails_proof_steps = []

        index = 0 # the index into the proof steps.
        while index<len(proof):
            #print 'statement stack:', [t.stringify() for t in statement_stack]
            label = proof[index]

            #was this a hypothesis: arity 0
            if label in prop.e:
                index+=1
                statement_stack.append(prop.e[label].tree)
                proof_steps_stack.append(proof_step(prop.e[label].tree, prop, prop.e[label], []))
                entails_proof_steps.append(proof_steps_stack[-1]) # all e-type statements are entails, I think
            elif label in prop.f:
                index+=1
                statement_stack.append(prop.f[label].tree)
                proof_steps_stack.append(proof_step(prop.f[label].tree, prop, prop.f[label], []))

            # this is a axiom or proposition
            else:
                axiom = self.propositions[label] # or proposition
                arity = axiom.arity()
                if arity>0:
                    new_statement = axiom.unify(statement_stack[-arity:],prop)
                    statement_stack = statement_stack[0:-arity]

                    new_proof_step = proof_step(new_statement, prop, axiom, proof_steps_stack[-arity:])
                    proof_steps_stack = proof_steps_stack[0:-arity]

                else: # deal correctly with the arity 0 propositions
                    new_statement = axiom.tree
                    new_proof_step = proof_step(axiom.tree, prop, axiom, [])

                statement_stack.append(new_statement)
                proof_steps_stack.append(new_proof_step)
                if axiom.vclass == '|-':
                    entails_proof_steps.append(new_proof_step)

                index+=1
                #print 'current stack:',statement_stack

        assert len(statement_stack)==1
        #print 'statement:', prop.statement
        #print 'final proof result:', concatonate_to_string(statement_stack[0])
        prop.tree = statement_stack[0]

        # This is questionable.  I'm not really sure why it was in here in the first place, or how the
        # other systems deal with it.
        #prop.entails_proof_steps = entails_proof_steps
        prop.entails_proof_steps = [t for t in entails_proof_steps if t.prop.type != 'e']
        prop.num_entails_proof_steps = len(prop.entails_proof_steps)

        assert statement==[prop.statement[0]]+tree_to_string(prop.tree,self,prop) # The proof returns the desired statement

if __name__ == '__main__':
    text = file_contents()
