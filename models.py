'''
This is an overarching class for the config class, the variable class, and
the model class.

It should implement a bunch of the the normal stuff that I want from all of
the models, which will allow me to transfer stuff between them nicely.

I'm going to implement a couple of my new paradigms, which should make things
smoother, especially the saving.
'''

import nnlibrary as nn
import pickle as pickle
import numpy as np

class ConfigParameters:
    def __init__(self):
        '''This is the set of parameters that will be saved by the
        save command, incase we want to pull them up later.'''

        # training parameters
        self.lr = 1.0e-4        # the learning rate
        self.lr_reduction = 1.5 # the reduction of the leanring rate after epochs
        self.max_epochs = None  # the maximum number of training epochs
        self.stop_after = None  # the number of epochs to train after the best epoch
        self.epsilon = 1.0e-8   # the Adam epsilon parameter

        # hidden dimension sizes
        self.r = 7

        # number of gru layers
        self.gru_depth = 1
        self.bidirectional = True

        # for the graph augmentation
        self.augmentation = False
        self.structure_data=False
        self.structure_data_scaling=0.1


        # regularization
        self.regularization = None
        self.dropout = None      # dropout during training

        # extra symbols
        self.special_symbols = [
                'END_OF_HYP', 'END_OF_SECTION', 'START_OUTPUT',
                'TARGET_UC', 'UC']

        # add various parameters for attention.  These usually won't be used
        self.attention = False
        self.matrix_attention = True
        self.full_state_attention = False

class DefaultConfig:
    def __init__(self, language_model):
        ''' this is a default configuration thing, which should have
        most of the normal parameters.'''

        # add the normal parameters
        self.p = ConfigParameters()

        # add the language_model dependent stuff
        self.lm = language_model

        # and the various lookup tables
        self.total_constructor_arity = language_model.total_constructor_arity
        # self.constructor_arity_indices = language_model.constructor_arity_indices
        self.label_to_arity = {label:len(language_model.constructor_arity_indices[label]) for label in language_model.constructor_arity_indices}

        # self.num_constructors = len(self.constructor_arity_indices)  # the number of propositions in the database
        self.constructor_label_to_number = language_model.constructor_label_to_number
        self.num_extra_variable_names = language_model.num_extra_variable_names
        # self.max_unconstrained_arity = language_model.max_unconstrained_arity

        # this builds a list of all the constructors
        self.all_constructor_list = [None] * len(self.constructor_label_to_number)
        for label in self.constructor_label_to_number:
            number = self.constructor_label_to_number[label]
            assert self.all_constructor_list[number] is None
            self.all_constructor_list[number] = label
        self.construct_dictionary()



    def construct_dictionary(self):
        # decode turns numbers into tokens
        # encode turns tokens into numbers
        self.decode = self.all_constructor_list
        self.decode = self.decode + self.p.special_symbols
        self.encode = {}
        for i in range(len(self.decode)):
            self.encode[self.decode[i]] = i

        self.num_tokens = len(self.decode)
        print('Config(): added '+str(self.num_tokens)+' tokens to dictionary')
        # print self.num_tokens, self.decode

    def save(self, file_path):
        ''' saves the variables to file_path '''
        with open(file_path, 'wb') as handle:
            pickle.dump(self.p, handle)

    def load(self, file_path):
        ''' loads the variables and replaces the current values '''
        with open(file_path, 'rb') as handle:
            self.p = pickle.load(handle)

class AugmentationVariables:
    def __init__(self, graph, r, name, bidirectional=False):
        ''' this defines a set of variables for the
        augmentation portion of the graph. If graph
        is not None, these variables get added to
        graph.  r is the dimension of the various
        parameters.  This includes variables for
        both the backward and forward passes.

        This creates the following variables:
        no_parent, no_left_sibling, no_right_sibling

        In theory I could add an extra set of the above
        for each of the special symbols.  I won't now,
        though, although I may chnage that later.
        '''
        self.no_parent = nn.VariableNode([r], None, name=name+'no_parent')
        self.no_left_sibling = nn.VariableNode([r], None, name=name+'no_left_sibling')
        self.vs = [self.no_parent, self.no_left_sibling]

        if bidirectional:
            self.no_right_sibling = nn.VariableNode([r], None, name=name+'no_right_sibling')
            self.vs.append(self.no_right_sibling)

        self.rvs = []
        return

class Container:
    def __init__(self):
        ''' this is a placeholder class, which we can use for
        whatever '''
        pass

class DefaultVariables:
    def __init__(self, config):
        self.config = config
        self.vs = []
        self.rvs = []
        r = self.config.p.r
        num_tokens = self.config.num_tokens
        # the embedding dictionary, which I think is the only shared variable
        self.L = nn.VariableNode([num_tokens, self.config.p.r], None, name='L')
        self.vs.append(self.L)


        # add the attention matrix if needed
        if self.config.p.attention and self.config.p.matrix_attention:
            # add the attention matrix
            left_size = (r * self.config.p.gru_depth) if self.config.p.full_state_attention else r

            # assume bidirectional
            right_size = 2*left_size if self.config.p.bidirectional else left_size

            self.attention_B = nn.VariableNode([left_size, right_size], None, name='attention_B')
            self.vs.append(self.attention_B)


    # add_trainer needs to be added after initializing all
    # of the variables
    def add_trainer(self):
        self.optimizer = nn.AdamOptimizer(self.vs, alpha=self.config.p.lr, beta1=0.9,
                beta2=0.999, epsilon=self.config.p.epsilon)

    def save(self, file_path):
        ''' saves the variables to file_path '''
        with open(file_path, 'wb') as handle:
            pickle.dump(self.vs, handle)

    def load(self, file_path):
        ''' loads the variables and replaces the current values '''
        with open(file_path, 'rb') as handle:
            vs = pickle.load(handle)

        # build a dictionary for the new variables
        vs_dict = {v.name:v for v in vs}

        warned = False
        for v in self.vs:
            if (v.name not in vs_dict) and (not warned):
                print(set([v.name for v in self.vs]))
                print(set(vs_dict.keys()))
                print('in saved but not new')
                print(set(vs_dict.keys()).difference([v.name for v in self.vs]))
                print('in new but not saved')
                print(set([v.name for v in self.vs]).difference(list(vs_dict.keys())))
                print('missing', v.name)
                print(v.name in vs_dict)
                print(list(vs_dict.keys()))
                raise Warning('Some variables not replaced.')
            else:
                v.load(vs_dict[v.name])

    def add_GRUb_block(self, name, bidirectional=False, takes_attention=False):
        ''' this creates a set of parameters for a block of
        GRUbs, based off of the current attentional model
        and stuff

        h_size is set to be r
        x_size is h_size (*2 if bidirectional)
                +(2r if forward and attention)
                +(r if backwards and attention)
                +(r if being fed attention)
        '''
        r = self.config.p.r
        depth = self.config.p.gru_depth
        vs = []
        rvs = []

        GRUbParameters_forward = []

        outputs = Container()
        outputs.forward = GRUbParameters_forward

        # forward pass
        for i in range(depth):
            h_size = r
            x_size = r if i==0 or not bidirectional else 2*r
            if i==0 and self.config.p.structure_data:
                x_size += 4 # depth, arity, parent_arity, leaf_position
            if self.config.p.augmentation: x_size += 2*r
            if takes_attention:
                x_size += r
                if self.config.p.bidirectional:  # assume that bidirectional applies to the attention source
                    x_size += r
            this_GRUb = nn.GRUbParameters(h_size, None, x_size=x_size, name=name + '_GRUb_forward_'+str(i))
            vs += this_GRUb.vs
            rvs += this_GRUb.rvs
            GRUbParameters_forward.append(this_GRUb)

        # backward pass
        if bidirectional:
            GRUbParameters_backward = []
            outputs.backward = GRUbParameters_backward

            for i in range(depth):
                h_size = r
                x_size = r if i==0 or not bidirectional else 2*r
                if i==0 and self.config.p.structure_data:
                    x_size += 4 # depth, arity, parent_arity, leaf_position
                if self.config.p.augmentation: x_size += r
                if takes_attention: x_size += 2*r  # assumes bidirectional input for attention

                this_GRUb = nn.GRUbParameters(h_size, None, x_size=x_size, name=name + '_GRUb_backward_'+str(i))
                vs += this_GRUb.vs
                rvs += this_GRUb.rvs
                GRUbParameters_backward.append(this_GRUb)

        if self.config.p.augmentation:
            augmentation_params = []
            outputs.aug = augmentation_params
            for i in range(depth):
                this_aug = AugmentationVariables(None, r, name+'augmentation_'+str(i), bidirectional=bidirectional)
                vs += this_aug.vs
                rvs += this_aug.rvs
                augmentation_params.append(this_aug)

        # add to the variables
        self.vs+=vs
        self.rvs+=rvs

        return outputs


class DefaultModel:
    def __init__(self, config, variables, train=False):
        self.config = config
        self.v = variables
        self.g = nn.ComputationalGraph(nodes = self.v.vs)
        self.lm = config.lm
        self.attention_has_been_set_up = False
        self.dropout = self.config.p.dropout if train else None
        self.train = train

        # add in regularization if the regularization
        # is not zero.
        if self.config.p.regularization is not None:
            reg_losses = [nn.L2Node(self.config.p.regularization, var, self.g)
                    for var in self.v.rvs]
            self.loss = nn.AddNode(reg_losses, self.g)
        else:
            self.loss = nn.ConstantNode(0.0, graph=self.g)

        # self.attention_memory should be a list of the
        # intermediate states for the GRU block:
        # self.attention_memory[i][j] is the ith input symbol
        # at the jth layer

        if self.config.p.attention:
            self.attention_memory = []

    def set_up_attention(self):
        self.attention_has_been_set_up = True
        if not self.config.p.attention: return

        #print 'attention', len(self.attention_memory),len(self.attention_memory[0])

        prestack = [nn.ConcatNode([layer[i] for layer in self.attention_memory], self.g) for i in range(len(self.attention_memory[0]))]
        #print prestack
        self.stacked_attention_memory = nn.StackNode(prestack, self.g)
        #print 'stacked_memory.shape()', self.stacked_attention_memory.shape()


        if self.config.p.full_state_attention:
            prestack = [nn.ConcatNode([layer[i] for layer in self.attention_memory], self.g) for i in range(len(self.attention_memory[0]))]
            self.to_alpha = nn.StackNode(prestack, self.g)
        else:
            prestack = self.attention_memory[0]
            self.to_alpha = nn.StackNode(prestack, self.g)
            #print len(self.attention_memory),len(self.attention_memory[0]), self.attention_memory[0][0].value.shape
            #print 'to_alpha shape',self.to_alpha.value.shape

        # transpose
        self.to_alpha = nn.TransposeInPlaceNode(self.to_alpha, self.g)

        # to_alpha is (length, rish)
        if self.config.p.matrix_attention:
            self.to_alpha = nn.DotNode(self.v.attention_B, self.to_alpha, self.g)

    def attention(self, state_list):
        assert self.config.p.attention
        assert self.attention_has_been_set_up

        if self.config.p.full_state_attention:
            state = nn.ConcatNode(state_list, self.g)
        else:
            state = state_list[0]

        alpha = nn.DotNode(state, self.to_alpha, self.g)
        #print 'alpha shape', alpha.shape(), self.stacked_attention_memory
        alpha = nn.SoftmaxNode(alpha, self.g)
        newstates = nn.DotNode(alpha, self.stacked_attention_memory, self.g)
        return nn.SplitNode(newstates, self.config.p.gru_depth, self.g)

    def encode(self, token, structure_data=None):
        index = self.config.encode[token]
        out = nn.SingleIndexNode(index, self.v.L, self.g)
        out = nn.DropoutNode(out, self.dropout, self.g)
        if self.config.p.structure_data:
            structure_data_node = nn.ConstantNode(self.config.p.structure_data_scaling*np.array(structure_data), self.g)
            out = nn.ConcatNode([out,structure_data_node], self.g)
        return out

    # returns a list of input vectors corresponding to the stuff.
    def encode_string(self, string, structure_datas = None):
        if self.config.p.structure_data:
            return [self.encode(token, structure_data=sd) for token, sd in zip(string, structure_datas)]
        return [self.encode(token) for token in string]

    def forward_vertical_slice(self, hs, parent, left, input_token, params, structure_data, takes_attention=True):
        takes_attention = takes_attention and self.config.p.attention

        # first construct the actual inputs, which is a bunch of stuff merged together
        if takes_attention: attention_in = self.attention(hs)

        x = self.encode(input_token, structure_data=structure_data)
        out_hs = []
        for i in range(self.config.p.gru_depth):
            x = nn.DropoutNode(x, self.dropout, self.g)
            if self.config.p.augmentation and takes_attention:
                merged_x = nn.ConcatNode([x, parent[i], left[i], attention_in[i]], self.g)
            elif self.config.p.augmentation and not takes_attention:
                merged_x = nn.ConcatNode([x, parent[i], left[i]], self.g)
            elif not self.config.p.augmentation and takes_attention:
                merged_x = nn.ConcatNode([x, attention_in[i]], self.g)
            elif not self.config.p.augmentation and not takes_attention:
                merged_x = x

            x = nn.GRUbCell(hs[i], merged_x, params[i], self.g, dropout=self.dropout)
            out_hs.append(x)

        return out_hs, x

    def gru_block(self, hs, input_tokens, params, hs_backward=None, parents=None,
            left_siblings=None, right_siblings=None, bidirectional=True,
            feed_to_attention=False, structure_data=None):

        # verify the parameters
        feed_to_attention = self.config.p.attention and feed_to_attention
        if self.config.p.augmentation:
            assert left_siblings is not None
            assert parents is not None
            if bidirectional:
                assert right_siblings is not None

        # this does the forward and backwards parts of a gru_block
        xs = self.encode_string(input_tokens, structure_datas=structure_data)
        length = len(input_tokens)

        # memory is a len * depth * directions list
        memory = []
        h_out_forward = []
        h_out_backward = [] if bidirectional else None

        # we proceed layer by layer
        for i in range(self.config.p.gru_depth):
            this_layer_foward = [None] * length

            #forward pass
            h = hs[i]
            for pos in range(length):
                this_params = params[pos]

                this_x = xs[pos]
                this_x = nn.DropoutNode(this_x, self.dropout, self.g)
                if self.config.p.augmentation:
                    # no attention, forward pass
                    parent = parents[pos]
                    parent_x = this_params.aug[i].no_parent if parent==-1 else this_layer_foward[parent]
                    left_sibling = left_siblings[pos]
                    left_sibling_x = this_params.aug[i].no_left_sibling if left_sibling==-1 else this_layer_foward[left_sibling]
                    this_x = nn.ConcatNode([this_x, parent_x, left_sibling_x], self.g)

                h = nn.GRUbCell(h, this_x, this_params.forward[i], self.g, dropout=self.dropout)
                this_layer_foward[pos] = h
            h_out_forward.append(h)

            # backward pass
            if bidirectional:
                this_layer_backward = [None] * length

                #forward pass
                h = hs_backward[i]
                for pos in range(length-1,-1,-1):
                    this_params = params[pos]

                    this_x = xs[pos]
                    this_x = nn.DropoutNode(this_x, self.dropout, self.g)
                    if self.config.p.augmentation:
                        # no attention, forward pass
                        right_sibling = right_siblings[pos]
                        right_sibling_x = this_params.aug[i].no_right_sibling if right_sibling==-1 else this_layer_backward[right_sibling]
                        this_x = nn.ConcatNode([this_x, right_sibling_x], self.g)

                    h = nn.GRUbCell(h, this_x, this_params.backward[i], self.g, dropout=self.dropout)
                    this_layer_backward[pos] = h

                h_out_backward.append(h)
                # now figure out the forward layer thingy
                xs = [nn.ConcatNode(x, self.g) for x in zip(this_layer_foward, this_layer_backward)]
            else:
                xs = this_layer_foward

            memory.append(xs)

        if feed_to_attention:
            self.attention_memory = memory

        # h_out is the forward out or the concatonation of the forward and backward outs
        h_out = [nn.ConcatNode(x, self.g) for x in zip(h_out_forward, h_out_backward)] if bidirectional else h_out_forward

        return h_out    # this is really all we need



''' This is a function that returns information about the graph
structure of a tree, particularly returning the augmentation information
Once it gets called, the string needs to be run through the decoder'''
def merge_graph_structures(gs_list, params_list):
    out_string = []
    out_parents = []
    out_left_siblings = []
    out_right_siblings = []
    out_params = []
    out_depth = []
    out_parent_arity = []
    out_leaf_position = []
    out_arity = []

    for gs, param in zip(gs_list, params_list):
        current_n = len(out_string)

        length = len(gs.string)

        out_params += [param] * length
        out_string += gs.string
        out_parents += [(-1 if x==-1 else x+current_n) for x in gs.parents]
        out_left_siblings += [(-1 if x==-1 else x+current_n) for x in gs.left_sibling]
        out_right_siblings += [(-1 if x==-1 else x+current_n) for x in gs.right_sibling]
        out_depth += gs.depth
        out_parent_arity += gs.parent_arity
        out_leaf_position += gs.leaf_position
        out_arity += gs.arity

    return out_string, out_parents, out_left_siblings, out_right_siblings, \
            out_params, out_depth, out_parent_arity, out_leaf_position, out_arity


# def get_graph_structure(trees, start_symbol=None, intermediate_symbol = None, end_symbol = None):
#     ''' this returns a bunch of things from the annotated tree
#     returns:
#         string:     the string corresponding to the labels
#         parents:    a list for each node containing the index of tha parent
#                     of that node. returns -1 if this is a root node, and
#                     -2 if this is a special symbol.
#         left_sibling: returns the index of the sibling. -1 if it has no
#                     left sibling, -2 if this is a special symbol
#         right_sibling:
#     '''
#     # print TreeInformation(trees, start_symbol=start_symbol,
#     #         intermediate_symbol=intermediate_symbol,
#     #         end_symbol=end_symbol).params()
#     return TreeInformation(trees, start_symbol=start_symbol,
#             intermediate_symbol=intermediate_symbol,
#             end_symbol=end_symbol).params()

class TreeInformation:
    def __init__(self, trees, start_symbol=None,
            intermediate_symbol=None, end_symbol=None):
        self.parents = []
        self.left_sibling = []
        self.right_sibling = []
        self.string = []

        self.depth = []
        self.parent_arity = []
        self.leaf_position = []
        self.arity = []

        self.n=0

        if start_symbol is not None:
            self.right_sibling.append(-1)
            self.parents.append(-1)
            self.left_sibling.append(-1)
            self.string.append(start_symbol)

            self.depth.append(-1)
            self.parent_arity.append(-1)
            self.leaf_position.append(-1)
            self.arity.append(-1)

            self.n+=1

        for i, tree in enumerate(trees):
            self.add_tree(tree)
            self.add_tree_right_siblings(tree)

            if i is not len(trees)-1 and intermediate_symbol is not None:
                self.right_sibling.append(-1)
                self.parents.append(-1)
                self.left_sibling.append(-1)
                self.string.append(intermediate_symbol)

                self.depth.append(-1)
                self.parent_arity.append(-1)
                self.leaf_position.append(-1)
                self.arity.append(-1)

                self.n+=1

        if end_symbol is not None:
            self.right_sibling.append(-1)
            self.parents.append(-1)
            self.left_sibling.append(-1)
            self.string.append(end_symbol)

            self.depth.append(-1)
            self.parent_arity.append(-1)
            self.leaf_position.append(-1)
            self.arity.append(-1)

            self.n+=1

        # verify some stuff
        length = len(self.string)
        assert len(self.right_sibling) == length
        assert len(self.parents) == length
        assert len(self.left_sibling) == length
        assert len(self.depth) == length
        assert len(self.parent_arity) == length
        assert len(self.leaf_position) == length
        assert len(self.arity) == length

    def params(self):
        return self.string, self.parents, self.left_sibling, self.right_sibling, \
                self.depth, self.parent_arity, self.leaf_position, self.arity

    def add_tree(self, tree, parent=-1, left_sibling=-1, depth=0, parent_arity=-1, leaf_position=-1):
        degree = len(tree.leaves)
        this_n = self.n
        tree.ti_index = this_n
        self.parents.append(parent)
        self.left_sibling.append(left_sibling)
        self.string.append(tree.value)
        self.depth.append(depth)
        self.parent_arity.append(parent_arity)
        self.leaf_position.append(leaf_position)
        arity = len(tree.leaves)
        self.arity.append(arity)
        self.n += 1

        prev_n = -1
        for i, c in enumerate(tree.leaves):
            self.add_tree(c, parent=this_n, left_sibling=prev_n, depth=depth+1, parent_arity=arity, leaf_position=i)
            prev_n=c.ti_index

    def add_tree_right_siblings(self, tree, right_sibling = -1):
        self.right_sibling.append(right_sibling)
        degree = len(tree.leaves)

        for i,c in enumerate(tree.leaves):
            if i < degree-1:
                next_right = tree.leaves[i+1].ti_index
            else:
                next_right = -1
            self.add_tree_right_siblings(c, next_right)
