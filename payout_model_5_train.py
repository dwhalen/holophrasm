import nnlibrary as nn
from models import *
import data_utils5 as data_utils
import numpy as np
import random


import pickle as pickle



'''
This loads the data sets and stores them as a global variable
Yes, it's probably ta bad idea.  No, I don't care.

The individual data will each be a tuple (tree, hyps, correct)
'''
data_loaded = False

def list_to_data(list_of_pds):
    out = []
    for thing in list_of_pds:
        for tree in thing.correct:
            out.append( (tree, thing.hyps, True) )
        for tree in thing.wrong:
            out.append( (tree, thing.hyps, False) )
    return out

def load_payout_data():
    payout_data = []
    for i in range(10):
        with open('payout_data_'+str(i), 'rb') as handle:
            payout_data.append( pickle.load(handle) )
        print('loaded data', i)

    global global_validation_data
    global global_test_data
    global global_training_data
    global_validation_data = list_to_data(payout_data[0])
    global_test_data = list_to_data(payout_data[1])
    global_training_data = list_to_data(payout_data[2]+payout_data[3]+
            payout_data[4]+payout_data[5]+payout_data[6]+payout_data[7]+
            payout_data[8]+payout_data[9])
    global data_loaded
    data_loaded = True
    print('processed data', len(global_training_data), len(global_validation_data), len(global_test_data))

'''
Standard neural network stuff
'''

def validation_data_set(lm):  # lm is a language model
    if not data_loaded: load_payout_data()
    assert data_loaded
    return global_validation_data

def test_data_set(lm):
    if not data_loaded: load_payout_data()
    assert data_loaded
    return global_test_data

def training_data_set(lm):
    if not data_loaded: load_payout_data()
    assert data_loaded
    np.random.seed()
    return np.random.permutation(global_training_data)


class Config(DefaultConfig):
    def __init__(self, language_model):
        # use the standard config initialization
        DefaultConfig.__init__(self, language_model)
        self.p.bidirectional = False
        self.p.out_layers = 1

class Variables(DefaultVariables):
    def __init__(self, config):
        DefaultVariables.__init__(self, config)
        # define the specific variables for this section
        r = self.config.p.r
        out_layers = self.config.p.out_layers

        # the parameters for the input section
        self.hyp_gru_block = self.add_GRUb_block('hyp', bidirectional = self.config.p.bidirectional)
        self.statement_gru_block = self.add_GRUb_block('statement', bidirectional = self.config.p.bidirectional)

        self.forward_start = [nn.VariableNode([r], None, name='forward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
        self.vs += self.forward_start
        if self.config.p.bidirectional:
            self.backward_start = [nn.VariableNode([r], None, name='backward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
            self.vs+=self.backward_start
        else:
            self.backward_start=None

        out_size = r * self.config.p.gru_depth * (2 if self.config.p.bidirectional else 1)
        self.main_first_W = nn.VariableNode([out_size, r], None, name='main_first_W')
        self.main_first_b = nn.VariableNode([r], None, name='main_first_b')

        self.main_Ws = [nn.VariableNode([r, r], None, name='main_W_'+str(i)) for i in range(self.config.p.out_layers)]
        self.main_bs = [nn.VariableNode([r], None, name='main_b_'+str(i)) for i in range(self.config.p.out_layers)]

        self.last_W = nn.VariableNode([r, 1], None, name='last_W_')
        self.last_b = nn.VariableNode([1], None, name='last_b_')

        self.vs+=[self.main_first_W, self.main_first_b,
                self.last_W, self.last_b]+self.main_Ws + self.main_bs
        self.rvs+=[self.main_first_W, self.last_W]+self.main_Ws

        # we always need to add the trainer
        self.add_trainer()

class Model(DefaultModel):
    def __init__(self, variables, config, proof_step, train=False):
        ''' this is the model.  As a single pass, it processes the
        inputs, and computes the losses, and runs a training step if
        train.
        '''

        # we just defined the proof step as a triple
        (tree, hyps, correct_output) = proof_step

        DefaultModel.__init__(self, config, variables, train=train)

        # fix the random seed
        if not self.train:
            np.random.seed(tree.size() + 100 * len(hyps) + 10000 * correct_output)

        correct_score = self.get_score(
                tree, hyps, None
                )

        wrong_score = nn.ConstantNode(np.array([0.0]), self.g)
        correct_output = 1*correct_output

        logits = nn.ConcatNode([wrong_score, correct_score], self.g)
        cross_entropy = nn.SoftmaxCrossEntropyLoss(correct_output, logits, self.g)
        self.loss = nn.AddNode([self.loss, cross_entropy], self.g)

        accuracy = 1 * (np.argmax(logits.value) == correct_output)
        self.outputs = [cross_entropy.value, accuracy, 1-correct_output]
        self.output_counts = [1, 1, 1]

        # perform the backpropagation if we are training
        if train:
            self.g.backprop(self.loss)

    def get_score(self, statement, hyps, f):
        in_string, in_parents, in_left, in_right, in_params, depths, \
                parent_arity, leaf_position, arity = self.parse_statement_and_hyps(
                statement, hyps, f)

        #print in_string
        to_middle = self.gru_block(self.v.forward_start, in_string, in_params,
                hs_backward=self.v.backward_start, parents=in_parents,
                left_siblings=in_left, right_siblings=in_right,
                bidirectional=self.config.p.bidirectional,
                structure_data = list(zip(depths, parent_arity, leaf_position, arity)),
                feed_to_attention=False)

        h = nn.ConcatNode(to_middle, self.g)
        h = nn.DropoutNode(h, self.dropout, self.g)
        h = nn.RELUDotAdd(h, self.v.main_first_W, self.v.main_first_b, self.g)
        h = nn.DropoutNode(h, self.dropout, self.g)
        for i in range(self.config.p.out_layers):
            h = nn.RELUDotAdd(h, self.v.main_Ws[i], self.v.main_bs[i], self.g)
            h = nn.DropoutNode(h, self.dropout, self.g)
        h = nn.DotNode(h, self.v.last_W, self.g)
        h = nn.AddNode([h, self.v.last_b], self.g)

        return h

    def get_wrong_proof_step(self, t):
        options = [step for step in t.context.entails_proof_steps
                if not (step.prop.type=='f' or step.prop.type == 'e' or
                step.height == t.height)]
        if len(options) == 0:
            return None
        # lower height means easier
        wrong = np.random.choice(len(options), 1)[0]
        wrong = options[wrong]

        self.easier_proof_step = 0 if t.height < wrong.height else 1
        return wrong

    def parse_statement_and_hyps(self, statement, hyps, f):
        # statement is a tree, hyps is a list of hyps (we'll filter for e-type)
        random_replacement_dict = self.config.lm.random_replacement_dict_f(f=f)

        statement = statement.copy().replace_values(random_replacement_dict)
        hyps = [h.copy().replace_values(random_replacement_dict) for h in hyps]

        # and get the graph structures
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        return merge_graph_structures(
                [statement_graph_structure, hyps_graph_structure],
                [self.v.statement_gru_block, self.v.hyp_gru_block])

def get_payout(vs, tree, context):
    scores = [PayoutModel(v, tree, context).score for v in vs]
    return sum(scores)/len(scores)

class PayoutModel(Model):
    def __init__(self, variables, tree, context):
        ''' this is the model.  As a single pass, it processes the
        inputs, and computes the losses, and runs a training step if
        train.
        '''
        
        hyps = [h.tree for h in context.hyps if h.type=='e']
        #print 'payout_hyps', hyps

        DefaultModel.__init__(self, variables.config, variables, train=False)
        self.g = None
        
        self.score = self.get_score(
                tree, hyps, None
                ).value[0]
        #print 'score', self.score