import nnlibrary as nn
from models import *
import data_utils5 as data_utils
import numpy as np

''' this is the object for training the generative model, which should be heavily
based on the structure defined in models.py.'''

def validation_data_set(lm):  # lm is a language model
    return [t for t in lm.validation_proof_steps if t.prop.unconstrained_arity()>0]

def test_data_set(lm):
    return [t for t in lm.test_proof_steps if t.prop.unconstrained_arity()>0]

def training_data_set(lm):
    np.random.seed()
    return np.random.permutation([t for t in lm.training_proof_steps if t.prop.unconstrained_arity()>0])


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
        self.known_gru_block = self.add_GRUb_block('known', bidirectional = self.config.p.bidirectional)
        self.to_prove_gru_block = self.add_GRUb_block('to_prove', bidirectional = self.config.p.bidirectional)
        self.out_gru_block = self.add_GRUb_block('out', bidirectional=False, takes_attention=self.config.p.attention)

        self.forward_start = [nn.VariableNode([r], None, name='forward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
        self.vs += self.forward_start
        if self.config.p.bidirectional:
            self.backward_start = [nn.VariableNode([r], None, name='backward_h_start'+str(i)) for i in range(self.config.p.gru_depth)]
            self.vs+=self.backward_start
        else:
            self.backward_start=None

        # the parameters for the output section
        self.last_W = nn.VariableNode([r,self.config.num_tokens], None, name='last_W')
        self.last_b = nn.VariableNode([self.config.num_tokens], None, name='last_b')

        self.out_Ws = [nn.VariableNode([r,r], None, name='out_W_'+str(i)) for i in range(out_layers)]
        self.out_bs = [nn.VariableNode([r], None, name='out_b_'+str(i)) for i in range(out_layers)]
        self.vs += [self.last_W, self.last_b] + self.out_Ws + self.out_bs
        self.rvs += [self.last_W] + self.out_Ws

        # the parameters for the middle section
        in_dim = 2*r if self.config.p.bidirectional else r
        self.middle_W = [nn.VariableNode([in_dim, r], None, name='middle_W'+str(i)) for i in range(self.config.p.gru_depth)]
        self.middle_b = [nn.VariableNode([r], None, name='middle_b'+str(i)) for i in range(self.config.p.gru_depth)]
        self.vs += self.middle_W + self.middle_b
        self.rvs += self.middle_W

        # we always need to add the trainer
        self.add_trainer()

class Model(DefaultModel):
    def __init__(self, variables, config, proof_step, train=False):
        ''' this is the model.  As a single pass, it processes the
        inputs, and computes the losses, and runs a training step if
        train.
        '''
        DefaultModel.__init__(self, config, variables, train=train)
        if not self.train:
            np.random.seed(proof_step.context.number +
                    + proof_step.prop.number + proof_step.tree.size())

        self.parse_and_augment_proof_step(proof_step)

        # merge the inputs together so that we can bidirection it
        in_string, in_parents, in_left, in_right, in_params, depths, parent_arity, leaf_position, arity = merge_graph_structures(
                [self.known_graph_structure, self.to_prove_graph_structure],
                [self.v.known_gru_block, self.v.to_prove_gru_block])

        # print
        # print in_string
        # print in_parents
        # print in_left
        # print in_right
        # print depths
        # print parent_arity
        # print leaf_position
        # print arity

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



        # process the right side
        out_string=self.out_graph_structure.string
        out_parents=self.out_graph_structure.parents
        out_left = self.out_graph_structure.left_sibling
        arity = self.out_graph_structure.arity
        leaf_position = self.out_graph_structure.leaf_position
        parent_arity = self.out_graph_structure.parent_arity
        depths = self.out_graph_structure.depth
        structure_data = list(zip(depths, parent_arity, leaf_position, arity))

        out_length = len(out_string)
        out_xs = []
        all_hs = []
        hs = from_middle
        for i in range(out_length):
            # figure out the augmentation stuff.
            if self.config.p.augmentation:
                parent = out_parents[i]
                parent_hs = [x.no_parent for x in self.v.out_gru_block.aug] if parent==-1 else all_hs[parent]
                left = out_left[i]
                left_hs = [x.no_left_sibling for x in self.v.out_gru_block.aug] if left==-1 else all_hs[left]
            else:
                parent_hs = None
                left_hs = None

            hs, x = self.forward_vertical_slice(hs, parent_hs, left_hs,
                    out_string[i], self.v.out_gru_block.forward,
                    structure_data[i],
                    takes_attention=self.config.p.attention)
            all_hs.append(hs)
            out_xs.append(x)

        # test

        # calculate logits and score
        self.correct_string = out_string[1:]+['END_OF_SECTION']
        #out_xs = [nn.ZerosNode([64], self.g) for token in correct_string]
        self.all_correct = True
        self.num_correct = 0
        self.all_logits = [self.x_to_predictions(x) for x in out_xs]
        all_costs = [self.score(logits, c_token)
                for logits, c_token in zip(self.all_logits, self.correct_string)]
        perplexity = nn.AddNode(all_costs, self.g)

        #self.logit_matrix = np.concat([l.value for l in all_logits])

        self.loss = nn.AddNode([perplexity, self.loss], self.g)

        # and train
        if train:
            # print 'training2'
            # print len(self.v.vs), len(self.g.nodes)
            self.g.backprop(self.loss)
            # for v in self.v.vs:
            #     print v.name, np.mean(v.value ** 2)
            #self.v.optimizer.minimize()

        # put the outputs in the standard training format
        self.outputs = [perplexity.value, self.num_correct, 1*self.all_correct]
        self.output_counts = [out_length, out_length, 1]

        # print 'correct', self.correct_string
        # print 'predictions:', [self.config.decode[np.argmax(logits.value)] for logits in all_logits]
        # print

        # DEBUG PRINTING STUFF
        # WTF is going wrong with the perplexity?
        # print 'total perplexity, mean', perplexity.value, perplexity.value/out_length, out_length
        # if out_length>10:
        #     print proof_step.context.label
        #     print out_length
        #     print proof_step.tree.stringify()
        #     print proof_step.prop.tree.stringify()
        #     print self.out_tree.stringify()

    def x_to_predictions(self, x):
        x = nn.DropoutNode(x, self.dropout, self.g)

        for W, b in zip(self.v.out_Ws, self.v.out_bs):
            x = nn.RELUDotAdd(x, W, b, self.g)
            x = nn.DropoutNode(x, self.dropout, self.g)

        logits = nn.RELUDotAdd(x, self.v.last_W, self.v.last_b, self.g)
        #print logits.value
        #print
        return logits

    def score(self, logits, correct):
        correct_index = self.config.encode[correct]

        # check if correct
        this_correct = np.argmax(logits.value) == correct_index
        if this_correct:
            self.num_correct += 1
        else:
            self.all_correct = False

        loss = nn.SoftmaxCrossEntropyLoss(correct_index, logits, self.g)
        return loss

    def parse_and_augment_proof_step(self, proof_step, out=True):
        ''' this gets the data for the proof step, but also does the
        randomization and the augmentation.  There are a number of
        strange choices in the augmentation, particularly with keeping
        track of the distinct unconstrained variables but we'll leave
        them as is for now. '''
        prop = proof_step.prop

        # figure out the unconstrained variables in prop, create
        # the corresponding replacement rules and such.
        # Eventually I'll distinguish the different non-target variables
        # but that is a task for another day.
        unconstrained_variables = prop.unconstrained_variables
        uv_dict = {var:'UC' for var in unconstrained_variables}
        target_index = np.random.choice(len(unconstrained_variables))
        target_variable = unconstrained_variables[target_index]
        uv_dict[target_variable] = 'TARGET_UC'
        if out: self.out_tree = proof_step.unconstrained[target_index].copy()

        # figure out the fit of prop to the proof step.
        # now all the variables should be in fit or in uv_dict
        fit = data_utils.prop_applies_to_statement(
                proof_step.tree, prop, proof_step.context)

        self.to_prove_trees = [hyp.tree.copy().replace(fit).replace_values(uv_dict)
            for hyp in prop.hyps if hyp.type == 'e']
        self.known_trees = [hyp.tree.copy()
            for hyp in proof_step.context.hyps if hyp.type == 'e']

        # now generate a random replacement dictionary
        random_replacement_dict = self.config.lm.random_replacement_dict_f(f=proof_step.context.f)
        self.random_replacement_dict = random_replacement_dict

        # perform the replacements
        if out: self.out_tree.replace_values(random_replacement_dict)
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
        if out: self.out_graph_structure = TreeInformation([self.out_tree],
                start_symbol='START_OUTPUT', intermediate_symbol=None,
                end_symbol=None)
