import gen_model_train as model

Config = model.Config
Variables = model.Variables

single_model = model.Model

class Model():
    def __init__(self, variables, config, proof_step, train=False, ensemble=10):
        assert not train
        assert ensemble > 0
        models = [model.Model(variables, config, proof_step, train=False) for _ in range(ensemble)]
        self.sample = models[0]

        self.all_correct=True
        self.num_correct=0

        correct_string = self.sample.correct_string

        loss = 0
        for i in range(len(correct_string)):
            token = correct_string[i]
            logits = [m.all_logits[i] for m in models]
            sum_logits = nn.AddNode(logits, None)
            mean_logits = nn.MultiplyNode(1.0/ensemble, sum_logits, None)

            loss += self.score(mean_logits, token).value

        self.outputs = [loss, num_correct, all_correct]
        self.output_counts = [len(correct_string), len(correct_string), 1]

    def score(self, logits, correct):
        correct_index = self.sample.config.encode[correct]

        # check if correct
        this_correct = np.argmax(logits.value) == correct_index
        if this_correct:
            self.num_correct += 1
        else:
            self.all_correct = False

        loss = nn.SoftmaxCrossEntropyLoss(correct_index, logits, self.g)
        return loss
