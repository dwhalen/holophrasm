import learning_history
import data_utils5 as data_utils

import random
import numpy as np
import time

import os
import sys

import pickle as pickle
import matplotlib.pyplot as plt

#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
from multiprocessing import Process, Queue

import signal

global_trainer = None

class Trainer(object):
    def __init__(self, config, load=False, load_location=None, save_location='./weights/', model=None, write_every=10,
        draw_plots=True):
        assert model is not None
        if load_location is None: load_location = save_location
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        self.load_location = load_location
        self.save_location = save_location
        self.p = None
        self.model = model
        self.draw_plots = draw_plots

        global global_trainer
        global_trainer = self

        self.config = config
        self.lm = self.config.lm
        self.training_steps = 0
        self.reset_batch()
        self.write_every = write_every
        self.learning_history = learning_history.LearningHistory(draw_plots=draw_plots)

        self.v = model.Variables(self.config)
        if load:
            self.v.load(self.load_location + '/train.weights')
            self.config.load(self.load_location + '/train.parameters')
            self.learning_history.load(load_location + '/train.history')
        else:
            self.reset_log()

        self.validation_data_set = self.model.validation_data_set(self.lm)
        self.training_data_set = None # we generate this for each epoch
        self.test_data_set = self.model.test_data_set(self.lm)

    def save_session(self, file_name=None):
        if file_name is None: file_name = self.save_location
        start = time.time()
        self.v.save(file_name+'/train.weights')
        self.config.save(file_name+'/train.parameters')
        self.save_learning_history(file_name+'/train.history')
        out_string = 'saved session in '+str(time.time()-start) + 's'
        print(out_string)
        self.add_to_log(out_string)

    def load_learning_history(self, load_location):
        # I think that this will automatically open a plot if there
        # was one already
        with open(load_location, 'rb') as handle:
            self.learning_history = pickle.load(handle)
        if not self.learning_history.draw_plots and self.draw_plots:
            self.learning_history.prep_plot(self)
        self.learning_history.draw_plots = self.draw_plots

    def save_learning_history(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self.learning_history, handle)

    def reset_log(self):
        file_location = self.save_location + '/log.txt'
        with open(file_location, "w") as log_file:
            log_file.write("Starting log")

    def add_to_log(self, string):
        file_location = self.save_location + '/log.txt'
        with open(file_location, "a") as log_file:
            log_file.write(string+'\n')

    # this is kind of ugly: it resets the stored accuracy and loss totals
    def reset_batch(self):
        # this is a lot of stuff to remember.  It's almost impressive...
        self.total_time = np.array([0.0, 0.0, 0.0])
        self.outputs = None
        self.output_counts = None
        self.model_calls = 0

    def add_to_history(self, data_type):
        # now that we've finished a batch, we should add it to the history
        #print
        #print 'adding',data_type

        if self.output_counts is None:
            #print 'empty outputs'
            return
        for x in self.output_counts: assert x > 0

        output_means = [total/count for total,count in zip(self.outputs, self.output_counts)]
        avg_loss = output_means[0]
        acc_list = output_means[1:]
        avg_time = self.total_time / self.model_calls

        self.learning_history.append(avg_loss, acc_list, type=data_type)
        out_string = self.batch_string(data_type)
        print(out_string)
        self.add_to_log(out_string)

        self.reset_batch()

    def batch_string(self,data_type):
        for x in self.output_counts: assert x > 0
        assert self.model_calls > 0

        output_means = [total/count for total,count in zip(self.outputs, self.output_counts)]
        avg_loss = output_means[0]
        acc_list = output_means[1:]
        avg_time = self.total_time / self.model_calls

        out_string = '\r{0:10s} loss = {1:6.3f}, outputs ='.format(data_type, avg_loss)
        for x in acc_list:
            out_string+= ' {0:4.1f}'.format(100.0*x)
        out_string += ', model_time ='
        for t in avg_time:
            out_string += ' {0:6.5f},'.format(t)
        out_string += ' at '+time.strftime("%Y-%m-%d %H:%M:%S")
        return out_string

    def batch_call(self, i):
       # runs the model and returns a tuple, a dictionary
        # d_list, outputs, output_counts
        data_type = self.current_data_type
        if data_type == 'validation':
            proof_step= self.validation_data_set[i]
        elif data_type == 'training':
            proof_step = self.training_data_set[i]
        elif data_type == 'training':
            proof_step = self.test_data_set[i]
        model = self.model.Model(self.v, self.config, proof_step, train=(data_type=='training') )
        d_list = [v.d for v in self.v.vs]
        return d_list, model.outputs, model.output_counts


    def steps(self, proof_steps, data_type='training', multiprocessing=None):
        self.current_data_type = data_type
        assert len(proof_steps) > 0

        # create the model
        t0 = time.time()

        if multiprocessing is not None:

            #t1 = time.time()
            if self.reorder:
                reorder(proof_steps, data_type)
            #self.total_time[1]+=time.time()-t1
            # multiprocessing
            t2 = time.time()
            with withPool(multiprocessing) as p:
                self.total_time[1]+=time.time()-t2
                self.p=p.p
                t3 = time.time()
                out = self.p.map(batch_call_multi, proof_steps, chunksize=1)
                self.total_time[2]+=time.time()-t3
        else:
            # normalprocessing
            t3 = time.time()
            out = list(map(batch_call_multi, proof_steps))
            self.total_time[2]+=time.time()-t3

        #print [(np.mean(x[0][0])) for x in out]

        # now run the minimize operation
        #t4 = time.time()
        if data_type == 'training':
            self.v.optimizer.minimize(d_vars=[sum(x[0][i] for x in out) for i in range(len(self.v.vs))])
        #self.total_time[4]+=time.time()-t4

        # update the model counts
        self.total_time[0] += time.time()-t0
        for x in out:
            _, outputs, output_counts = x
            self.model_calls+=1

            # update the state information
            if self.outputs is None:
                self.outputs = outputs
                self.output_counts = output_counts
            else:
                for i in range(len(outputs)):
                    self.outputs[i]+=1.0*outputs[i]
                    self.output_counts[i]+=1.0*output_counts[i]

        if self.model_calls % self.write_every == 0:
            sys.stdout.write(self.batch_string(data_type))
            sys.stdout.flush()

    def run_epoch(self):
        self.last_save_time = time.time()

        epoch_start_time = time.time()
        self.training_data_set = self.model.training_data_set(self.lm)
        data_set = self.training_data_set
        self.index = 0
        self.reset_batch()

        # make sure that we get rid of all the processes
        if self.p is not None:
            self.p.close()
            self.p.join()

        batch = []
        for tindex in range(len(data_set)):
            t = data_set[tindex]
            self.index+=1
            batch.append(tindex)
            if len(batch) == self.batch_size or (len(batch)>0 and tindex==len(data_set)-1):
                # if we've filled out a batch or
                self.steps(batch, data_type='training', multiprocessing=self.multiprocessing)
                batch = []

            if (self.index % self.plot_every) == 0:
                self.add_to_history('training')
                self.learning_history.plot()
                self.reset_batch()

            if self.save_every is not None and time.time()-self.last_save_time > 60 * self.save_every:
                self.last_save_time = time.time()
                self.save_session()

            if self.validate_every is not None and (self.index % self.validate_every == 0):
                self.add_to_history('training')
                self.learning_history.plot()
                self.reset_batch()
                self.validate()

            ####
            if self.early_stop and self.index>self.early_stop: return

        # add any stragglers to the training
        self.add_to_history('training')
        self.learning_history.plot()
        self.reset_batch()

        self.validate()

        # save the session every epoch.  Possibly this should happen more often.
        # takes ~10 seconds
        if self.save_epoch:
            self.save_session(file_name=self.save_location)

        out_string = 'total epoch time {0:11.2f}'.format( time.time()-epoch_start_time )
        print(out_string)
        self.add_to_log(out_string)


    def validate(self):
        # validation
        batch = []
        data_set = self.validation_data_set
        for tindex in range(len(data_set)):
            t = data_set[tindex]
            batch.append(tindex)
            if len(batch) == self.batch_size or (len(batch)>0 and tindex==len(data_set)-1):
                # if we've filled out a batch or
                self.steps(batch, data_type='validation', multiprocessing=self.multiprocessing)
                batch = []
        val = (1.0*self.outputs[0])/self.output_counts[0]
        self.add_to_history('validation')
        self.learning_history.plot()
        self.reset_batch()

        # validation decay
        if val > self.last_val:
            out_string = 'Reducing learning rate '+str(self.config.p.lr)+' -> '+str(self.config.p.lr / self.config.p.lr_reduction)
            print(out_string)
            self.add_to_log(out_string)
            self.config.p.lr /= self.config.p.lr_reduction
            self.v.optimizer.update_learning_rate(self.config.p.lr)  # this works, although I don't like it.
        self.last_val = val

    def test(self):
        # validation
        self.reset_batch()
        self.early_stop = None
        batch = []
        data_set = self.test_data_set
        print('number of test data,', len(data_set))
        for tindex in range(len(data_set)):
            t = data_set[tindex]
            batch.append(tindex)
            if len(batch) == self.batch_size or (len(batch)>0 and tindex==len(data_set)-1):
                # if we've filled out a batch or
                self.steps(batch, data_type='test', multiprocessing=self.multiprocessing)
                batch = []
        self.add_to_history('test')
        self.learning_history.plot()
        self.reset_batch()

    def run_many_epochs(self,lm,
            plot_every=1000, write_every=10, early_stop=None,
            save_every=10000,
            multiprocessing=None, batch_size=50,
            validate_every=None, reorder=True):
        self.reorder = reorder
        self.plot_every = plot_every
        self.write_every = write_every
        self.early_stop = early_stop
        self.save_every = save_every
        self.save_best = True       # because saving is not fast
        self.save_epoch = True      # because saving is not fast
        self.multiprocessing = multiprocessing
        self.batch_size = batch_size
        self.validate_every = validate_every

        self.best_val = 10000000
        self.last_val = 10000000

        print('running epochs')

        global global_trainer
        global_trainer = self

        best_val_epoch = 0
        best_val = 10000000
        epoch = -1
        while True:
            epoch += 1
            if self.config.p.max_epochs is not None and epoch >= self.config.p.max_epochs: return
            self.run_epoch()
            out_string = 'finished epoch '+str(epoch)+'\n'
            print(out_string)
            self.add_to_log(out_string)
            #print 'added'
            if not early_stop is None: return
            val = self.learning_history.validation_loss[-1]
            if val < best_val:
                best_val = val
                best_val_epoch = epoch
                if self.save_best:
                    out_string = 'saving best epoch'
                    print(out_string)
                    self.add_to_log(out_string)
                    self.v.save(self.save_location+'/best.weights')
                    print('saved best')

def init_func():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class withPool:
    def __init__(self, procs):
        self.p = Pool(procs, init_func)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.close()

def proof_step_difficulty(i, data_type):
    sel = global_trainer
    # approximate the difficulty of the proof step
    if data_type == 'validation':
        t=sel.validation_data_set[i]
    elif data_type == 'training':
        t=sel.training_data_set[i]
    elif data_type == 'test':
        proof_step = sel.test_data_set[i]
    return (t.tree.size() + sum(h.tree.size() for h in t.prop.hyps if h.type=='e')
            +sum(h.tree.size() for h in t.context.hyps if h.type=='e'))

def reorder(proof_steps, data_type):
    decorated = [(-1*proof_step_difficulty(t, data_type) , t) for t in proof_steps]
    decorated.sort()
    proof_steps[:] = [t for _, t in decorated]


def batch_call_multi(i):
    sel = global_trainer
    data_type = sel.current_data_type
    if data_type == 'validation':
        proof_step=sel.validation_data_set[i]
    elif data_type == 'training':
        proof_step = sel.training_data_set[i]
    elif data_type == 'test':
        proof_step = sel.test_data_set[i]
    model = sel.model.Model(sel.v, sel.config, proof_step, train=(data_type=='training') )
    d_list = [1.0*v.d for v in sel.v.vs]
    return d_list, model.outputs, model.output_counts
