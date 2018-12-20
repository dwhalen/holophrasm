import matplotlib.pyplot as plt
import numpy as np
import pickle as pickle

# a utility function for the plotting
def batch_mean(it,val,window):
    assert len(it) == len(val) #the iteration list should have the same length as the values

    x = []
    y = []

    current_index = 0
    while current_index+window<len(val):
        x.append(it[current_index+window])
        y.append((1.0*sum(val[current_index:current_index+window]))/ window)
        current_index+=window

    return x,y

def color_tuple(color, shade):
    if shade <= 0.5: return [c*2.0*shade for c in color]
    if shade > 0.5:
        shade2 = 2.0 * (shade-0.5)
        return [c * (1.0-shade2) + shade2 for c in color]

class LearningHistory(object):
    def __init__(self, draw_plots=True):
        # defines a figure object, which we'll slowly update
        print("initializing figure")
        self.draw_plots = draw_plots
        if self.draw_plots: self.prep_plot()

        self.current_iteration = 0 # number of training iterations

        self.training_iteration = []
        self.training_loss = []
        self.training_accuracy = []
        self.training_depth_accuracy = []
        self.training_b_accuracy = []
        self.training_bd_accuracy = []

        self.validation_iteration = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.validation_depth_accuracy = []
        self.validation_b_accuracy = []
        self.validation_bd_accuracy = []

        self.test_iteration = []
        self.test_loss = []
        self.test_accuracy = []
        self.test_depth_accuracy = []
        self.test_b_accuracy = []
        self.test_bd_accuracy = []


    def append(self, loss, accs, type = 'training'):
        if type == 'training':
            self.training_iteration.append(self.current_iteration)
            self.training_loss.append(loss)
            self.training_accuracy.append(accs)

            self.current_iteration+=1

        elif type == 'validation':
            self.validation_iteration.append(self.current_iteration)
            self.validation_loss.append(loss)
            self.validation_accuracy.append(accs)

        elif type == 'test':
            self.test_iteration.append(self.current_iteration)
            self.test_loss.append(loss)
            self.test_accuracy.append(accs)


        else:
            raise ValueError("type should be \"training\", \"validation\", or \"test\"")

    def prep_plot(self):
        self.f, self.axarr = plt.subplots(2, sharex=True)
        plt.xlabel('iteration')
        self.axarr[0].set_title('Accuracy')
        self.f.canvas.draw()
        self.axarr[0].set_ylim([0, 1])


    def plot(self):
        if not self.draw_plots: return

        if len(self.training_iteration) + len(self.validation_iteration) + len(self.test_iteration) == 0: return
        # plt.close("all")

        # prints a plot of the learning history for all the parts that have been defined

        if len(self.training_iteration)>0:
            color = (0,1,0)
            num_accs = len(self.training_accuracy[0])
            for i in range(len(self.training_accuracy[0])):
                #x0,y0 = batch_mean(self.training_iteration, [x[i] for x in self.training_accuracy], training_window_size)
                x0,y0 = (self.training_iteration, [x[i] for x in self.training_accuracy])
                self.axarr[0].plot(x0, y0, color = color_tuple(color, (1.0*i+1)/(num_accs+1)))

        if len(self.validation_iteration)>0:
            color = (1,0,0)
            num_accs = len(self.validation_accuracy[0])
            for i in range(len(self.validation_accuracy[0])):
                x0,y0 = (self.validation_iteration, [x[i] for x in self.validation_accuracy])
                self.axarr[0].plot(x0, y0, color = color_tuple(color, (1.0*i+1)/(num_accs+1)))

        if len(self.test_iteration)>0:
            color = (0,0,1)
            num_accs = len(self.test_accuracy[0])
            for i in range(len(self.test_accuracy[0])):
                x0,y0 = (self.validation_iteration, [x[i] for x in self.test_accuracy])
                self.axarr[0].plot(x0, y0, color = color_tuple(color, (1.0*i+1)/(num_accs+1)))

        #self.axarr[0].set_ylim([0, 1])

        # LOSS
        if len(self.training_iteration)>0:
            #x0,y0 = batch_mean(self.training_iteration, self.training_loss, training_window_size)
            x0,y0 = (self.training_iteration, self.training_loss)
            self.axarr[1].plot(x0, y0, color = (0.0,1.0,0.0))

        a = np.array(self.training_loss+self.validation_loss+self.test_loss)
        # print a
        y_min = np.min(a)-0.1
        y_max = np.percentile(a, 90)+1.0
        self.axarr[1].set_ylim([y_min, y_max])

        if len(self.validation_iteration)>0:
            x0,y0 = (self.validation_iteration, self.validation_loss)
            self.axarr[1].plot(x0, y0, color = (1.0,0.0,0.0))

        if len(self.test_iteration)>0:
            x0,y0 = (self.test_iteration, self.test_loss)
            self.axarr[1].plot(x0, y0, color = (0.0,0.0,1.0))

        self.axarr[1].set_title('Loss')



        #plt.show()
        self.f.canvas.draw()

    def save(self, file_path):
        temp = self.f, self.axarr
        self.f=None
        self.axarr=None
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle)
        self.f, self.axarr = temp

    def load(self, load_location):
        this_draw_plots = self.draw_plots

        with open(load_location, 'rb') as handle:
            new = pickle.load(handle)

        new.draw_plots = this_draw_plots

        # close the current plot
        plt.close("all")
        if new.draw_plots:
            new.prep_plot()
            new.plot()
