import cpuinfo
info = cpuinfo.get_cpu_info()
print(('Vendor ID: {0}'.format(info['vendor_id'])))
print(('Brand: {0}'.format(info['brand'])))
print(('Hz Advertised: {0}'.format(info['hz_advertised'])))
print(('Hz Actual: {0}'.format(info['hz_actual'])))
print(('Count: {0}'.format(info['count'])))


print()
print()

import numpy as np
from sys import getsizeof
from tree_parser import *
import pickle
import random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from data_utils5 import *
import multitrainer as trainer

text = file_contents()
database = meta_math_database(text,n=None)
print()
language_model = LanguageModel(database)

print()
print()

# this is the main routine for actual training
#import pred_model as model
import gen_model_train as model

config = model.Config(language_model)
config.p.lr = 1.0e-3
config.p.r = 128
config.regularization = 1.0e-4
config.p.lr_reduction = 2
config.p.gru_depth = 1
config.p.dropout = None
config.p.augmentation= False
config.p.structure_data = True
config.p.attention=True
config.p.full_state_attention=False
config.p.bidirectional = False
config.p.out_layers = 1

config.p.max_epochs = None
train_object = trainer.Trainer(config, load=False,
        draw_plots=False, save_location='./weights/gen', model=model)

train_object.v.optimizer.gradient_clipping = 100.0 # clip the norm to this amount

print()
print()

train_object.run_many_epochs(language_model, plot_every=1000,
                             write_every=10, early_stop=None, save_every=30,
                             validate_every = 150000,
                             batch_size=10, multiprocessing=None)
