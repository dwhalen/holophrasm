'''
This is a slightly more resilient version of multiprocessing pool when dealing
with keyboard interrupts.

It currently implements map and apply.

setting the number of processes to None makes this single threaded
'''

import multiprocessing
import signal

def init_func():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class Pool:
    def __init__(self, procs):
        self.procs = procs
        if self.procs is None:
            self.p = None
        else:
            self.p = multiprocessing.Pool(procs, init_func, maxtasksperchild=1)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.procs is None:
            return
        else:
            #print 'closing pool'
            self.p.close()
            self.p.terminate()
            self.p.join()
            self.p = None
    def apply(self, *args, **kwargs):
        if self.procs is None:
            return apply(*args, **kwargs)
        else:
            return self.p.apply(*args, **kwargs)
    def map(self, *args, **kwargs):
        if self.procs is None:
            return list(map(*args))
        else:
            return self.p.map(*args, **kwargs)
