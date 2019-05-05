# C.B. Scott, December 2018

# Code to reproduce experiments in 
# "Multilevel Artificial Neural Network Training for Spatially Correlated Learning"
# by C.B. Scott, E. Mjolsness. 
# https://arxiv.org/abs/1806.05703
# Dependencies:

#    A copy of MNIST, resized to 32 x 32 and flattened.
#    Numpy v1.15.2+.
#    Scipy v1.1.0+.
#    Tensorflow vX, where 1.10.1 <= X < 2.0 (note: many functions were deprecated/removed in TF 2.0).

# Contact scottcb AT uci DOT edu with questions.

# Some imports:
import tensorflow as tf
import numpy as np
from scipy.linalg import circulant, dft, orth
from scipy.spatial.distance import cdist
from MsANNFeed import *

# Supress some of TF's error reporting. 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128 


# Build a list of weight variables with the same 'aspect ratio', that is:
# If dims is (X, Y), this creates a list of variables with sizes
# (aX, aY), (bX, bY) ... 
# the list is (max_depth + 1) long. 
# You must also supply a function which produces the new dimensions (aX, aY) given (X,Y)
def variable_stack(dims, name="", max_depth=0, red=lambda x:x, init=tf.zeros_initializer()):
    V = tf.get_variable(name+"%d" % max_depth, shape=dims, initializer=init)
    tf.add_to_collections("depth_%d" % max_depth, V)
    if max_depth <= 0:
        return [V]
    else:
        return [V] + variable_stack(red(dims), name=name, max_depth=max_depth-1,red=red, init=init)
    
# Given a list of vairable dimensions, return the list with each dimension halved. 
def red2(dims):
    return tuple([int(item/2) for item in dims])

# Returns a 1D interpolation matrix of dimensions (nn x (nn/2))
def getPmatr1D(nn, mm):
    assert nn == 2*mm
    pi_init = np.zeros((nn, int(nn/2)))
    pi_init[np.arange(nn), np.floor(np.arange(nn)/2).astype('int')] = 1.0/np.sqrt(2.0)
    return pi_init.astype('float32')

# Returns one of three types of prolongation matrix:
# 
def getPmatr(nn, mm, mode='grid'):
    assert nn == 2*mm
    if mode == 'path':
        pi_init = np.zeros((nn, int(nn/2)))
        pi_init[np.arange(nn), np.floor(np.arange(nn)/2).astype('int')] = 1.0/np.sqrt(2.0)
        return pi_init.astype('float32')
    elif 'grid' in mode:
        ns = int(np.sqrt(nn))
        ms = int(np.sqrt(mm))
        if ns == np.sqrt(nn):
            tr = np.kron(getPmatr1D(ns, int(ns/2)), getPmatr1D(ns, int(ns/2)))
            tr1 = tr.copy()
            tr2 = tr.copy()
            tr1[::2,:] = 0
            tr2[1::2,:] = 0
            tr = np.sqrt(2.0)*np.concatenate([tr1,tr2],axis=1)
        else:
            tr = (1.0/np.sqrt(2.0))*np.concatenate([np.eye(mm),np.eye(mm)],axis=0)
        if 'shuf' in mode:
            np.random.shuffle(tr)
        return tr.astype('float32')
    else:
        raise Exception
    
def testProlong(old_dims, new_dims, var, restrict=False):
    if not restrict:
        pmatrs = [getPmatr(new_dims[i], old_dims[i]) for i in range(len(old_dims))]
    else:
        pmatrs = [np.transpose(getPmatr(new_dims[i], old_dims[i])) for i in range(len(old_dims))]
    if len(old_dims) > 1:
        return tf.matmul(tf.matmul(pmatrs[0], var), pmatrs[1], transpose_b = True)
    else:
        return tf.squeeze(tf.matmul(pmatrs[0], tf.expand_dims(var,axis=-1)))

def prolong(var):
    dims = var.get_shape().as_list()
    return tf.matmul(
        tf.matmul(getPmatr(2*dims[0],dims[0]),var),
        getPmatr(2*dims[1],dims[1]),
        transpose_b=True
    )

def restrict(var):
    dims = var.get_shape().as_list()
    return tf.matmul(
        tf.matmul(getPmatr(dims[0],int(dims[0]/2)), var, transpose_a=True),
        getPmatr(dims[1],int(dims[1]/2))
    )

def getmultiP(nn, mm):
    P = np.eye(mm)
    while P.shape[0] < nn:
        P = np.dot(getPmatr(2*P.shape[0],P.shape[0]), P)
    return P.astype('float32')
        
def multiprolong(dims, var):
    old_dims = var.get_shape().as_list()
    if len(dims) > 1:
        P_prior = getmultiP(dims[0], old_dims[0])
        P_post = getmultiP(dims[1], old_dims[1])
        return tf.matmul(tf.matmul(P_prior, var), P_post, transpose_b=True)
    else:
        P = getmultiP(dims[0],old_dims[0])
        return tf.squeeze(tf.matmul(P, tf.expand_dims(var,axis=-1)))

class MsAEN:
    def __init__(self,
            mspec,
            smoothing = 1,
            depth = 0,
            lamb = 0,
            red = red2,
            df = None
                ):
        self.d = depth
        self.nln = tf.nn.sigmoid
        self.build_model(mspec, depth, red)
        self.build_optimizers()
        if df is None:
            self.df = DataFeeder(mspec[0])
        else: 
            self.df = df
        self.cost=0.0
        self.history=[]
        self.sess = None
        self.budget = 256000
    
    def build_model(self, layers, max_depth, red):
        self.input = tf.placeholder(tf.float32, [None,layers[0]])
        self.var_list = []
        self.bias_list = []
        #current = [self.input for i in range(max_depth+1)]
        current = self.input
        prolong_ops_list = [[] for i in range(max_depth+1)]
        restrict_ops_list = [[] for i in range(max_depth+1)]
        for i in range(1, len(layers)):
            new_weight_stack = variable_stack((layers[i-1],layers[i]),
                                        name="layer%dweight"%i,
                                        max_depth=max_depth,
                                        red=red,
                                        init=tf.glorot_normal_initializer()
                                       )
            new_bias_stack = variable_stack((layers[i],),
                                        name="layer%dbias"%i,
                                        max_depth=max_depth,
                                        red=red,
                                        init=tf.zeros_initializer()
                                       )
            p_weights = [multiprolong((layers[i-1],layers[i]),new_weight_stack[j]) for j in range(len(new_weight_stack))]
            p_biases = [multiprolong((layers[i],),new_bias_stack[j]) for j in range(len(new_bias_stack))]
            #print(sum(p_weights[:0]))
            #quit()
            #current = [self.nln(
            #    tf.matmul(current[j], sum(p_weights[:j+1])) + sum(p_biases[:j+1])
            #    ) for j in range(len(current))
            #]
            current = self.nln(tf.matmul(current,sum(p_weights)) + sum(p_biases))
            #print([item.shape for item in current])
            for j in range(len(new_weight_stack)-1):
                prolong_ops_list[j].append(
                    new_weight_stack[j].assign_add(
                        prolong(new_weight_stack[j+1])
                    )
                )
                with tf.control_dependencies([
                    new_weight_stack[j+1].assign(restrict(new_weight_stack[j])),
                    new_weight_stack[j].assign_add(-prolong(new_weight_stack[j+1]))
                ]):
                    restrict_ops_list[j+1].append(
                       tf.no_op() 
                    )
            #self.var_list.append(new_var)
            #self.bias_list.append(new_bias)
        #self.all_vars = self.var_list + self.bias_list
        #self.Ypr = current
        """for i in range(len(prolong_ops_list)):
            print(prolong_ops_list[i])
        
        for i in range(len(restrict_ops_list)):
            print(restrict_ops_list[i])
        quit()"""
        self.Pops = [tf.group(prolong_ops_list[i] + [tf.no_op()]) for i in range(len(prolong_ops_list))]
        self.Rops = [tf.group(restrict_ops_list[i] + [tf.no_op()]) for i in range(len(restrict_ops_list))]
        self.Y = tf.placeholder(tf.float32, [None,layers[-1]])
        self.error = tf.losses.mean_squared_error(self.Y, current) 
        #self.errors = [tf.losses.mean_squared_error(self.Y, current[i]) for i in range(len(current))]
    
    def build_optimizers(self):
        from itertools import chain
        variable_collections = [
            #list(chain(*[tf.get_collection("depth_%d" % j) for j in range(i)]))
            #for i in range(1,self.d+1)
            tf.get_collection("depth_%d" % i)
            for i in range(self.d+1)
        ]
        variable_collections.reverse()
        self.level_opts = [tf.train.RMSPropOptimizer(0.0005).minimize(self.error,
                                                              var_list =variable_collections[i]
                                                             )
                      for i in range(len(variable_collections))
                     ]
        self.var_counts = [
                np.sum([np.prod(v.get_shape().as_list()) for v in variable_collections[i]])
                for i in range(len(variable_collections))
        ]
        print(self.var_counts)
        
    def level_train(self, l, in_batch):
        if in_batch == None:
            in_batch = self.df.batch(BATCH_SIZE)
        if self.cost > self.budget:
            return in_batch
        self.cost += (self.var_counts[l]*in_batch[0].shape[0])/self.var_counts[0]
        self.sess.run(self.level_opts[l],
                 feed_dict={
                 self.input:in_batch[0],
                 self.Y:in_batch[1]
                 }
        )
        return in_batch

    def initial_train(self,k):
        pass
        for i in range(self.d, -1, -1):
            print(i)
            for j in range(k):
                self.evaluate(i)
                self.level_train(i, None)
                self.evaluate(i)

    def train(self, k, gamma, in_batch=None, level=0):
        #k = min(k, 128)
        self.evaluate(level)
        for i in range(k):
            batch = self.level_train(level, in_batch)
        if level+1 < len(self.level_opts):
            for gg in range(gamma):
                if self.cost > self.budget:
                    break
                #self.sess.run(self.Rops[level+1])    
                self.train(k, gamma, in_batch=None,level=level+1)
                #self.sess.run(self.Pops[level])
                for i in range(k):
                    batch = self.level_train(level, in_batch)
        for i in range(k):
            batch = self.level_train(level, in_batch)    
            
    def evaluate(self, level):
        self.history.append((self.cost, level, self.sess.run(self.error,
                                                feed_dict={
                                         self.input:self.df.test_set[0],
                                         self.Y:self.df.test_set[1]
                                         })
                            )
                           )
    
def exptrun(args):
    task, sm, lllam, dep = args
    task_prefix = prefix+"_"+task+"_"
    print(args)
    global tf
    import sys
    def is_non_zero_file(fpath):  
        return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
    #if is_non_zero_file('res/%s_%1d_%1d_%03d.csv' % (task_prefix, lllam, dep, sm)):
    #    return
    
    sys.stdout = open("logs/"+task_prefix+"_log_sm%03d_lam%03d_dep%03d.csv" % args[1:], 'w',1)
    if task == '2OBJ':
        df = DataFeederTwoObj(1024)
    elif task == 'MNIS':
        df = DataFeederMNIST()
    else:
        df = DataFeederOneObj(1024)
    with tf.Graph().as_default() as graph:
        model = MsAEN(
            [item for item in [1024,256,128,256,1024]],
            smoothing = sm,
            depth = dep,
            lamb = lllam,
            red = red2,
            df = df
        )
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        all_hist = []
        with tf.Session(config=config) as sess:
            sess.run(init)
            model.sess = sess
            model.initial_train(1)
            try:
                #for i in range(int(totaltrain/(sm*(lllam+1)*BATCH_SIZE))):
                while model.cost < model.budget:
                    model.train(sm, lllam)
                    print(model.history[-1])
            except KeyboardInterrupt:
                pass
            except:
                raise
        resss = "\n".join([" ".join(str(it) for it in list(item)) for item in model.history])
        f = open('res/%s_%1d_%1d_%03d.csv' % (task_prefix, lllam, dep, sm),'w')
        f.write(resss)
        f.close()
        print(resss)
        
def _init():
    global tf
    from time import sleep    

if __name__ == '__main__':
    #test = MsAEN([1024,256,128,256,1024],depth=2)
    #quit()
    prefix = "ch_mnis_grid"
    totaltrain = 256000
    sm = 1
    lllam = 2
    dep = 6
    args_list = []
    for dep in [0,1,2,3,4,5,6]:
        args_list.append(("MNIS", 1, 3, dep))
    #exptrun(('MNIS', sm, lllam, dep))
    #quit()
    #for tsk in ['2OBJ', '1OBJ','MNIS']:#
    #    for sm in [1,2,4,8,16,32,64,128]:
    #        for lllam in [0,1,2,3]:
    #            for dep in [0,1,2,3,4,5,6]:
    #                    args_list.append((tsk, sm, lllam, dep))
    from random import shuffle
    shuffle(args_list)
    #args_list = args_list[:15]
    #args_list.append((1,3,5))
    #args_list = [
    # ('2OBJ', 1,0,0),   
    # ('MNIS', 1,0,0),   
    # ('1OBJ', 1,0,0)
    #]
    from multiprocessing import Pool
    import contextlib
    num_pool_workers = min(len(args_list),7)
    try:
        with contextlib.closing(Pool(num_pool_workers, initializer = _init,maxtasksperchild=1)) as p:
            val = p.map_async(exptrun, args_list)
            results = val.get()
    except KeyboardInterrupt:
        print("\n".join([str(item) for item in results]))
    except:
        raise
