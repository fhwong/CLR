# MIT License

# Copyright (c) 2017 Bradley Kenstler

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



from keras.callbacks import *

class OneCyclicLR(Callback):

    """
    modified from  https://github.com/bckenstler/CLR
    
    from original paper https://arxiv.org/abs/1708.07120 : 
    Here we suggest a slight modification of cyclical learning rate policy for 
    super-convergence; always use one cycle that is smaller than the total number of 
    iterations/epochs and allow the learning rate to decrease several orders of magnitude 
    less than the initial learning rate for the remaining iterations.
    
    LN Smith's reply in fastai forum
    https://forums.fast.ai/t/research-paper-recommendations/13768/27
    I am sorry for the confusion on the 1cycle policy. It is one cycle but I let the cycle 
    end a little bit before the end of training (and keep the learning rate constant at 
    the smallest value) to allow the weights to settle into the local minima.
    
    however this modification is different, the cycle is still triangular but every iteration
    afterwards are const learning rate. so please use one epoch (or more) than a full cycle.
    
    #lr vs iteration
    #
    #   -             <- max_lr
    #  - -
    # -   -
    #-     -          <- base_lr
    #       
    #       --------  <- settle_lr
    #      ^last iteration of 1 cycle
          
    
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = OneCyclicLR(base_lr=0.001, max_lr=0.006, settle_lr = 0.00001,
                                step_size=2000., mode='triangular', scale_mode = 'cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
            
            
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
            
        settle_lr: learning rate after certain iteration/cycle,
            should be 'several orders less than base_lr, 
        settle_cycle: after this cycle , lr is settle_lr(const)
        settle_itertion: after this iteration, lr is settle_lr(const)
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., settle_cycle = 1, settle_iteration = 4000, settle_lr = 0.00001, scale_mode='cycle'):
        super(OneCyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.settle_cycle = settle_cycle # after this cycle , lr is settle_lr(const)
        self.settle_iteration = settle_iteration #after this iteration, lr is settle_lr(const)
        self.settle_lr = settle_lr 
        
        if True:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
        
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
        
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
        self.scale_mode = scale_mode
        
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            if cycle > self.settle_cycle:
                
                return self.settle_lr
            else:
                return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            if x > self.settle_iteration:
                return self.settle_lr
            else:
                return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

def test_OneCyclicLR():
    from keras.optimizers import Nadam
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Input
    #from oneclr_callback import *
    import matplotlib.pyplot as plt


    inp = Input(shape=(15,))                
    x = Dense(10, activation='relu')(inp)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inp, x)

    X = np.random.rand(200000,15)

    Y = np.random.randint(0,2,size=200000)

    
    clr_triangular1= OneCyclicLR(settle_lr= 0.00001, scale_mode='cycle',step_size = 2000)

    step_size = 2000; ndata = len(Y); batch_size = 200
    epoch_per_cycle = step_size *2 / (ndata/batch_size ) #4
    print('epoch_per_cycle', epoch_per_cycle)
    #use epochs =  epoch_per_cycle + 1 
    
    model.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])

    
    model.fit(X, Y, batch_size=200, epochs= 5 , callbacks=[clr_triangular1], verbose=0)

    plt.plot(clr_triangular1.history['iterations'], clr_triangular1.history['lr'])
    plt.title('lr vs iteration')
    print('last 10 lr')
    print(clr_triangular1.history['lr'][-10:])
    print('first 10 lr')
    print(clr_triangular1.history['lr'][0:10])
