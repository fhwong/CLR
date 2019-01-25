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
    https://forums.fast.ai/t/research-paper-recommendations/13768/27  :
      I am sorry for the confusion on the 1cycle policy. It is one cycle but I let the cycle 
      end a little bit before the end of training (and keep the learning rate constant at 
      the smallest value) to allow the weights to settle into the local minima.
    
    
    #lr vs iteration
    #
    #   ----             <- max_lr
    #  -    -
    # -      -
    #-        -          <- base_lr
    #          -
    #           --------  <- settle_lr
    #   
    #< a><b ><c ><  d  > 
    #a:step_up , b:step_max, c:step_down, d:step_settle
    
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    
    
    # Example
        ```python
            clr = OneCyclicLR(base_lr=0.001, max_lr=0.006, settle_lr = 0.00001,
                                step_size=2000., mode='triangular', scale_mode = 'cycle')
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

        step_settle: number of training iterations used in 
            settle_lr in the down step: the down step_size
            is step_size - step_settle
            
        settle_lr: learning rate after certain iteration/cycle,
            should be 'several orders less than base_lr, 
        settle_cycle: after this cycle , lr is settle_lr(const)
        settle_itertion: after this iteration, lr is settle_lr(const)
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, settle_lr = 0.00001,
                step_size=2000., step_max = 0, step_settle = 0, step_down = None ):
        super(OneCyclicLR, self).__init__()

        if step_size <=0:
            raise Exception("step_size <=0")
        if step_max < 0 :
            raise Exception("step_max < 0")
        if step_settle > step_size :
            raise Exception("step_settle > step_size")
        if step_settle < 0 :
            raise Exception ("step_settle < 0")
            
           
            
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.settle_lr = settle_lr 
        
        self.step_size = step_size
        self.step_up = step_size
        self.step_max = step_max
        if step_down is None:
            self.step_down = step_size - step_settle
        else:
            self.step_down = step_down
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_settle_lr = None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_settle_lr != None:
            self.settle_lr = new_settle_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size)) #number of cycle
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1) 
        
        if self.clr_iterations <= self.step_up:
            return self.base_lr + (self.max_lr-self.base_lr) * (self.clr_iterations/self.step_up)
            
        elif self.clr_iterations <= self.step_up + self.step_max:
            return self.max_lr
            
        elif self.clr_iterations <= self.step_up + self.step_max + self.step_down:
            x = np.abs((self.clr_iterations - self.step_up - self.step_max)/self.step_down -1)
            return self.settle_lr + (self.max_lr-self.settle_lr) * np.maximum(0, x)
            
        else:
            #all extra iterations will be at settle_lr
            return self.settle_lr
        
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

def get_step_size(batch_size, ndata, epoch_per_step =6):
    '''
    epoch_per_step can be float
    #one cycle is step_size *2 
    '''
    iteration_per_epoch = ndata/batch_size
    step_size  = int(ndata/batch_size * epoch_per_step) #epoch_per_step could be float
    
    return int(step_size)

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

    
    #clr_triangular1= OneCyclicLR(settle_lr= 0.00001, scale_mode='cycle',step_size = 2000)
    
    if 1==2:
        #1) give step size as argument
        clr_triangular1= OneCyclicLR(base_lr=0.00001, max_lr=0.00006, settle_lr = 0.0000001,
                    step_size=2000., step_max = 0, step_settle = 400 )
        step_size = 2000; ndata = len(Y); batch_size = 200
        epoch_per_cycle = step_size *2 / (ndata/batch_size ) #4
        print('epoch_per_cycle', epoch_per_cycle)

    if 1==2:
        #2) given total number of epochs get step size
        epochs = 4
        step_size = get_step_size(batch_size, ndata, epoch_per_step = epochs/2.)
        clr_triangular1= OneCyclicLR(base_lr=0.00001, max_lr=0.00006, settle_lr = 0.0000001,
                    step_size=step_size, step_max = 0, step_settle = int(step_size*0.1) )
    

    if 1==1:
        #3) use epoch as base
        step_per_epoch = len(Y)/batch_size
        clr_triangular1= OneCyclicLR(base_lr=0.00001, max_lr=0.00006, settle_lr = 0.0000001,
                    step_size=step_per_epoch*2, step_max = step_per_epoch*1,
                    step_down = step_per_epoch*4.5, step_settle = step_per_epoch*0.5 )
                
                
    model.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, batch_size=200, epochs= 8 , callbacks=[clr_triangular1], verbose=0)

    plt.plot(clr_triangular1.history['iterations'], clr_triangular1.history['lr'])
    plt.title('lr vs iteration')
    print('last 10 lr')
    print(clr_triangular1.history['lr'][-10:])
    print('first 10 lr')
    print(clr_triangular1.history['lr'][0:10])

    
    
    
