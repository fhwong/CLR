# Cyclical Learning Rate (CLR) and 1Cycle policy

Forked from https://github.com/bckenstler/CLR

Added 1Cycle policy from https://arxiv.org/abs/1708.07120

It will use a very small learning rate after one cycle.

```
lr vs iteration

   ----             <- max_lr
  -    -
 -      -
-        -          <- base_lr
          -
           --------  <- settle_lr
   
< a><b ><c ><  d  > 
a:step_up , b:step_max, c:step_down, d:step_settle
```
   
