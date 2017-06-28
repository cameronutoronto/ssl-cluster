"""garbage.py
Usage:  
   garbage.py <f_data_config> [--prefix <something>] 
   garbage.py --test
   garbage.py -g
   
Arguments:
    <f_data_config>  example ''data/config/train_rev0.yaml''
    <f_model_config> example 'model/config/conv2d-3layers.yaml'
    <f_opt_config> example 'opt/config/basic.yaml'

Example:

Options:
"""

from docopt import docopt

#### 
arguments = docopt(__doc__)
print ("...Docopt... ")
print(arguments)
print ("............\n")