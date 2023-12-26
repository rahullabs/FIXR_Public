from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from model import get_all_models

def add_inference_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=False, default='ravdess',
                        
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
                        
    parser.add_argument('--model', type=str, required=False,default='inference',
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int,default=5,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    
    parser.add_argument('--backbone', type=bool, default=False, help='Select backbone')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='0, 1, 2')
    
    parser.add_argument('--save_ckpt',type = bool, default=False, help='Save Checkpoint')
    parser.add_argument('--plot_eval',type = bool, default=False, help='Plot Scores')
    
    parser.add_argument('--usewandb',type = bool, default=False, help='Inhibit wandb logging')
    parser.add_argument('--wandb_exp', type=str, default='DER_exp', help='Wandb experiment Name')
    parser.add_argument('--wandb_project', type=str, default='DERmeetsDAN', help='Wandb project name')
    
    
    # New domain

    parser.add_argument('--domain_id', type=str, default='1',
                        help='Domain ID: Eg: 1,2,3,..')  



def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--ckpt_dir', type=str, required=False,default=None,
                        help='Checkpoint directory for GCAM')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
                        
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int,default=5,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    
    parser.add_argument('--backbone', type=bool, default=False, help='Select backbone')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='0, 1, 2')
    
    parser.add_argument('--save_ckpt',type = bool, default=False, help='Save Checkpoint')
    parser.add_argument('--plot_eval',type = bool, default=False, help='Plot Scores')
    
    parser.add_argument('--usewandb',type = bool, default=False, help='Inhibit wandb logging')
    parser.add_argument('--wandb_exp', type=str, default='DER_exp', help='Wandb experiment Name')
    parser.add_argument('--wandb_project', type=str, default='DERmeetsDAN', help='Wandb project name')
    
    
    # New domain

    parser.add_argument('--domain_id', type=str, default='1',
                        help='Domain ID: Eg: 1,2,3,..')  

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')

    parser.add_argument('--in_dir', type=str, required=False,default=None,
                        help='Input Image directory for Inference')
    parser.add_argument('--out_dir', type=str, required=False,default=None,
                        help='Out Image directory for Inference')

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
    
    
def add_ader_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--p_der', type=float, default=1,
                        help='P  for partition loss.')
    
def add_gcam_args(parser: ArgumentParser) -> None:
    parser.add_argument('--in', type=str, required=True, help='Input image folder to visualize the GCAM')
    parser.add_argument('--out', type=str, required=True, help='Output GCAM image')
    parser.add_argument('--target_class', type=int, required=False, default = None, help='Target Class: Default None')
    

    
    
