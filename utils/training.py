# Based on https://github.com/aimagelab/mammoth
import json
import torch
from utils.status import progress_bar, create_stash, progress_bar_with_acc
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from utils.misc import process_domain_id, mk_dir
from argparse import Namespace
from model.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
from datetime import datetime
import json
try:
    import wandb
except ImportError:
    wandb = None
from torchinfo import summary

from sklearn.metrics import confusion_matrix, f1_score,roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def wandb_init(args):
    date_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # naming convention
    
    wandb_name =str(args.model)+"_"+str(date_time)
    wandb.init(project=args.wandb_project, name = wandb_name, config=vars(args))
    
def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

""" ------------------------------------------------------------------------------------------------
                                        Custome Training
    ------------------------------------------------------------------------------------------------
"""
def ctrain(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print("="*100)
    print(json.dumps(vars(args),sort_keys=False, indent=4))
    print("="*100)
             
    result_dir = mk_dir("results")
    date_time = datetime.now().strftime('%Y%m%d%H%M%S')
    result_dir = mk_dir(os.path.join(result_dir, str(args.model)+"_"+date_time))
    result_data_dir = mk_dir(os.path.join(result_dir, 'metrics'))
    if args.save_ckpt:
        result_ckpt_dir = mk_dir(os.path.join(result_dir, "ckpt"))
    with open(os.path.join(result_dir,'args.json'), 'w') as fp:
        json.dump(vars(args), fp)
    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)    
    model.net.to(model.device)
    summary(model, input_size=(args.batch_size, 3, 224, 224))
    domain_id = process_domain_id(args.domain_id)
    print("Performing Domain Incremental Learning in Domains with classes {}: {}\n".format(dataset.N_CLASSES_PER_TASK, domain_id))
    if args.usewandb:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb_init(args)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)
    
    dataset_copy = get_dataset(args)
    for t in range(len(domain_id)):
        model.net.train()
        _, _ = dataset_copy.data_loader_with_did(did=str(t+1))
    print("="*100)
    print(file=sys.stderr)
    model.net.train()               
    print("Random Evaluation on one of the domain!!!")
    random_results_class, random_results_task, f1= custom_evaluate(args, 
                                                                    model, 
                                                                    dataset_copy, 
                                                                    t = domain_id[-1], 
                                                                    random_test=True)
    model.net.train()
    record_list =[]
    #training each Domain Dataset
    for id in range(len(domain_id)):
        print("Domain ID: ", domain_id[id])
        model, accuracy  = train_each(model, dataset, args, csvlogger = csv_logger, 
                                      user_id =domain_id[id], perm=False, t=id, 
                                      metric_dir=result_data_dir)
        mean_acc = np.mean(accuracy, axis=1)
        if args.csv_log:
            csv_logger.log(mean_acc)
        results.append(accuracy[0])
        results_mask_classes.append(accuracy[1])
        record_list.append(results[-1][-1])
        if domain_id[id] == domain_id[-1]:
           for idx in range(len(results[-1])):
               record_list.append(results[-1][idx])
        print("-"*30)
        if args.save_ckpt:
            model = model.to('cpu')
            torch.save(model.state_dict(), os.path.join(result_ckpt_dir, str(domain_id[id])+".pt"))
            model = model.to(model.device)



    ilogger(os.path.join(result_dir, "acc.csv"), domain_id, record_list)
    
    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        csv_logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if args.csv_log:
        csv_logger.write(vars(args))

    
    if args.usewandb:
        wandb.log({'bwt': csv_logger.bwt})
        wandb.log({'fwt': csv_logger.fwt})
        wandb.log({'forgetting': csv_logger.forgetting})
        wandb.finish()


def train_each(model, dataset, args, csvlogger, user_id, perm=False, t=0, metric_dir=None):
    result_metric_dir = metric_dir
   
    train_loader, test_loader = dataset.data_loader_with_did(did = user_id)
        
    if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
            
    user_id_no = user_id
    scheduler = dataset.get_scheduler(model, args)
    for epoch in range(model.args.n_epochs):
        for i, data in enumerate(train_loader):
            # print(i, data)
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, not_aug_inputs, logits = data
                inputs = inputs.to(model.device)

                labels = labels.clone().detach()
                # labels = torch.tensor(labels)
                not_aug_inputs = not_aug_inputs.to(model.device)
                logits = logits.to(model.device)
                loss = model.observe(inputs, labels, not_aug_inputs, logits)

                total += labels.shape[0]
                outputs_,_ = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                acur = (correct / len(train_loader)) * 100                
            else:
                inputs, labels, not_aug_inputs = data
                inputs = inputs.to(model.device)
                labels = labels.clone().detach()

                inputs = inputs.to(model.device)
                # labels = torch.tensor(labels)
                labels = labels.to( model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                loss = model.observe(inputs, labels, not_aug_inputs)
                
                total += labels.shape[0]
    
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                acur = (correct / len(train_loader)) * 100
                
            progress_bar_with_acc(i, len(train_loader), epoch, user_id, loss, acur)
            
            if args.usewandb:
                wandb.log({'train_loss': loss})
                wandb.log({'train_accuracy': acur*100})

            
        
        if scheduler is not None:
            scheduler.step()
            
        #### New addition
        if epoch % 5 == 0:
            accs = custom_evaluate(args, model, dataset, t=user_id_no, metric_dir=None, random_test=True)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy_face(mean_acc, t + 1, dataset.SETTING)
            if args.usewandb:
                wandb.log({'Validation Accuracy': round(mean_acc[0], 2)})
    

    if hasattr(model, 'end_task'):
        model.end_task(dataset)

    accs = custom_evaluate(args, model, dataset, t=user_id_no, metric_dir=result_metric_dir)
    mean_acc = np.mean(accs, axis=1)
    print_mean_accuracy_face(mean_acc, t + 1, dataset.SETTING)
    if args.usewandb:
            wandb.log({'Overall Test Accuracy': round(mean_acc[0], 2)})
    
    if args.csv_log:
        csvlogger.log(mean_acc)
    
    return model, accs



def custom_evaluate(args, model: ContinualModel, 
                    dataset: ContinualDataset, 
                    last=False, t = 0, 
                    metric_dir = None, 
                    random_test = False)-> Tuple[list, list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    compare_label = 0
    f1_temp = 0
    f1 = []
    
    # f1_target = []
    # f1_predicted = []
    # f1_scores = []
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        f1_target = []
        f1_predicted = []
        f1_scores = []
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels, _ = data
                inputs  = inputs.to(model.device)
                labels =  labels.clone().detach()
                labels = labels.to(model.device)
                # if 'class-il' not in model.COMPATIBILITY:
                #     outputs = model(inputs, k)
                # else:
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                # total += 1
                f1_predicted.append(pred.cpu())  
                f1_scores.append(outputs.data.cpu()) 
                f1_target.append(labels.cpu()) 
                f1_temp = f1_score(labels.cpu(), pred.cpu(), average='macro')   # need work on it
                
        individual_acc = correct / total * 100
        print("\nTrain on Domain 0{} - Test Accuracy on Domain 0{}: {}%".format(str(int(t)), str(int(k)+1), round(individual_acc, 2)))
              
                       
        f1.append(f1_temp)
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        
        
        f1_predicted = np.concatenate(f1_predicted)
        f1_target = np.concatenate(f1_target)
        f1_scores = np.concatenate(f1_scores)
        
        name = 'train_{}_test_acc_0{}'.format(str(int(t)), str(int(k)+1))
        if args.plot_eval and not random_test:
            metrics_eval(args, metric_dir, f1_predicted, f1_scores, f1_target, name, dataset.N_CLASSES_PER_TASK)      

    model.net.train(status)
    return accs, accs_mask_classes, f1

""" ------------------------------------------------------------------------------------------------
                                        Custom matrics evaluation
    ------------------------------------------------------------------------------------------------
"""
def metrics_eval(args, result_dir, predicted, score, target, name, no_classess ):
    f1_temp = None
    cm_temp = None
    f1_temp = f1_score(target, predicted, average='macro') 
    matrix_size = 6
    print("F1 score for {}: {}".format(name, round(f1_temp, 2)))
    cm_temp = confusion_matrix(target, predicted)
    df_cm = pd.DataFrame(cm_temp, range(matrix_size), range(matrix_size))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}) # font size
    
    cf_dir = mk_dir(os.path.join(result_dir, "conf_mat"))
    plt.savefig(os.path.join(cf_dir,name+'cm.png'))
    plt.clf()

    tpr,fpr,roc_auc = ([[]]* no_classess for _ in range(3))
    f,ax = plt.subplots()
    #generate ROC data
    lw = 2
    for i in range(no_classess):
        fpr[i], tpr[i], _ = roc_curve(target==i, score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i],tpr[i])
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.rc('legend',fontsize=8)
    plt.legend(['Class {:d}'.format(d) for d in range(no_classess)])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(name)
    roc_dir = mk_dir(os.path.join(result_dir, "roc"))
    plt.savefig(os.path.join(roc_dir, name+'roc.png'))
    plt.clf()   
 
              





    
    



