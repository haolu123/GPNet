#%%
from dataloader_heatmap import load_data
from model import *
import torch.optim as optim
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
from torch.distributed import all_reduce, ReduceOp
import numpy as np

# def parse_args():
#     parser = argparse.ArgumentParser(description='Model Training')
#     # parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
#     # parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
#     # parser.add_argument("--distributed", action="store_true", help="use monai distributed training")
#     # parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
#     # parser.add_argument("--model_save_dir", default="./model_save", type=str, help="model save directory")
#     # parser.add_argument("--result_dir", default="./result", type=str, help="result save directory")
#     # parser.add_argument("--val_interval", default=5, type=int, help="validation interval")
#     # parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
#     parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
#     # parser.add_argument('--rank_id', required=False, default=0, type=int, help='Needed to identify the node and save separate weights.')
#     # parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    
#     argv = parser.parse_args()
#     return argv

# args = parse_args()


#%% fuctions for ddp

def distributed_init():
    torch.distributed.init_process_group(
                                        backend='nccl',
                                        init_method="env://",
                                        world_size=int(os.environ['WORLD_SIZE']),
                                        rank=int(os.environ["RANK"])
                                        )
    
    torch.distributed.barrier()


def distributed_params():
    return int(os.environ['LOCAL_RANK'])


def set_device(local_rank_param, multi_gpu = True):
    """Returns the device

    Args:
        local_rank_param: Give the local_rank parameter output of distributed_params()
        multi_gpu: Defaults to True.

    Returns:
        Device: Name the output device value
    """
    
    if multi_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(local_rank_param))
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
        
    return device

def main():
    
    data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv"
    batch_size = 100
    max_epoch = 50
    feature_transform = False
    eval_interval = 2
    atention_pooling_flag = True
    outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/DLBI_CoNT/model_saves"
    gene_space_dim = 3
    LOSS_SELECT = 'CE' # 'CE' or 'NLL'
    WEIGHT_LOSS_FLAG = False
    MULTI_GPU_FLAG = False
    pre_trained = False
    lr=0.001

    if MULTI_GPU_FLAG:
        ## initializing multi-node setting
        distributed_init()
        local_rank = distributed_params() # local_rank = gpu in some cases
        ## Setting device
        device = set_device(local_rank_param = local_rank, multi_gpu = True)
        torch.cuda.set_device(device) # set the cuda device, this line doesn't included in Usman's code. But appears in MONAI tutorial
        print(
                "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
                % (torch.distributed.get_rank(), torch.distributed.get_world_size())
            )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader = load_data(file_path=data_dir, batch_size=batch_size, Multi_gpu_flag=MULTI_GPU_FLAG)

    class_num = len(number_to_label.keys())
    print("class_num:", class_num)
    samples_per_class = [feature_num[i] for i in range(len(feature_num))]
    # Calculate class weights
    total_samples = sum(samples_per_class)
    class_weights = [total_samples / samples_per_class[i] for i in range(len(samples_per_class))]

    # Normalize weights so that their sum equals the number of classes
    weight_sum = sum(class_weights)
    normalized_weights = torch.tensor([w / weight_sum * len(feature_num) for w in class_weights])
    print("Normalized Class Weights:", normalized_weights)
    #%%
    
    model = CustomCNN(class_num=class_num)
   
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    model.to(device)
    if MULTI_GPU_FLAG:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def get_loss_criterion(LOSS_SELECT, WEIGHT_LOSS_FLAG, normalized_weights):
        if LOSS_SELECT == 'CE':
            if WEIGHT_LOSS_FLAG:
                criterion = nn.CrossEntropyLoss(weight=normalized_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        elif LOSS_SELECT == 'NLL':
            if WEIGHT_LOSS_FLAG:
                criterion = nn.NLLLoss(weight=normalized_weights)
            else:
                criterion = nn.NLLLoss()
        else:
            raise ValueError("Invalid LOSS_SELECT value.")
        return criterion
    #%% train
    best_acc = 0
    for epoch in range(max_epoch):
        scheduler.step()
        confusion_matrix_all = np.zeros((class_num, class_num))
        for i , data in enumerate(train_loader, 0):
            features1_count, labels = data
            features1_count, labels = features1_count.float().to(device), labels.to(device)
            optimizer.zero_grad()
            model = model.train()
            pred = model(features1_count)
            normalized_weights = normalized_weights.to(device)
            criterion = get_loss_criterion(LOSS_SELECT, WEIGHT_LOSS_FLAG, normalized_weights)
            if LOSS_SELECT == 'NLL':
                pred = F.log_softmax(pred, dim=1)
            loss = criterion(pred, labels)
            # loss = F.nll_loss(F.log_softmax(pred, dim=1), labels)

            loss.backward()
            optimizer.step()
            pred_labels = torch.argmax(pred, dim=1)
            correct = torch.sum(pred_labels == labels)/float(batch_size)
            if MULTI_GPU_FLAG:
                if torch.distributed.get_rank() == 0:
                    print(f"[{epoch}: {i}/({len(train_loader)})] train loss: {loss.item()} accuracy: {correct.item()}")
            else:
                print(f"[{epoch}: {i}/{len(train_loader)}] train loss: {loss.item()} accuracy: {correct.item()}")
            pred_labels_np = pred_labels.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for idx_batch in range(labels_np.shape[0]):
                label_i = labels_np[idx_batch]
                pred_i = pred_labels_np[idx_batch]
                confusion_matrix_all[label_i,pred_i] += 1
        print(confusion_matrix_all)
        
        if epoch % eval_interval == 0:
            confusion_matrix_all = np.zeros((class_num, class_num))
            correct_all = 0
            total_valset = 0
            for i, data in enumerate(val_loader, 0):
                features1_count, labels = data
                features1_count, labels = features1_count.float().to(device), labels.to(device)
                model = model.eval()
                pred = model(features1_count)
                pred_labels = pred.data.max(1)[1]
                correct = torch.sum(pred_labels == labels)
                correct_all += correct.item()
                total_valset += features1_count.shape[0]
                # print("debug_total_valset{}_correct_all{}".format(total_valset, correct_all))
                print(f"[{epoch}: {i}/{len(val_loader)}] val accuracy: {correct.item()/float(batch_size)}")
                pred_labels_np = pred_labels.cpu().numpy()
                labels_np = labels.cpu().numpy()
                for idx_batch in range(labels_np.shape[0]):
                    label_i = labels_np[idx_batch]
                    pred_i = pred_labels_np[idx_batch]
                    confusion_matrix_all[label_i,pred_i] += 1
            # if MULTI_GPU_FLAG:
            #     all_reduce(correct_all, op=ReduceOp.SUM)
            #     correct_all = correct_all.cpu().numpy()/len(val_loader)
            
            # else:
            #     correct_all = correct_all.cpu().numpy()/len(val_loader)
            correct_all = correct_all/float(total_valset)
            print("final accuracy {}".format(correct_all))
            print("best accuracy {}".format(best_acc))
            
            print(confusion_matrix_all)
            
            if correct_all > best_acc:

                best_acc = correct_all
                if MULTI_GPU_FLAG:
                    if torch.distributed.get_rank() == 0:
                        torch.save(model.state_dict(), f"{outf}/cls_model_geneSpaceD_{gene_space_dim}_transfeat_{feature_transform}_attenpool_{atention_pooling_flag}_pretrain_best.pth")
                else:
                    torch.save(model.state_dict(), f"{outf}/cls_model_geneSpaceD_{gene_space_dim}_transfeat_{feature_transform}_attenpool_{atention_pooling_flag}_best.pth")

                # total_correct = 0
                # total_testset = 0
                # confusion_matrix_all_test = np.zeros((class_num, class_num))
                # for i,data in enumerate(test_loader, 0):
                #     features1_count, features2_gene_idx, labels = data
                #     features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
                #     features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
                #     model = model.eval()
                #     pred, _, _ = model(features1_count,features2_gene_idx)
                #     pred_choice = pred.data.max(1)[1]
                #     correct = torch.sum(pred_choice == labels)
                #     total_correct += correct.item()
                #     total_testset += features1_count.size()[0]

                #     pred_labels_np = pred_choice.cpu().numpy()
                #     labels_np = labels.cpu().numpy()
                #     for idx_batch in range(labels_np.shape[0]):
                #         label_i = labels_np[idx_batch]
                #         pred_i = pred_labels_np[idx_batch]
                #         confusion_matrix_all_test[label_i,pred_i] += 1
                # test_acc = total_correct / float(total_testset)
                # print("test accuracy {}".format(test_acc))
                # print(confusion_matrix_all_test)
            # print("so far best model test accuracy {}".format(test_acc))
        torch.save(model.state_dict(), f"{outf}/cls_model_geneSpaceD_{gene_space_dim}_transfeat_{feature_transform}_attenpool_{atention_pooling_flag}_{epoch}.pth")

    total_correct = 0
    total_testset = 0
    confusion_matrix_all_test = np.zeros((class_num, class_num))
    for i,data in enumerate(test_loader, 0):
        features1_count, labels = data
        features1_count, labels = features1_count.float().to(device), labels.to(device)
        model = model.eval()
        pred = model(features1_count)
        pred_choice = pred.data.max(1)[1]
        correct = torch.sum(pred_choice == labels)
        total_correct += correct.item()
        total_testset += features1_count.size()[0]

        pred_labels_np = pred_choice.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for idx_batch in range(labels_np.shape[0]):
            label_i = labels_np[idx_batch]
            pred_i = pred_labels_np[idx_batch]
            confusion_matrix_all_test[label_i,pred_i] += 1

    print("final accuracy {}".format(total_correct / float(total_testset)))
    print(confusion_matrix_all_test)

    def compute_metrics(cm):
        # Initialize variables
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        # Precision, Recall and F1 Score for each class
        precision = TP / (TP + FP+1e-6)
        recall = TP / (TP + FN+1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall+1e-6)

        # Micro averaging (considering all classes together)
        micro_precision = np.sum(TP) / (np.sum(TP) + np.sum(FP)+1e-6)
        micro_recall = np.sum(TP) / (np.sum(TP) + np.sum(FN)+1e-6)
        micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall+1e-6)

        # Macro averaging (taking the average across classes)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1_score = np.mean(f1_score)

        return {
            "Micro Precision": micro_precision,
            "Micro Recall": micro_recall,
            "Micro F1 Score": micro_f1_score,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1 Score": macro_f1_score,
            "Per Class Precision": precision.tolist(),
            "Per Class Recall": recall.tolist(),
            "Per Class F1 Score": f1_score.tolist()
        }
    metrics = compute_metrics(confusion_matrix_all_test)
    # Print the metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")
    # save metrics as json file
    import json
    with open(f"results_cnn.json", 'w') as fp:
        json.dump(metrics, fp)
    # save confusion matrix use np.save
    np.save(f"confusion_matrix_cnn.npy", confusion_matrix_all_test)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Model Training')

    parser.add_argument('--local-rank',
                        required=False,
                        default=0,
                        type=int,
                        help='Needed to identify the node and save separate weights.')


    argv = parser.parse_args()
    main()