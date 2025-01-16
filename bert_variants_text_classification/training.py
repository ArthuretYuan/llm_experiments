import json
import os
import random
import shutil
from datetime import datetime
from urllib.parse import urlparse
import ast
import mlflow
import numpy as np
import pandas as pd
import math
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
import transformers
from finquest_utils.logging import logger
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from skmultilearn.model_selection import IterativeStratification
from settings import CUDA_DEVICE_NAME

#CUDA_VISIBLE_DEVICES=6

# For reproducibility
SEED_INT = 123
os.environ['PYTHONHASHSEED'] = str(SEED_INT)
np.random.seed(SEED_INT)
random.seed(SEED_INT)
torch.manual_seed(SEED_INT)
torch.cuda.manual_seed_all(SEED_INT)
TIMESTAMP = str(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [ast.literal_eval(label) for label in df['label']]
        self.doc_ids = [doc_id for doc_id in df['doc_id']]
        
        self.texts = []
        for tok in df['text_token']:
            tok_dict = ast.literal_eval(tok)
            tok_dict_new = {}
            for key, value in tok_dict.items():
                tok_dict_new[key] = torch.tensor(value)
            self.texts.append(tok_dict_new)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def train(model, model_name, train_data, val_data, weight, learning_rate, epochs, batch_size, target_names, log_mlflow, kpi_save_path):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=10, drop_last=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device(CUDA_DEVICE_NAME if use_cuda else "cpu")
    
    # Set the class weight if needed
    class_weights = torch.FloatTensor(weight).to(device)
    
    # Define criterion
    
    # NOTE: BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
    # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as.
    # pos_weight is for BCEWithLogitsLoss
    # ref: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) 
    
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Deinfine optimizer
    optimizer = Adam(model.parameters(), lr= learning_rate)

    val_accuracy_best_checkpoint = -1
    val_accuracy_values = []
    epoch_best_checkpoint = -1
    file_name_best_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt_folder")
    if os.path.isdir(file_name_best_path):
        shutil.rmtree(file_name_best_path)
    file_name_best_checkpoint = ''
    file_names_saved_checkpoints = []
    recent_model_degradations = []

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    for epoch_num in range(epochs):
        model.train()
        logger.info('start epoch {}, number of iteration is {}'.format(epoch_num + 1, len(train_dataloader)))
        total_loss_train = 0
        y_true_train, y_pred_train = [], []

        
        # NOTE: training
        training_iter = 0
        for train_input, train_label in tqdm(train_dataloader):
            training_iter += 1
            train_label = train_label.to(device)
            if 'xlm' in model_name:
                mask = torch.squeeze(train_input['attention_mask']).to(device)
                input_id = torch.squeeze(train_input['input_ids']).to(device)
            else:
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].to(device)
            
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.float())
            total_loss_train += batch_loss.item()
            
            # check average loss every 1000 iter
            if training_iter%1000 == 0:
                logger.info(f'ave loss: {total_loss_train/training_iter}')
            
            for row, row2 in zip(output,train_label):
                # if we use BCEWithLogitsLoss, the output is without sigmoid, we need to apply sigmoid here
                row_sigmoid = 1/(1 + np.exp(-row.to('cpu').detach().numpy()))
                threshold = np.ones(8)*0.5
                row1 = np.greater(row_sigmoid, threshold)
                row2 = np.array(row2.to('cpu').bool())
                y_pred_train.append(row1.astype(int))
                y_true_train.append(row2.astype(int))
            
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        report = classification_report(np.array(y_true_train),
                                       np.array(y_pred_train),
                                       output_dict=False,
                                       target_names= target_names)
        with open(kpi_save_path, 'a') as f:
            f.write(f'-------------training epoch {epoch_num + 1}-------------\n')
            f.write(f'{report}\n')

        
        
        
        
        # NOTE: validation
        model.eval()
        with torch.no_grad():
            y_true_val, y_pred_val = [], []
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                if 'xlm' in model_name: 
                    mask = torch.squeeze(val_input['attention_mask']).to(device)
                    input_id = torch.squeeze(val_input['input_ids']).to(device)
                else:
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label.float())
                for row, row2 in zip(output,val_label):
                    row_sigmoid = 1/(1 + np.exp(-row.to('cpu').detach().numpy()))
                    threshold = np.ones(8)*0.5
                    row1 = np.greater(row_sigmoid, threshold)
                    row2 = np.array(row2.to('cpu').bool())
                    y_pred_val.append(row1.astype(int))
                    y_true_val.append(row2.astype(int))
        
            report = classification_report(
                np.array(y_true_val),
                np.array(y_pred_val),
                output_dict=False,
                target_names=target_names)
        
        with open(kpi_save_path, 'a') as f:
            f.write(f'-------------validation epoch {epoch_num + 1}-------------\n')
            f.write(f'{report}\n')


        # early termination condition 
        p, r, f1, s = precision_recall_fscore_support(np.array(y_true_val),
                                    np.array(y_pred_val),
                                    average='micro')
        
        validation_accuracy = f1
        val_accuracy_values.append(validation_accuracy)
        epoch_stats_str = f'validation_f1: {f1}'
        
        CONVERGENCE_THRESHOLD = 0.001
        MAX_MODEL_DEGRADATIONS = 3
        MIN_NUM_EPOCHES = 4

        if validation_accuracy > val_accuracy_best_checkpoint:
            val_accuracy_best_checkpoint = validation_accuracy
            epoch_best_checkpoint = epoch_num
            if not os.path.exists(file_name_best_path):
                os.makedirs(file_name_best_path, exist_ok=True)
            file_name_best_checkpoint = os.path.join(file_name_best_path, f"checkpoint_{TIMESTAMP}_{epoch_num}.pt")
            torch.save(model, file_name_best_checkpoint)
            file_names_saved_checkpoints.append(file_name_best_checkpoint)

        if len(val_accuracy_values) >= 2:
            try:
                current_improvement = (val_accuracy_values[-1] - val_accuracy_values[-2]) / val_accuracy_values[-2]
            except Exception:
                current_improvement = 0
            if current_improvement >= 0:
                epoch_stats_str += f' (+{100*current_improvement:.2f}%)'
            else:
                epoch_stats_str += f' ({100*current_improvement:.2f}%)'
            logger.info(epoch_stats_str)
            if current_improvement < 0:
                recent_model_degradations.append(epoch_num)
                if len(recent_model_degradations) >= 2:
                    logger.info('Number of recent degradations currently taken into account for early stopping: {} (epochs {})'.format(
                        len(recent_model_degradations),
                        f"{', '.join(str(rd + 1) for rd in recent_model_degradations[:-1])} and {recent_model_degradations[-1] + 1}"
                    ))
            elif recent_model_degradations and epoch_num - recent_model_degradations[-1] > MAX_MODEL_DEGRADATIONS:
                # "Forget" oldest degradation on record if MAX_MODEL_DEGRADATIONS + 1 epochs pass with no further degradations
                recent_model_degradations.pop(0)

            if len(val_accuracy_values) >= MIN_NUM_EPOCHES and current_improvement < CONVERGENCE_THRESHOLD:
                try:
                    previous_improvement = (val_accuracy_values[-2] - val_accuracy_values[-3]) / val_accuracy_values[-3]
                except Exception:
                    previous_improvement = 0
                if previous_improvement < CONVERGENCE_THRESHOLD:
                    logger.info(f'Improvements in last two epochs have been below {100 * CONVERGENCE_THRESHOLD}%. Stopping training...')
                    break
                elif len(recent_model_degradations) >= MAX_MODEL_DEGRADATIONS:
                    logger.info(f'Improvement in last epoch was below {100 * CONVERGENCE_THRESHOLD}% and the model has recently degraded {len(recent_model_degradations)} times. Stopping training...')
                    break

        else:
            logger.info(epoch_stats_str)
    
    if epoch_num != epoch_best_checkpoint:
        logger.info(f'Reloading best checkpoint ({epoch_best_checkpoint+1}-th epoch)...')
        model = torch.load(file_name_best_checkpoint)
    for fn in file_names_saved_checkpoints:
        if fn!=file_name_best_checkpoint:
            os.remove(fn)

    
    if log_mlflow:
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # client = mlflow.tracking.MlflowClient()
        # client.delete_registered_model(name="multipassage-BERT-description-test") # delete all previous versions
        # mlflow.log_dict(df.to_dict(), 'data.txt') # activate if need to logger data
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("tr_dataset_size", len(train_data))

        registered_model_name = 'multipassage-LSTMBERT-recurated-denoise-minorv2'
        if tracking_url_type_store != "file": # Model registry does not work with file store
            mlflow.pytorch.log_model(model, "model", registered_model_name=registered_model_name)
        else:
            mlflow.pytorch.log_model(model, "model")




def test(model, model_name, test_data, batch_size, target_names, kpi_save_path):
    out_of_sample_prob_epoch = []
    model.eval()
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device(CUDA_DEVICE_NAME if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        logger.info('start testing ...')
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            if 'xlm' in model_name: 
                mask = torch.squeeze(test_input['attention_mask']).to(device)
                input_id = torch.squeeze(test_input['input_ids']).to(device)
            else:
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].to(device)
            output = model(input_id, mask)
            
            for row, row2 in zip(output,test_label):
                row_sigmoid = 1/(1 + np.exp(-row.to('cpu').detach().numpy()))
                out_of_sample_prob_epoch.append(row_sigmoid.tolist())
                threshold = np.ones(8)*0.5
                row1 = np.greater(row_sigmoid, threshold)
                row2 = np.array(row2.to('cpu').bool())
                y_pred_test.append(row1.astype(int))
                y_true_test.append(row2.astype(int))
        
        report = classification_report(np.array(y_true_test),
                                    np.array(y_pred_test),
                                    output_dict=False,
                                    target_names=target_names)
        
    
    with open(kpi_save_path, 'a') as f:
        f.write(f'-------------testing-------------\n')
        f.write(f'{report}\n')

    with open(f'out_of_sample_prob_{TIMESTAMP}.jsonl', 'a') as f:
        for prob, true_label, doc_id in zip(out_of_sample_prob_epoch, test.labels, test.doc_ids):
            json.dump({'doc_id': doc_id, 'label': true_label, 'probability': prob}, f)
            f.write('\n')


def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def start_training(model_name, input_training_data, config_file, kpi_save_path):

    config_dict = json.load(open(config_file))
    training_df = pd.read_csv(input_training_data)

    class_count = config_dict["class_count"]    
    class_weights = config_dict["class_weights"]
    batch_size = config_dict["batch_size"]
    epochs = config_dict["epochs"]
    k_value = config_dict["k_value_for_k_fold_cv"]
    learning_rate = config_dict["learning_rate"]
    passage_count = config_dict["passage_count"] # for cancatenate models
    dropout = config_dict["dropout"]
    log_mlflow = config_dict["log_mlflow"]
    target_names = config_dict["target_names"]
    tmp_cache_dir = config_dict["tmp_cache_dir"]

    
    # load all data
    df = training_df.sample(frac=1, random_state=105)

    # REF: http://scikit.ml/api/0.1.0/api/skmultilearn.model_selection.iterative_stratification.html
    # 'IterativeStratification' will take into account when balancing sample distribution across labels
    k_fold = IterativeStratification(n_splits=k_value, order=1) 

    logger.info('Prepare data for training ...')
    x_data = [[data_str] for data_str in df['text_token']]
    y_data = np.array([eval(lbl) for lbl in df['label']])
    
    # select training mode or cv mode
    k_fold_outofsameple_prob = True
    if k_fold_outofsameple_prob:
        logger.info('Cross validation ...')
    else:
        logger.info('Start training ...')

    #df_new_order = pd.DataFrame()
    cv_round = 1
    for train_val_index, test_index in k_fold.split(x_data, y_data):
        
        # Check if the tmp folder to cache model exists``
        if os.path.exists(tmp_cache_dir):
            # Delete all files and subfolders
            shutil.rmtree(tmp_cache_dir)
            logger.info(f"Deleted all contents of folder: {tmp_cache_dir}")
        else:
            logger.info(f"Folder does not exist: {tmp_cache_dir}")
        # Recreate an empty folder if needed
        os.makedirs(tmp_cache_dir)
        logger.info(f"Create tmp cache folder for pretrained model: {tmp_cache_dir}")

        from bert_model_varieties import BertConcatMultiPassageClassifier, BertLSTMMultiPassageClassifier, DistilBertConcatMultiPassageClassifier, MultilabelXlmRobertaClassifier, MultilabelLSTMXlmRobertaClassifier, MultilabelSelfAttentionXlmRobertaClassifier
        transformers.logging.set_verbosity_error() # change the verbosity to the ERROR level
        model_mapping = {'xlm-roberta-base': MultilabelXlmRobertaClassifier(model_name="xlm-roberta-base", cache_dir=tmp_cache_dir, number_labels=class_count, dropout=dropout),
                        'concatbert': BertConcatMultiPassageClassifier(passage_count=passage_count, class_count=class_count, dropout=dropout),
                        'lstmbert': BertLSTMMultiPassageClassifier(class_count=class_count, dropout=dropout),
                        'concatdistillbert': DistilBertConcatMultiPassageClassifier(passage_count=passage_count, class_count=class_count, dropout=dropout),
                        'lstm-roberta': MultilabelLSTMXlmRobertaClassifier(model_name="xlm-roberta-base", number_labels=class_count),
                        'self-attention-roberta': MultilabelSelfAttentionXlmRobertaClassifier(doc_count=4, class_count=class_count),
                        #'custom_pretrained': torch.load('./ckpt_folder_save/checkpoint_22_04_2024__14_13_35_5.pt')
                        }


        # load pre-trained model
        model = model_mapping[model_name] # initialize model for every iteration
        #model.apply(initialize_weights) # reinitialize model weights manually (should not be applied to a pre-trained model, it will force all params to be 0)
        
        # get the train, val, test data
        df_test = df.iloc[list(test_index),]
        df_train_val = df.iloc[list(train_val_index),]
        # Calculate the index for the first 90% for training the rest 10% for validation
        split_index = int(len(df_train_val) * 0.9)
        df_train = df_train_val[:split_index]
        df_val = df_train_val[split_index:]
        logger.info(f'Training size: {len(df_train)} | Training size: {len(df_val)} | Testing size: {len(df_test)}')
        logger.info(f'Batch size: {batch_size} | Model: {model_name} | cv_round: {cv_round}')
        
        train(model = model, 
              model_name = model_name, 
              train_data = df_train, 
              val_data = df_val, 
              weight = class_weights, 
              learning_rate = learning_rate, 
              epochs = epochs, 
              batch_size = batch_size, 
              target_names = target_names, 
              log_mlflow = log_mlflow,
              kpi_save_path = kpi_save_path
              )
        test(model = model,
             model_name = model_name,
             test_data = df_test,
             batch_size = batch_size,
             target_names = target_names,
             kpi_save_path = kpi_save_path)
        cv_round += 1
        
        if not k_fold_outofsameple_prob:
            break
    
        
if __name__ == "__main__":
    model_name = 'xlm-roberta-base'
    config_file = "/home/yaxiong/training/webpage_predictor/config/config_webpage_predictor.json"
    input_training_data = "/home/yaxiong/training/webpage_predictor/data/gpt_annotation_data_convert_token_with_id.csv"
    kpi_save_path = f'./metrics_{TIMESTAMP}.txt'
    
    start_training(model_name, input_training_data, config_file, kpi_save_path)
        
