import torch
import numpy as np
import pandas as pd
import json

from training import Dataset
from settings import CUDA_DEVICE_NAME
from finquest_utils.logging import logger
from sklearn.metrics import precision_recall_fscore_support



def get_thresholds_average(model, model_name, test_data, batch_size):
    model.eval()
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device(CUDA_DEVICE_NAME if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()


    with torch.no_grad():
        row_sigmoid_sum = np.zeros(8)
        size_val = 0
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
            
            for row in output:
                size_val += 1
                row_sigmoid = 1/(1 + np.exp(-row.to('cpu').detach().numpy()))
                row_sigmoid_sum += row_sigmoid
                    
        logger.info(f'average threshold is {row_sigmoid_sum/size_val}')



def get_thresholds_grid_search(model, model_name, test_data, batch_size):
    model.eval()
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device(CUDA_DEVICE_NAME if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    bias_list = [-0.4, -0.3, -0.2, -0.1, -0.0, 0.1, 0.2, 0.3, 0.4]
    #bias_list = [-0.04, -0.03, -0.02, -0.01, -0.00, 0.01, 0.02, 0.03, 0.04]
    with torch.no_grad():
        for vec_id in range(11):
            f1_record = []
            records = {'dim': vec_id}
            for bias in bias_list:
                threshold = np.ones(8)*0.5
                # [0.5, 0.1, 0.9, ]
                threshold[vec_id] = threshold[vec_id] + bias
                y_true_test, y_pred_test = [], []
                logger.info(f'start testing with threshold {threshold}')
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
                        row1 = np.greater(np.array(row.to('cpu')), threshold)
                        y_pred_test.append(row1)
                        y_true_test.append(row2.to('cpu').int())
                    
                p, r, f1, s = precision_recall_fscore_support(
                    np.array(y_true_test),
                    np.array(y_pred_test),
                    average='micro')
        
                logger.info(f'f1 score is {f1}')
                f1_record.append(f1)
            max_id = f1_record.index(max(f1_record))
            best_threshold = 0.5 + bias_list[max_id]
            logger.info(f'best threshold for {vec_id} dimension is {best_threshold}')
            records['f1_list'] = f1_record
            records['best_threshold'] = best_threshold
            with open('./search_threshold.jsonl', 'a') as fw:
                json.dump(records, fw)
                fw.write('\n')


def test(model, model_name, test_data, batch_size):
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
                row1 = row.ge(0.5)
                y_pred_test.append(row1.to('cpu').int())
                y_true_test.append(row2.to('cpu').int())
        
        p, r, f1, s = precision_recall_fscore_support(
                    np.array(y_true_test),
                    np.array(y_pred_test),
                    average='micro')
        
        print(f1)



if __name__ == "__main__":
    input_training_data = "./data/gpt_annotation_data_convert_token.csv"
    training_df = pd.read_csv(input_training_data)

    test_data = training_df.sample(frac=1, random_state=105)
    batch_size = 5
    model = torch.load('./model/xmlrobertabase_gpt_annotation/checkpoint_20_12_2024__13_08_46_11.pt')
    model_name = 'xlm-roberta-base'

    #get_thresholds_grid_search(model, model_name, test_data, batch_size)
    get_thresholds_average(model, model_name, test_data, batch_size)