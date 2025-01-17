import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, XLMRobertaTokenizer
import json

LABEL_MAPPING = {
    "description.general_and_supplementary": 0,
    "description.general_but_not_supplementary": 1,
    "description.supplementary_but_not_general": 2,
    "description.product": 3,
    "description.client_or_partner": 4,
    "description.person": 5,
    "description.place": 6,
    "not_description.technical_info": 7,
    # "not_description.user_interactions": 8,
    # "not_description.news_and_announcements": 9,
    # "not_description.other": 10
}

num_categories = len(LABEL_MAPPING)

#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
rTokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


def get_split_by_words(text, interval, num_pieces):
    text_split = text.split()
    num_word_text = len(text_split)
    div = num_word_text // interval
    text_pieces = []
    if num_word_text <= interval:
        text_piece = text_split
        text_piece = (' ').join(text_piece)
        text_pieces.append(text_piece)
    if num_word_text > interval:
        for i in range(min(div, num_pieces-1)):
            text_piece = text_split[i*interval: (i+1)*interval]
            text_piece = (' ').join(text_piece)
            text_pieces.append(text_piece)
        if num_word_text <= interval*num_pieces:
            last_piece = text_split[(i+1)*interval:]
            last_piece = (' ').join(last_piece)
            text_pieces.append(last_piece)
        else:
            last_piece = text_split[-interval:]
            last_piece = (' ').join(last_piece)
            text_pieces.append(last_piece)
    if len(text_pieces) < num_pieces:
        text_pieces = text_pieces + [''] * (num_pieces - len(text_pieces))
    return text_pieces


def bert_tokenizer(df):
    split = True
    texts_token = []
    for text, url, company_name in tqdm(zip(df['text'], df['url'], df['company_name']), total=len(df)):
        if split:
            text_reform = get_split_by_words(text=text, interval=400, num_pieces=5)
            input_elements = [url, company_name] + text_reform
        else:
            input_elements = [text] # single input
        tok = tokenizer(input_elements, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        texts_token.append(tok)
    return texts_token


def xlmroberta_tokenizer(text):
    split = False
    
    if split:
        text_reform = get_split_by_words(text=text, interval=400, num_pieces=3)
        text_input = text_reform
        tok = rTokenizer.batch_encode_plus(text_input,
                                        add_special_tokens=True,
                                        max_length=512,
                                        padding='max_length',
                                        return_token_type_ids=True,
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='pt')
    else:
        text_input = text
        tok = rTokenizer.encode_plus(text_input,
                                    None,
                                    add_special_tokens=True,
                                    max_length=512,
                                    padding='max_length',
                                    return_token_type_ids=True,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
    
    return tok


def tokenize_training_data(input_training_data, output_path):
    df = pd.DataFrame([])
    
    token_list = []
    company_name_list = []
    mapped_labels_list = []
    page_urls_list = []
    doc_id_list = []
    num_trianing_data = sum(1 for _ in open(input_training_data))
    with open(input_training_data, 'r') as fr:
        for line in tqdm(fr, total=num_trianing_data):
            line_dict = json.loads(line)

            page_url = line_dict['page_url']
            page_urls_list.append(page_url)

            doc_id = line_dict['doc_id']
            doc_id_list.append(doc_id)

            company_name = line_dict['company_name']
            company_name_list.append(company_name)

            text = line_dict['text']
            token = xlmroberta_tokenizer(text)
            token_no_tensor = {}
            for key, value in token.items():
                token_no_tensor[key] = value.tolist()

            token_list.append(token_no_tensor)        
            labels = line_dict['gpt_labels']
            mapped_labels = [0] * num_categories
            for label in labels:
                mapped_label_index = LABEL_MAPPING[label]
                mapped_labels[mapped_label_index] = 1
            mapped_labels_list.append(mapped_labels)
    
    df['doc_id'] = doc_id_list
    df['company_name'] = company_name_list
    df['label'] = mapped_labels_list
    df['text_token'] = token_list
    processed_data = df
    print(processed_data)
    df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    input_training_data = '/home/yaxiong/training/webpage_predictor/data/gpt_annotation_data_with_id.jsonl'
    output_path = '/home/yaxiong/training/webpage_predictor/data/gpt_annotation_data_convert_token_with_id.csv'
    tokenize_training_data(input_training_data, output_path)