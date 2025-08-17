'''
Author: Hugo
Date: 2025-08-13 13:52:54
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-08-13 13:54:00
Description: 
'''
from .kronos import KronosTokenizer, Kronos, KronosPredictor

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError