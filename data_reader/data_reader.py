import pandas as pd

def read_data_source_target(file_name_source, file_name_target):
    file_source = open(file_name_source, 'r', encoding='utf8')
    file_target = open(file_name_target, 'r', encoding='utf8')

    source = file_source.readlines()
    target = file_target.readlines()

    if len(source) != len(target):
        raise ValueError(
            "The length of the source file should be equal to target file"
        )
    source_target_pair = [["predict", source[i], target[i]] for i in range(len(source))] # "" for "prefix" used in t5_util.py
    data_df = pd.DataFrame(source_target_pair, columns=["prefix", "input_text", "target_text"])
    return data_df
