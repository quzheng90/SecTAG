import os
HASH_PATH = './hijack_data'
TRAIN_PATH = os.path.join(HASH_PATH, 'hashtag_matching_1v1.csv')
TEST_PATH = os.path.join(HASH_PATH, 'hashtag_matching_1v1.csv')
SAVE_PATH = './save'
LABEL_DICT = {
    'a': {'OFF': 0, 'NOT': 1},
    'b': {'TIN': 0, 'UNT': 1, 'NULL': 2},
    'c': {'IND': 0, 'GRP': 1, 'OTH': 2, 'NULL': 3}
}