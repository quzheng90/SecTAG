import os
HASH_PATH = './hijack_data'
TRAIN_PATH = os.path.join(HASH_PATH, 'hashtag_train_1v1.csv')
TEST_PATH = os.path.join(HASH_PATH, 'hashtag_test_1v1.csv')
SAVE_PATH = './save'
LABEL_DICT = {
    'a': {'pos': 0, 'neg': 1},
    'b': {'pos': 0, 'neg': 1},
    'c': {'pos': 0, 'neg': 1}
}
