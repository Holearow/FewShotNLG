import os
import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser


class WikiDataset(Dataset):
    def __init__(self,
                 root_path='./data_release/',
                 domain='books',
                 mode='train',
                 data_availability=False
                 ):
        # Initialization
        self.root_path = root_path
        self.domain = domain
        self.mode = mode
        self.data_availability = data_availability

        assert self.domain in ['humans', 'books', 'songs'], \
            'Invalid domain type! Pass humans/books/songs to initialize the domain attribute.'
        assert self.mode in ['train', 'valid', 'test'], \
            'Invalid mode type! Pass train/val/test to initialize the mode attribute.'

        # get table inputs
        if data_availability:
            table_file_path = os.path.join(self.root_path, f'{self.domain}/{self.mode}.table')
            with open(table_file_path, encoding='UTF-8') as f:
                tmp_list = f.readlines()
            self.table_input = [item[:-1] for item in tmp_list]
        else:
            table_file_path = os.path.join(self.root_path, f'{self.domain}/original_data/{self.mode}.box')
            self.table_input, _ = self.process_table_file(table_file_path)

        # get reference inputs
        ref_file_path = os.path.join(self.root_path, f'{self.domain}/original_data/{self.mode}.summary')
        with open(ref_file_path, encoding='UTF-8') as f:
            tmp_list = f.readlines()
        self.ref_input = [item[:-1] for item in tmp_list]

        # get data amount
        assert len(self.table_input) == len(self.ref_input), \
            'The number of tables and that of descriptions do not match.'
        self.length = len(self.table_input)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.table_input[i], self.ref_input[i]

    def process_table_file(self, path):
        att_val_pairs = []
        table_input = []

        with open(path, encoding='UTF-8') as f:
            tmp_list = f.readlines()
        tmp_list = [item[:-1].split('\t') for item in tmp_list]

        # get seperated attribute/value pair list
        sep_att_val_pairs = []
        for item in tmp_list:
            table_pairs = []
            for pair in item:
                try:
                    attribute, value = pair.split(':')
                except ValueError:
                    j = pair.find(':')
                    attribute = pair[:j]
                    value = pair[j+1:]
                attribute = attribute[:-2]
                table_pairs.append([attribute, value])
            sep_att_val_pairs.append(table_pairs)

        # get merged attribute/value pair list and processed table input
        for item in sep_att_val_pairs:
            sep_table_input = []
            current_attribute = item[0][0]
            current_value = []

            for index, [attribute, value] in enumerate(item):
                if current_attribute == attribute:
                    current_value.append(value)
                else:
                    # get a merged attribute/value pair
                    merged_value = ' '.join(current_value)
                    att_val_pairs.append([current_attribute, merged_value])

                    # get a table input
                    sep_table_input.append(current_attribute + ' is ' + merged_value + ' .')

                    current_attribute = attribute
                    current_value.clear()
                    current_value.append(value)

            merged_value = ' '.join(current_value)
            att_val_pairs.append([current_attribute, merged_value])
            sep_table_input.append(current_attribute + ' is ' + merged_value + ' .')
            table_input.append(' '.join(sep_table_input))

        # save the file
        my_table_file_path = os.path.join(self.root_path, f'{self.domain}/{self.mode}.table')
        with open(my_table_file_path, 'w', encoding='UTF-8') as f:
            for item in table_input:
                f.write(item + '\n')

        return table_input, att_val_pairs


'''_test = WikiDataset(domain='humans', data_availability=False)
print(_test.__len__())
while True:
    i = int(input())
    if i > _test.__len__():
        break
    _test.__getitem__(i)'''