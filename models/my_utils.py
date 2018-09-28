#!/usr/bin/env python

import sys
import csv
import torch


# IO
ver = sys.version_info
if ver > (3, 0):
    opts_read = {'encoding': 'utf-8'}
else:
    opts_read = {}


def read_csv(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        csv_reader = csv.reader(opdrf)
        data = [term for term in csv_reader]
        return data


# Save/load
def load_params(fp, device_id, all_ids=[0, 1, 2, 3, 'cpu']):
    if device_id is 'cpu':
        params = torch.load(
            fp,
            map_location=lambda storage, loc: storage)
    else:
        params = torch.load(
            fp,
            map_location={
                'cuda:{}'.format(gid): 'cuda:{}'.format(device_id)
                for gid in all_ids})
    return params


def load_model(fp, network, device_id=0):
    obj = load_params(fp, device_id)
    model_state_dict = obj['state_dict.model']

    network.load_state_dict(model_state_dict)
