# imports
import os
import csv
import sys
import torch
import clamp
import numpy as np
import struct
import json
import torch.serialization as ts
from ersilia_pack_utils.core import read_smiles, write_out

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

ts.add_safe_globals([np.core.multiarray.scalar])

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(root, "..", "..", "checkpoints", "clamp_clip")

model = clamp.CLAMP(path_dir = ckpt_path, device='cpu', pretrained=False)
model.eval()

# read input
_, smiles_list = read_smiles(input_file)

with torch.no_grad():
    mol_embeddings = model.encode_smiles(smiles_list)
    outputs = mol_embeddings.cpu().numpy()

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

num_dims = outputs.shape[1]
header = [f"feat_{str(i).zfill(3)}" for i in range(num_dims)]

# write output in a .csv file
write_out(outputs, header, output_file, np.float32)
