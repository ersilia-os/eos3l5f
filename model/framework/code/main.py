# imports
import os
import csv
import sys
import torch
import clamp
import numpy as np
import torch.serialization as ts

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

ts.add_safe_globals([np.core.multiarray.scalar])

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(root, "..", "..", "checkpoints", "clamp_clip")

model = clamp.CLAMP(path_dir = ckpt_path, device='cpu', pretrained=False)
model.eval()

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

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
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    for row in outputs:
        writer.writerow(row.tolist())
