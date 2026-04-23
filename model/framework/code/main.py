import os
import sys
import json
import numpy as np
import onnxruntime as ort
from ersilia_pack_utils.core import read_smiles, write_out

root = os.path.dirname(os.path.abspath(__file__))
from utils import convert_smiles_to_fp
input_file = sys.argv[1]
output_file = sys.argv[2]

ckpt_path = os.path.join(root, "..", "..", "checkpoints", "clamp_clip")
hp_path = os.path.join(ckpt_path, "hp.json")
onnx_path = os.path.join(ckpt_path, "compound_encoder.onnx")

with open(hp_path, "r") as f:
    hp = json.load(f)

compound_mode = hp["compound_mode"]

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

shape = sess.get_inputs()[0].shape
fp_size = shape[1] if isinstance(shape, (list, tuple)) and len(shape) == 2 and isinstance(shape[1], int) else 8192

_, smiles_list = read_smiles(input_file)

fp = convert_smiles_to_fp(smiles_list, which=compound_mode, fp_size=fp_size, njobs=1).astype(np.float32)
outputs = sess.run([out_name], {in_name: fp})[0]

num_dims = outputs.shape[1]
header = [f"feat_{str(i).zfill(3)}" for i in range(num_dims)]
write_out(outputs, header, output_file, np.float32)
