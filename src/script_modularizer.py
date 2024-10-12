import subprocess
import time
import os

batch_size = 128
script_path = "modularizer.py"
cuda_device = 1

model = "simcnn"
dataset = "svhn"
lr_model = 0.05
lr_mask = 0.05

alpha = 0.5
beta = 1.9
kn_zoom = 1

threshold = 0.9
target_classes = [0]

for i in range(1,10):
    target_classes.append(i)
    tc_str = ''.join([str(tc) for tc in target_classes])

    output_path = f"/home/bixh/Documents/NeMo/src/MwT_logs/modularizer_logs_{model}_{dataset}_txt/{dataset}_lr_{lr_model}_{lr_mask}_a{alpha}_b{beta}_bs_{batch_size}_tc{tc_str}.log"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    script_args = [
        "--batch_size",str(batch_size),
        "--model", str(model),
        "--dataset", str(dataset),
        "--lr_model", str(lr_model),
        "--lr_mask", str(lr_mask),
        "--alpha", str(alpha),       
        "--threshold", str(threshold),
        "--beta", str(beta),      
        "--kn_zoom", str(kn_zoom),     
        "--batch_size", str(batch_size),
        "--target_classes", *[str(x) for x in target_classes],
        "--cuda_device", str(cuda_device),
    ]

    command = ["python", "-u", script_path] + script_args
    start = time.time()
    with open(output_path, 'w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, text=True)

    end = time.time()
    print(f"TIME COST:{end-start}")
