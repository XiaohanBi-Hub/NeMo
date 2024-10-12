import subprocess
import time
import os

batch_size = 128
script_path = "modularizer_contra.py"
cuda_device = 0

model = "simcnn"
dataset = "cifar10"
lr_model = 0.05
lr_mask = 0.05

alpha = 1.3
temperature = 0.2
kn_zoom = 1

threshold = 0.9
target_classes = [0]

for i in range(1,10):
    target_classes.append(i)
    tc_str = ''.join([str(tc) for tc in target_classes])

    output_path = f"/home/bixh/Documents/NeMo/src/Contra_logs/modularizer_logs_{model}_{kn_zoom}_{dataset}_txt/{dataset}_lr_{lr_model}_{lr_mask}_a{alpha}_t{temperature}_bs_{batch_size}_tc{tc_str}.log"
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
        "--temperature", str(temperature),      
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
