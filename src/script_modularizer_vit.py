import subprocess
import time
import os

batch_size = 128
script_path = "modularizer_vit.py"
cuda_device = 0

model = "vit_s"
dataset = "cifar10"
lr_model = 0.05
lr_mask = 0.05

alpha_list = [0.06, 0.15]
temperature = 0.2

threshold = 0.9
target_classes = [0]
for alpha in alpha_list:
    for i in range(1,10):
        target_classes.append(i)
        tc_str = ''.join([str(tc) for tc in target_classes])

        output_path = f"./ViT_logs_contra/modularizer_logs_{model}_{dataset}_txt/lr_{lr_model}_{lr_mask}_a{alpha}_t{temperature}_bs_{batch_size}_tc{tc_str}_noEnc0.log"
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
            # "--beta", str(beta),
            "--temperature", str(temperature),
            "--contra",
            "--threshold", str(threshold),
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
