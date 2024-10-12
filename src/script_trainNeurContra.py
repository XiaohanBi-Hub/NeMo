import subprocess
import time
import os

batch_size = 128
script_path = "modular_trainer_Neur_contra.py"

model = "deit_s"
dataset = "svhn"
lr_model = 0.05
lr_mask = 0.05
alpha_list = [0.08]
temperature = 0.2

cuda_device = 1
for alpha in alpha_list:
    model_save_path =f"/home/bixh/Documents/MwT_ext/data/data/modular_trained/{model}_{dataset}_contra/lr_{lr_model}_{lr_mask}_a{alpha}_t{temperature}_bs_{batch_size}.pth"

    output_path = f"./DeiT_logs_contra/logs_{model}_{dataset}_txt/lr_{lr_model}_{lr_mask}_a{alpha}_t{temperature}_bs_{batch_size}.log"
    log_path = f"./DeiT_logs_contra/logs_{model}_{dataset}/lr_{lr_model}_{lr_mask}_a{alpha}_t{temperature}_bs_{batch_size}"

    model_save_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    script_args = [
        "--model", str(model),
        "--dataset", str(dataset),
        "--lr_model", str(lr_model),
        "--lr_mask", str(lr_mask),
        "--alpha", str(alpha),       
        "--temperature", str(temperature),          
        "--batch_size", str(batch_size),
        "--n_epochs", "200",
        "--log_dir", log_path,
        "--cuda_device", str(cuda_device),
        "--model_save_path", str(model_save_path),
        "--replace_attention",
        # '--modular_superclass',
    ]

    command = ["python", "-u", script_path] + script_args
    start = time.time()
    with open(output_path, 'w') as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, text=True)

    end = time.time()
    print(f"TIME COST:{end-start}")
