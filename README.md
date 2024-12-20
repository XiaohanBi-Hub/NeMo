# NeMo:a Neuron-Level Modularizing-While-Training Approach for Decomposing DNN Models.
## Abstract
As deep neural networks (DNNs) are increasingly integrated into modern software systems, their construction costs pose significant challenges. While model reuse can reduce training costs, indiscriminately reusing entire models can lead to substantial inference overhead. DNN modularization, inspired by software engineering, offers a solution. The "modularizing-while-training" (MwT) paradigm, which incorporates modularization during training, is more effective than post-training modularization. However, existing MwT methods focus on small-scale convolutional neural networks (CNNs) at the convolutional kernel level and struggle with diverse and large-scale models, especially Transformers.
To overcome these limitations, we propose **NeMo**, a scalable and more generalizable MwT approach. NeMo operates at the neuron level—the fundamental building block of all DNNs—ensuring applicability to Transformers and various DNN architectures. We design a contrastive learning-based modular training method with an effective composite loss function, making it scalable to large models. Experiments on two Transformer models and four CNNs across two popular classification datasets demonstrate NeMo's superiority over the state-of-the-art MwT method, with an average increase of 1.72% in module classification accuracy and a 58.10% reduction in module size. Our findings show that NeMo effectively modularizes both CNNs and large-scale Transformers, offering a promising approach for scalable and generalizable DNN modularization.
## Requirements
+ fvcore 0.1.5.post20221221<br>
+ numpy 1.23.1<br>
+ python 3.9.12<br>
+ pytorch 1.12.0<br>
+ tensorboard 2.10.1<br>
+ torchvision 0.13.0<br>
+ tqdm 4.64.0 <br>
+ GPU with CUDA support is also needed

<br>

## Structure of the directories
```shell
  |--- README.md                        :  the user guidance
  |--- data/                            :  the experimental data
  |--- src/                             :  the source code of our work
       |--- configs.py                  :  setting the path
       |--- modular_trainer.py          :  training modular CNN models
       |--- modularizer.py              :  modularizing trained modular CNN models and then reusing modules on sub-tasks
       |--- standard_trainer.py         :  training CNN models using the standard training method
       |--- script_train.py             :  script to run the training process
       |--- script_modularizer.py       :  script to run the modularizing process
       |--- ...
       |--- models/                    
            |--- utils_v2.py            :  the implementation of mask generator 
            |--- vgg.py                 :  the standard vgg16 model
            |--- vgg_masked.py          :  the modular vgg16 model, i.e., the standard vgg16 model with mask generators
            |--- ...                    :  Note that ViT and DeiT models replace linear layers and add generators directly, do not need to apply with this part
       |--- modules_arch/
            |--- vit_module_HF.py       :  the ViT module which retains only relevant neurons.
            |--- ...
       |--- ...
```

<br>

## Replication of experimental results
### Downloading experimental data
The following sections describe how to reproduce the experimental results in our paper. 
1. We provide the resulting models trained by standard training and modular models trained by modular training<br>
One can download `data/` from [here](https://pan.quark.cn/s/b05460d0d9ed) and then move it to `NeMo/`.<br>
The datasets will be downloaded automatically by PyTorch when running our project. 
2. Modify `self.root_dir` in `src/configs.py`.

### Modular training, modularizing, and module reuse
1. Training a modular ViT model with NeMo, you should first modify the file script_trainNeurContra.py, and then:
```commandline
python script_trainNeurContra.py
```

2. Training a modular VGG16 model with NeMo, you should first modify the file script_trainContra.py, and then:
```commandline
python script_trainContra.py
```

3. Modularizing the modular VGG16 model and reusing the resulting modules on a sub-task, you should first modify the file script_modularizer_contra.py, and then:
```commandline
python script_modularizer_contra.py
```

4. Modularizing the modular ViT model and reusing the resulting modules on a sub-task, you should first modify the file script_modularizer_vit.py, and then:
```commandline
python script_modularizer_vit.py
```

Note that the script continuously modularize the modules from 2 to 10 categories. If you don't want this, please modify the script.

### Standard training
1. Training a ViT-small model
```commandline
python standard_trainer_vit.py --model vit_s --dataset cifar10 --lr_model 0.05 --batch_size 128
```
2. Training a VGG16 model
```commandline
python standard_trainer.py --model vgg16 --dataset cifar10 --lr_model 0.05 --batch_size 128
```

## Supplementary experimental results
### Discussion of the effect of threshold on modularizing the modular ResNet18-CIFAR10 model.
TBC



### Discussion of the effect of threshold on reusing the ResNet18-CIFAR10's modules.
TBC

