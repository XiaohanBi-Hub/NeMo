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
```powershell
  |--- README.md                        :  the user guidance
  |--- data/                            :  the experimental data
  |--- src/                             :  the source code of our work
       |--- configs.py                  :  setting the path
       |--- modular_trainer.py          :  training modular CNN models
       |--- modularizer.py              :  modularizing trained modular CNN models and then reusing modules on sub-tasks
       |--- standard_trainer.py         :  training CNN models using the standard training method 
       |--- ...
       |--- models/                    
            |--- utils_v2.py            :  the implementation of mask generator 
            |--- vgg.py                 :  the standard vgg16 model
            |--- vgg_masked.py          :  the modular vgg16 model, i.e., the standard vgg16 model with mask generators
            |--- ...
       |--- modules_arch/
            |--- vgg_module_v2.py       :  the vgg16 module which retains only relevant kernels and removes mask generators.
            |--- ...
       |--- exp_cnnsplitter_reusing/
            |--- reuse_modules.py       :  reusing modules published by CNNSplitter on sub-tasks
            |--- calculate_cohesion.py  :  calculating the cohesion of modules
            |--- ...                    :  published by CNNSplitter
       |--- ...
```

<br>

## Replication of experimental results
### Downloading experimental data
The following sections describe how to reproduce the experimental results in our paper. 
1. We provide the resulting models trained by standard training and modular models trained by modular training<br>
One can download `data/` from [here](https://mega.nz/file/1T8ExJrL#uUr2Jh-j1NN0m575mojKDPiDvn0aZVw_tRIeq9GbhXE) and then move it to `MwT/`.<br>
The datasets will be downloaded automatically by PyTorch when running our project. 
2. Modify `self.root_dir` in `src/configs.py`.

### Modular training, modularizing, and module reuse
1. Training a modular VGG16 model.
```commandline
python modular_trainer.py --model vgg16 --dataset cifar10 --lr_model 0.05 --alpha 0.5 --beta 1.5 --batch_size 128
```

2. Modularizing the modular VGG16 model and reusing the resulting modules on a sub-task containing "class 0" and "class 1".
```commandline
python modularizer.py --model vgg16 --dataset cifar10 --lr_model 0.05 --alpha 0.5 --beta 1.5 --batch_size 128 --target_classes 0 1
```

### Standard training
1. Training a VGG16 model
```commandline
python standard_trainer.py --model vgg16 --dataset cifar10 --lr_model 0.05 --batch_size 128
```

### Reusing modules from CNNSplitter
1. Downloading the published modules at CNNSplitter's project webpage.
2. Modifying `root_dir` in `src/exp_cnnsplitter_reusing/global_configure.py`
3. Modifying `dataset_dir` in `src/exp_cnnsplitter_reusing/reuse_modules.py`
4. Reusing SimCNN-CIFAR10's modules on a sub-task containing "class 0" and "class 1"
```commandline
python reuse_modules.py --model simcnn --dataset cifar10 --target_classes 0 1
```
5. Calculating the cohesion of modules
```commandline
python calculate_cohesion.py --model simcnn --dataset cifar10
```

## Supplementary experimental results
### Discussion of the effect of threshold on modularizing the modular ResNet18-CIFAR10 model.
The value of threshold directly affects the results of modularizing and module reuse. 
As shown in the figure below, as the threshold increases from 0.1 to 0.9, the kernel retention rate of the modules gradually decreases, from 37.36% to 24.74%. 
A larger threshold makes each module tend to retain convolutional kernels that are required by all samples of the corresponding category, leading to an increase in cohesion from 0.8572 to 0.9437 and a decrease in coupling from 0.3594 to 0.2412. 



### Discussion of the effect of threshold on reusing the ResNet18-CIFAR10's modules.
Regarding the effect on module reuse, the figure below presents the performance of the modules in terms of kernel retention rate and accuracy on the 3-class classification sub-task. 
As the threshold increases, the KRR of the module decreases, from 72.57% to 50.51%. 
Nonetheless, the decrease of KRR has a negligible impact on the accuracy of the module, which only drops from 97.77% to 97.23%. 
The experimental results also demonstrate that our default settings are appropriate.

