# Readme
## *The generalization ability of Lottery ticket hypothesis*
## Introduction
- With the development of machine learning, the performance of learning models improves a lot. Although the large models are powerful enough to deal with plenty of difficult tasks, the almost unbearable computational cost to train and infer large models on mobile or small devices has become a hot issue. In order to solve such issues, model pruning is needed, and one of the most important researches among them is the lottery ticket hypothesis (LTH). Although the lottery ticket hypothesis is a commonly used model pruning method, people lack a deep understanding of it. Our work explore the generalization ability of the "winning ticket" and find that: 1) the winning tickets always outperform randomly initialized models across different model scales, 2) the winning tickets found on one dataset or optimizer can be well trained on a new dataset or optimizer, 3) the winning tickets found with few samples could perform well on the whole dataset. Thus, we could draw two conclusions: 1) the winning tickets have good generalization ability across different model scales, different datasets, and different optimizers, 2) we could iteratively prune original models with limited samples to find good winning tickets, which could help us save a lot of computational resources.

## File structure
- **ESE539_Project_Q1_Q4.ipynb**
  - experiments to explore the Q1&Q4
    - Q1: For different model sizes, do the winning tickets always outperform randomly initialized models?
    - Q4: Whether winning tickets pruned on small samples can perform well on the whole dataset? If so, we could save a lot of computational resources on model pruning.
- **ESE539_Project_Q2_Q3.ipynb**
  - experiments to explore the Q2&Q3
    - Q2: Whether winning tickets trained on one dataset can also be effective when we train it on a new dataset?
    - Q3: Whether winning tickets trained on one optimizer can also be effective when we trian it on a new optimizer?
- **exp_result**
- **model_save**

## Function Description
- Define basic architectures (adjustable FC, LeNet-5)
- Utils (operations about dataloader and result saving)
- Prune methods (layer_wize pruning)
- Weight Related Operation (weight-mask, fix weight grads, control the update of unpruned weights)
- Main functions
  - `expIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device)`
  - `evalIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device,tickets_path)`

## Examples
- This example implement the lottery ticket hypothesis and store the accuracy and validation loss of both winning tickets and reinitialized models.
- The trained models and winning tickets are stored in `model_save` folder, the experiment results (prune rate, validation loss, accuracy) are stored in `exp_result` folder.
```
# hyperparameters initalization
n_classes = 10
n_epochs_train = 20
n_epochs_iterative_prune = 10
num_train = 60000
random_seed = 42
reinit = True # False: winning ticket, True: reinitialized model
batch_size = 16
model_type = 'lenet'
dataset_type = 'mnist'
n_layers = 2
n_neurons = 10
lr = 1e-3
pr = 20 # prune rate
momentum = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tickets_path = expIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device)
evalIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device,tickets_path)

# hyperparameters initalization
n_classes = 10
n_epochs_train = 20
n_epochs_iterative_prune = 10
num_train = 60000
random_seed = 42
reinit = False # False: winning ticket, True: reinitialized model
batch_size = 16
model_type = 'lenet'
dataset_type = 'mnist'
n_layers = 2
n_neurons = 10
lr = 1e-3
pr = 20 # prune rate
momentum = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tickets_path = expIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device)
evalIntegration(n_classes,n_epochs_train,n_epochs_iterative_prune,num_train,random_seed,reinit,batch_size,model_type,dataset_type,n_layers,n_neurons,lr,pr,momentum,device,tickets_path)

```


## Experiment Settings
- **Iterative pruning**: 20% pruning rate, 10 interations
- **Datasets**: MNIST, Fashion MNIST
- **Optimizers**: SGD, Adam (learning rate=1e-3, momentum=0.9)
- **Different sample sizes to find the winning ticket**: 100 / 1,000 / 10,000 / 60,000
- **Validation dataset**: 10,000 test samples

## Contributions
- Daiwei Chen: code the Q1&Q4 experiments, write the project report and the Readme.
- Xiao Zhong: code the Q2&Q3 experiments, write the project report

## Contacts
- daiweic@seas.upenn.edu
- xiaozong@seas.upenn.edu
