# FedOnce
One-Shot Vertical Federated Learning Algorithm

## Code Structure
This project contains 4 folders:
* model
* privacy
* torchdp (Cloned from facebook/pytorch-dp, now facebook/opacus)
* utils      

## Requirements
This project requires __conda 4.11.0__. The required environment is included in ```environment.yml```.
Please run 
```
$ conda env create --name fedonce --file=environments.yml
$ conda activate fedonce
```

## Reproduce the experimental results
Each script is corresponding to a dataset in our experiments
* gisette: gisette
* covtype: covtype
* phishing: phishing
* uj: UJIIndoorLoc
* superconduct: Superconduct
* mnist: MNIST
* kmnist: KMNIST
* fashion_mnist: Fashion-MNIST
* wide: NUS-WIDE
* movielens: MovieLens

### Experiment 1: Communication efficiency of FedOnce-L0
Command format:
```python3 run_<dataset_name>.py```
For example: 
```python3 run_mnist.py```
The script will output the accuracy/RMSE for FedOnce-L0 and baselines (Experiment 5 will run after Experiment 1).
The communication cost in each round will be recorded under ```logs/```.
Then, we use ```utils/output_loader.py``` to help load information from logs.

### Experiment 2: Performance of FedOnce-L0
Command format:
```python3 run_<dataset_name>.py```
For example: 
```python3 run_mnist.py```
The script will output the accuracy/RMSE for FedOnce-L0 and baselines.

### Experiment 3: Performance of FedOnce-L1
Command format:
```python3 run_dp_<dataset_name>.py```
For example: 
```python3 run_dp_mnist.py```
The script will output the accuracy/RMSE for FedOnce-L1 and baselines. 

### Experiment 4: Performance of FedOnce-L0 on biased datasets
Command format:
```python3 run_bias_<dataset_name>.py```
For example: 
```python3 run_bias_phishing.py```
The script will output the accuracy/RMSE for FedOnce-L1 and baselines. 

### Experiment 5: Communication efficiency of FedOnce-L0 (more rounds)
Command format:
```python3 run_<dataset_name>.py```
For example: 
```python3 run_mnist.py```
The script will also output the accuracy/RMSE for FedOnce-L0 (more rounds) after finishing Experiment 1.
The communication cost in each round will be recorded under ```logs/```.
Then, we use ```utils/output_loader.py``` to help load information from logs.


### Experiment 6: Scalability of FedOnce-L0
Command:
```python3 run_multi_gisette.py```

### Experiment 7: Linear-Combine
Command:
```python3 run_linear.py```