# BOHB-template

- The template of Bayesian optimization Hyperband (BOHB)

## How to Run

```python main.py```

in 2021-10-22

* [x] tf keras version (tunecallback) completed. 
* [x] tf gradienttape version (train_iteration) completed.
* [x] pytorch version completed.


## 방법 소개

### 1. Prerequisite

```pip install ray[tune] hpbandster ConfigSpace```

### 2. How to max_t setting in scheduler

* The epoch of keras or torch is terminated based on max_t value.

Running example

```
+-----------------------+----------+-------+--------------+-----------+-----------+--------------+
| Trial name            | status   | loc   | activation   |   neuron1 |   neuron2 | optimizers   |
|-----------------------+----------+-------+--------------+-----------+-----------+--------------|
| objective_dd8c7_00000 | PENDING  |       | tanh         |        43 |        50 | adam         |
| objective_dd8c7_00001 | PENDING  |       | relu         |        63 |        45 | adam         |
| objective_dd8c7_00002 | PENDING  |       | tanh         |        63 |        52 | rmsprop      |
+-----------------------+----------+-------+--------------+-----------+-----------+--------------+
```


### 4. How to set stop condition

* It can be stopped according to the value of t in the stop item of tune.run.

For example, 

```
stop={
"mean_accuracy": 0.99, # 정확도가 0.99 이상일 경우 Terminate
"training_iteration": 1
},
```
In this case, no matter how long max_t is, when all trials are running (one.), it is considered as ends.

```
Number of trials: 3/3 (3 TERMINATED)
+-----------------------+------------+-------+--------------+-----------+-----------+--------------+----------+--------+------------------+
| Trial name            | status     | loc   | activation   |   neuron1 |   neuron2 | optimizers   |      acc |   iter |   total time (s) |
|-----------------------+------------+-------+--------------+-----------+-----------+--------------+----------+--------+------------------|
| objective_dd8c7_00000 | TERMINATED |       | tanh         |        43 |        50 | adam         | 0.938278 |      1 |          5.3518  |
| objective_dd8c7_00001 | TERMINATED |       | relu         |        63 |        45 | adam         | 0.942333 |      1 |          5.15036 |
| objective_dd8c7_00002 | TERMINATED |       | tanh         |        63 |        52 | rmsprop      | 0.94     |      1 |          4.28608 |
+-----------------------+------------+-------+--------------+-----------+-----------+--------------+----------+--------+------------------+
```


### 5. Visualization

* Records are accumulated in the specified path during execution, which can be checked with the tensorboard command.

```
tensorboard --logdir bohb_results/
```
![image](https://user-images.githubusercontent.com/38157496/136788989-352e7580-d84a-48ee-97bf-68c3e81de296.png)

![image](https://user-images.githubusercontent.com/38157496/136788728-420ef170-9b7d-4ffe-8516-8d990957cd1c.png)
