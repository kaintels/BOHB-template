# BOHB-template

## 실행 방법

```python main.py```

2021-10-10 기준

* [x] tf keras 버전 (tunecallback 방식) 완료
* [x] gradienttape 버전 (train_iteration 방식) 완료
* pytorch 버전은 구현 준비중


## 방법 소개

### 1. 라이브러리 설치

ray 및 tune 설치

```pip install ray[tune]```

bohb의 경우

```pip install hpbandster ConfigSpace```


### 2. 스케쥴러의 max_t 설정

* keras나 torch의 epoch을 늘려도 max_t에 따라서 terminate된다. HB 계열일 경우 eta (reduction_factor)도 잘 고려해서 설정하자.

### 3. 샘플 수 설정

* tune.run의 샘플 수는 config에서 정한 조합 중 seed에 맞춰 설정한 수에 맞춰 선별된다.

선별된 샘플은 다음과 같이 pending 되고, running 된다.

```
+-----------------------+----------+-------+--------------+-----------+-----------+--------------+
| Trial name            | status   | loc   | activation   |   neuron1 |   neuron2 | optimizers   |
|-----------------------+----------+-------+--------------+-----------+-----------+--------------|
| objective_dd8c7_00000 | PENDING  |       | tanh         |        43 |        50 | adam         |
| objective_dd8c7_00001 | PENDING  |       | relu         |        63 |        45 | adam         |
| objective_dd8c7_00002 | PENDING  |       | tanh         |        63 |        52 | rmsprop      |
+-----------------------+----------+-------+--------------+-----------+-----------+--------------+
```


### 4. stop 조건 설정

* tune.run의 stop 항목의 t의 값에 따라 멈출 수 있다.

예를 들어

```
stop={
"mean_accuracy": 0.99,
"training_iteration": 1
},
```
이 경우 max_t가 아무리 길어도, trial이 모두 다 running되었을 때(1번), 1번으로 간주되어 종료된다.

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


### BOHB의 max_concurrent 및 cpu 값을 잘 활용

* 병렬적으로 처리하게 되어 최적화 속도가 개선될 수 있다.