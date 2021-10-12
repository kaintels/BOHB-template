import ray
from ray import tune
from model_tf_manual import objective
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.suggest.bohb import TuneBOHB

bohb_seed =777
epochs = 10
def tune_mnist(training_iter):

    # https://arxiv.org/abs/1810.05934
    sha_scheduler = ASHAScheduler(
    max_t=epochs,  # 10 training iterations
    grace_period=1,
    reduction_factor=2)

    # https://arxiv.org/abs/1603.06560
    hyperband = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        max_t=epochs,
        reduction_factor=3,
        brackets=1
    )
    # https://arxiv.org/abs/1807.01774
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=epochs,
        reduction_factor=3,
        stop_last_trials=False)

    bohb_search = TuneBOHB(
        max_concurrent=4, # 병렬적으로 작동할 수
        seed=bohb_seed) # 재현성을 위한 시드

    config = {
        "training_iteration" : epochs,
        "activation": tune.choice(["relu", "tanh"]),
        "neuron1": tune.randint(32, 64),
        "neuron2": tune.randint(32, 64),
        "optimizers": tune.choice(["rmsprop", "adam"]),
    }

    analysis = tune.run(
        objective,
        config=config,
        scheduler=sha_scheduler, # or hyperband or bohb_hyperband
        # search_alg=bohb_search, # 따로 설정하지 않을 시 랜덤 서치 및 그리드 서치 적용 
        num_samples=3, # 고려할 샘플 수들
        # stop={
        #     "mean_accuracy": 0.91
        # },
        metric="mean_accuracy",
        local_dir="./bohb_results",
        mode="max",
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },)

    print("Best hyperparameters found were: ", analysis.best_config)
    analysis.dataframe().to_csv("result.csv")

if __name__ == "__main__":
    ray.init(num_cpus=2, log_to_driver=False) # cpu 코어 수
    tune_mnist(1)
