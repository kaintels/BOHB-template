import ray
from ray import tune
from model_tf_manual import objective
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bohb import TuneBOHB

bohb_seed =777

def tune_mnist(training_iter):

    sha_scheduler = ASHAScheduler(
    max_t=10,  # 10 training iterations
    grace_period=1,
    reduction_factor=2)

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=81,
        reduction_factor=3,
        stop_last_trials=False)

    bohb_search = TuneBOHB(
        max_concurrent=4, seed=bohb_seed) # 병렬적으로 작동할 수

    config = {
        "activation": tune.choice(["relu", "tanh"]),
        "neuron1": tune.randint(32, 64),
        "neuron2": tune.randint(32, 64),
        "optimizers": tune.choice(["rmsprop", "adam"]),
    }

    analysis = tune.run(
        objective,
        config=config,
        scheduler=sha_scheduler,
        # search_alg=bohb_search,
        num_samples=3, # 고려할 샘플 수들
        stop={
            "mean_accuracy": 0.91
        },
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
