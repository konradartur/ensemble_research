import sys
import yaml
from experiments import reproduce_experiment

if __name__ == "__main__":

    args = sys.argv[1:]
    mode = args[0]

    if mode == "reproduce":

        experiment_name = args[1]
        config_file = open("./configs/reproduction_configs.yaml", 'r')
        config = yaml.safe_load(config_file).get(experiment_name, None)
        config_file.close()

        reproduce_experiment(experiment_name, **config)

    else:
        raise NotImplementedError(f"Only reproduction supported.")
