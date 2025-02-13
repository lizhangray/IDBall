import torch
import data
from evaluator import Eval
import argparse  

def get_dataset_config(task_name):
    task_configs = {
        "Dehaze-OTS-05": {
            "scale": 2,
            "dataset_paths": {
                "input_dir": "datasets/SOTS/outdoor/hazy",
                "gt_dir": "datasets/SOTS/outdoor/gt"
            },
            "model_path":"model/OTS-05.pth",
            "eval_dataset":"EvalData_haze_SOTS_05"

        },
         "Dehaze-OTS-025": {
            "scale": 4,
            "dataset_paths": {
                "input_dir": "datasets/SOTS/outdoor/hazy",
                "gt_dir": "datasets/SOTS/outdoor/gt"
            },
            "model_path":"model/OTS-025.pth",
            "eval_dataset":"EvalData_haze_SOTS_025"

        },
        "Delowlight-ACDC_night-05": {
            "scale": 2,
            "dataset_paths": {
                "input_dir": "datasets/ACDC_night/test_low",
                "gt_dir": "datasets/ACDC_night/test_gt"
            },
            "model_path":"model/ACDC_night-05.pth",
            "eval_dataset":"EvalData_lowlight_05"

        },
        "Delowlight-ACDC_night-025": {
            "scale": 4,
            "dataset_paths": {
                "input_dir": "datasets/ACDC_night/test_low",
                "gt_dir": "datasets/ACDC_night/test_gt"
            },
            "model_path":"model/ACDC_night-025.pth",
            "eval_dataset":"EvalData_lowlight_025"

        },
        "Deblur-GoPro-025": {
            "scale": 4,
            "dataset_paths": {
                "root_dir": "datasets/GoPro/test",
            },
            "model_path":"model/GoPro-025.pth",
            "eval_dataset":"EvalData_GoPro_025"

        },
    }
    return task_configs.get(task_name, None)


def create_dataset(task_name):
    config = get_dataset_config(task_name)
    if config is None:
        raise ValueError(f"Task configuration not found: {task_name}")

    dataset_class_name = config["eval_dataset"]
    dataset_paths = config["dataset_paths"]

    try:
        dataset_class = getattr(data, dataset_class_name)
    except AttributeError:
        raise ValueError(f"Dataset class '{dataset_class_name}' not found in 'data' module.")

    try:
        dataset = dataset_class(**dataset_paths)
    except TypeError as e:
        print(f"Error occurred while creating dataset '{dataset_class_name}' instance: {e}")
        print(f"Please check if the constructor of dataset class '{dataset_class_name}' accepts these parameter names: '{dataset_paths.keys()}'.")
        raise

    return dataset
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate model performance on a specific task.")
    parser.add_argument("task_name", type=str, help="Name of the task to execute (e.g., Deblur-GoPro-025, Dehaze-OTS-05)") # Add task_name argument
    args = parser.parse_args() # Parse command-line arguments
    task_name = args.task_name # Get task_name from parsed arguments

    # Get task configuration
    config = get_dataset_config(task_name)

    if config is None:
        raise ValueError(f"Task configuration not found: {task_name}")


    # Create dataset
    test_dataset = create_dataset(task_name)

    # Initialize evaluator
    myEvaluator = Eval(device=device,scale=config["scale"])

    myEvaluator.loadmodel(config["model_path"])
    myEvaluator.eval(test_dataset, task=task_name)

if __name__ == '__main__':
    main()
