from sklearn.pipeline import FeatureUnion
import torch
import multiprocessing


def args_interpreter(args):

    print(f"Accelerator: {args.accelerator}")

    if args.devices.isdigit():
        args.devices = int(args.devices)

    n_cpus = multiprocessing.cpu_count()

    # Print the number and names of GPUs used
    if args.accelerator == "gpu":
        n_gpus = torch.cuda.device_count()
        if args.devices == "auto":
            print(f"Using all {n_gpus} GPUs:")
            for i in range(n_gpus):
                print(f" - {torch.cuda.get_device_name(device=i)}")
        else:
            if args.devices > n_gpus:
                print(f"Requested number of GPUs is superior to the number of GPUs available on this machine ({n_gpus}).")
                print(f"Setting number of used GPUs to maximum.")
                args.devices = n_gpus
            else:
                print(f"Using {args.devices} GPU(s):")
            for i in range(args.devices):
                print(f" - {torch.cuda.get_device_name(device=i)}")
    # Print the number of cores used if CPU is selected
    elif args.accelerator == "cpu":
        if args.devices == "auto":
            print(f"Using all {n_cpus} CPU cores.")
        else:
            if args.devices > n_cpus:
                print(f"Requested number of CPU cores is superior to the number of CPU cores available on this machine ({n_cpus}).")
                print("Setting number of used CPU cores to maximum.")
                args.devices = n_cpus
            print(f"Cores used: {args.devices}")

    if args.workers > n_cpus:
        print("Requested number of workers is superior to the number of CPU cores available on this machine." )
        print("Setting number of workers to maximum.")
        args.workers = n_cpus
    print(f"Number of workers used: {args.workers}")

    print(f"Maximum number of epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Initial learning rate: {args.lr}")

    return args


def calculate_input_shape(feature_name, feature_processing_parameters):
    if feature_name == "spectrogram":
        input_height = (feature_processing_parameters["n_fft"] // 2) + 1
        input_width = (feature_processing_parameters["n_samples"]) // (feature_processing_parameters["n_fft"] - (feature_processing_parameters["n_fft"] // feature_processing_parameters["hop_denominator"])) - 1
    elif feature_name == "mel-spectrogram":
        input_height = feature_processing_parameters["n_mels"]
        input_width = (feature_processing_parameters["n_samples"] + feature_processing_parameters["n_fft"]) // (feature_processing_parameters["n_fft"] - (feature_processing_parameters["n_fft"] // feature_processing_parameters["hop_denominator"]))
    elif feature_name == "mfcc":
        input_height = feature_processing_parameters["n_mfcc"]
        input_width = (feature_processing_parameters["n_samples"] + feature_processing_parameters["n_fft"]) // (feature_processing_parameters["n_fft"] - (feature_processing_parameters["n_fft"] // feature_processing_parameters["hop_denominator"]))
    print(f"Input dimensions: {input_height}x{input_width}")
    return input_height, input_width
