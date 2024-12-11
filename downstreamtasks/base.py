import PLA
import EI
import PFC

def main():
    print("Argument Reader for Downstream Tasks")
    
    tasks = ["PLA", "PFC", "EI", "MSP"]
    modalities = ["sequence", "graph", "pointcloud", "multimodal"]
    modes = ["process", "train", "test", "train-test", "all"]
    pfc_datasets = ["family", "fold", "superfamily"]
    pla_datasets = ["DAVIS", "KIBA"]
    task_dataset = ""
    batches = False
    
    task = None
    modality = None
    mode = None
    dataset = None
    
    # Read Arguments
    while task not in tasks:
        task = input(f"Select the task ({', '.join(tasks)}): ").strip().upper()
        if task not in tasks:
            print("Invalid task. Please choose from the available options.")
    
    while mode not in modes:
        mode = input(f"Select the mode ({', '.join(modes)}): ").strip().lower()
        if mode not in modes:
            print("Invalid mode. Please choose from the available options.")
            
    batch = input(f"Choose to process in batches (True/False): ").strip().lower()
    if batch not in ["true", "false"]:
        print("Invalid selection. Please choose 'True' or 'False'.")
    else:
        batches = batch == "true"
            
    if mode in ['train', 'test', 'train-test', 'all']:
        while modality not in modalities:
            modality = input(f"Select the modality ({', '.join(modalities)}): ").strip().lower()
            if modality not in modalities:
                print("Invalid modality. Please choose from the available options.")
    
    # Task-Specific Arguments
    if task == "PFC" and mode == "test":
        while dataset not in pfc_datasets:
            dataset = input(f"Select the test dataset for PFC ({', '.join(pfc_datasets)}): ").strip().lower()
            if dataset not in pfc_datasets:
                print("Invalid test dataset. Please choose from the available options.")
        task_dataset = dataset
    
    if task == "PLA":
        while dataset not in pla_datasets:
            dataset = input(f"Select the dataset for PLA ({', '.join(pla_datasets)}): ").strip().upper()
            if dataset not in pla_datasets:
                print("Invalid dataset. Please choose from the available options.")
        task_dataset = dataset
    
    # Print Final Configuration
    print("\nFinal Configuration:")
    print(f"Task: {task}")
    print(f"Mode: {mode}")
    if mode == "train" or mode == "test":
        print(f"Modality: {modality}")
    if mode == "process":
        print(f"Batch Processing: {batches}")
    if task == "PFC" and mode == "test" or task == "PLA":
        print(f"Dataset: {task_dataset}")
    
    # TODO: Implement the downstream task logic based on the selected configuration
    
    if task == "PLA":
        match mode:
            case "process":
                PLA.process(task_dataset, batches)
            case "train":
                PLA.train(task_dataset, modality, batches)
            case "test":
                PLA.test(task_dataset, modality, batches)
            case "train-test":
                PLA.train(task_dataset, modality, batches)
                PLA.test(task_dataset, modality, batches)
            case "all":
                print(f"Note: Only {modality} will be trained and tested.")
                PLA.process(task_dataset, batches)
                PLA.train(task_dataset, modality, batches)
                PLA.test(task_dataset, modality, batches)
    elif task == "PFC":
        match mode:
            case "process":
                PFC.process(batches)
            case "train":
                PFC.train(modality, batches)
            case "test":
                PFC.test(task_dataset, modality, batches)
            case "train-test":
                PFC.train(modality, batches)
                PFC.test(task_dataset, modality, batches)
            case "all":
                print(f"Note: Only {modality} will be trained and tested.")
                PFC.process(batches)
                PFC.train(modality, batches)
                PFC.test(task_dataset, modality, batches)
    elif task == "EI":
        match mode:
            case "process":
                EI.process(batches)
            case "train":
                EI.train(modality, batches)
            case "test":
                EI.test(modality)
            case "train-test":
                EI.train(modality, batches)
                EI.test(modality)
            case "all":
                print(f"Note: Only {modality} will be trained and tested.")
                EI.process(batches)
                EI.train(modality, batches)
                EI.test(modality)
    elif task == "MSP":
        pass

if __name__ == "__main__":
    main()
