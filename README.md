# Main Repository

This repository contains classes and functions that are called in multiple notebook files.

Important Files/Folder in this repo:
  * main.py [link](main.py): Following functions are present in this file
    * train: For training model on train dataset
    * test: For evaluating model on test dataset
    * fit_model: Model is executed in this function. Optimized, scheduler, number of epochs, lambda for L1 etc are managed in this.
  * utils.py [link](utils.py): Following functions are present in this file
    * Class cifar_ds10: Loads data for CIFAR10
    * tl_ts_mod: Create DataLoader for images
    * set_albumen_params: Define albumentations to be applied
    * load_data: To load actual data for Mean & Standard Deviation calculations
    * display_incorrect_pred: Show misclassified images
    * show_sample: Show sample images
    * process_dataset: Apply transformations to dataset
    * plot_acc_loss: Plot accuracy & Loss curve
    * find_lr: Find LR 
    * Class GradCAM: GradCAM class 
    * generate_gradcam: Generate GradCam for layers
    * plot_gradcam: Show GradCam generated images    
  * models folder
    * resnet.py [link](models/resnet.py): It contains Resnet18 & Resnet34 definition
    * custom_resnet.py [link](models/custom_resnet.py): It contains custom Resnet definition
    * model9.py [link](models/model9.py): It contains basic transformer definition
