
# ToothNet: Enabling Multi-Task Learning in YOLOX for Caries and Sealant Detection

[YOLOX](https://arxiv.org/abs/2107.08430) represents a substantial advancement in the realm of object detection models.
Building upon the [YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once) framework,
YOLOX significantly enhances both accuracy and efficiency in object detection tasks.
It switches the YOLO detector to an anchor-free manner and conduct other advanced detection techniques,
i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results. 
Due to the decoupled output heads of YOLOX,
it is straightforward to extend the model's outputs to cater to multiple categories,
thus allowing seamless integration into multi-task learning.
Consequently, we conducted experiments in the domains of oral caries and sealant detection.

## Key Features

- **Higher Accuracy and Faster Speed**.Compared to models handling individual tasks that require repeated runs,
  The entire ToothNet was optimized in an end-to-end manner and executed simultaneously to produce results for all three tasks,
  resulting in faster processing.
  Furthermore, it allows learning relationships between tasks, enhancing the model's overall performance.

- **End-to-End Modular Design**. ToothNet leverages decoupled output heads, allowing the combination of multiple detection and classification tasks.
  This enables users to effortlessly construct customized multi-task detection analysis models.


## How to use it?

We recommend users to follow our best practices for installing and using ToothNet.
Fortunately, this code repository has no additional deep learning framework dependencies and is built solely on [PyTorch](https://pytorch.org/).
Feel free to proceed with confidence, following the practical instructions below to train and test the code in this repository.


### Step0: Install the required dependencies

In this section, we will list the essential core dependencies for this project,
including the **PyTorch** framework and commonly used data manipulation libraries such as **NumPy**.
Meeting these dependencies is crucial for successfully building and running the project:

```
numpy
scipy
torch
torchvision
opencv-python
scikit-image
tqdm
yml
```

### Step1: Prepare your dataset
In order to make the data set structure compatible with the files provided by this repository,
you should convert your own task data format to a coco-style data set.
In this step, we will guide you through the process of preparing the dataset for your project. A well-prepared dataset is crucial for training and evaluating your model effectively. Follow these steps to ensure your dataset is ready for use:

1. **Data Collection**: Gather the necessary data for your specific task. Ensure the data is relevant, diverse, and well-annotated, adhering to the requirements of your project.

2. **Data Preprocessing**: Clean and preprocess the collected data as needed. This may involve tasks such as resizing images, normalizing pixel values, handling missing data, etc.

3. **Data Splitting**: Divide the dataset into appropriate subsets for training, validation, and testing. Common splits include 70% for training, 10% for validation, and 20% for testing.

Ensure that your dataset is well-organized and ready for use before proceeding to the next steps of the project.

### Step2: Train

Specify the configuration and working directory,
and modify the data type and category information in `train.py`, then click `run`.

```python
classes_path = 'model_data/classes.txt'                  # class information 

model_path = '/data/ToothData/model_data/ToothNet.pth'   # model weight save path

input_shape = [640, 640]                                 # image input shape
```


### Step3: Test

Specify the configuration and working directory,
and modify the image directory path in `predict.py`, then click `run`.

```python
dir_origin_path = "/data/ToothData/testimages"        # the test images directory path

dir_save_path = "/data/ToothData/testpredictimages"   # the prediction result save directory path
```


### Step4: Statistics analysis

In this step, we encourage readers to perform statistical analysis using specialized software such as _MedCalc_ or _SPSS_.
Utilize these tools to analyze the data comprehensively,
calculate relevant statistical measures,
and derive meaningful insights from your experiments.
Whether it's hypothesis testing, correlation analysis, or any other statistical procedure, employing dedicated software will enhance the accuracy and robustness of your statistical analysis.
Remember, sound statistical analysis is fundamental to drawing valid conclusions and making informed decisions based on your data.


## License

For academic purposes, this project is licensed under the 2-clause BSD license - please refer to the license file for detailed information. For commercial use, please contact the author.