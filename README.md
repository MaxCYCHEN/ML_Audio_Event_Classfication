# ML_Audio_Event_Classfication
An Audio event detection training tool &amp; deployment

- We easy the process of training and deployment on MCU.


## 1. First step
### 1. Install virtual env  
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have done.
### 2. Running
- The `train.ipynb` will help you train the model, and convert it to a TFLite, C++ file and Vela TFLite.

## 2. Work Flow
### 1. data prepare
(a.) No matter downloaded [ESC-50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download) or your dataset, they should be in `dataset` and with `ESC format`.
(b.) This tool uses [ESC-50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download) dataset for training and testing which are given in ESC format.
(c.) Your dataset must be comprised of:
  - A folder containing the audio files with all same format, like `.wav`.
  - A `.csv` file with at least a "filename" column and a "category" column.

### 2. training
- `train.ipynb` provides various attributes for training configuration. User can control them by the easy UI in jupyter notebook. There are default model&training setting for `miniresnetv2` and `yamnet`,
  and user can start from here with pre-train model.
- More detail configuration are in `cfg/my_config.yaml`.
- After training, testing result of normal & quantization model will be show as pictures.
- The training reult and model are in `workspace/YOUR_PROJECT_NAME`
- <img src="https://github.com/MaxCYCHEN/tiny_nu_audio/assets/105192502/0453479b-8a74-44c2-a9e6-076ba4a70cfb" width="30%">


### 3. Test
- Use `Test` tab in `train.ipynb` to test the tflite model with single test audio file which not go through preprocessing (rearranging the audio file). This testing is more like MCU inference scenario.
- <img src="https://github.com/MaxCYCHEN/tiny_nu_audio/assets/105192502/812386b6-9ac0-4909-a12a-504bef43df4c" width="40%">

### 4. Fully test on Board (Optional)
- In `board_test` folder, we offer a pyOCD script to communicate the board to test all test dataset.
- We have a chance to test large number dataset from MCU and get the result. 

### 5. Deployment
- Utilize the `Deployment` tab in `train.ipynb` to convert the TFLite model to C source/header files and Vela TFLite.
-  <img src="https://github.com/MaxCYCHEN/tiny_nu_audio/assets/105192502/09eca44b-74c9-45a8-8124-f855255263d0" width="40%">

## 3. Inference code
- [ML_M460_SampleCode](https://github.com/OpenNuvoton/ML_M460_SampleCode)
- [M55M1]

