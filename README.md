# Video Based Sign Language Gesture Using CNN-RNN Model


## Requirements
* Set up a Virtual Environment using Anaconda
* Install opencv
* Install tensorflow:
  ```shell
  pip install tensorflow==1.15
  ```
* Install tflearn
  ```shell
  pip install tflearn
  ```

## Training and Testing

### 1. Data Folder
  
Create two folders with any name say **train_videos**  and **test_videos** in the project root directory. It should contain folders corresponding to each cateogry, each folder containing corresponding videos.

For example:

```
train_videos
├── Accept
│   ├── 050_003_001.mp4
│   ├── 050_003_002.mp4
│   ├── 050_003_003.mp4
│   └── 050_003_004.mp4
├── Appear
│   ├── 053_003_001.mp4
│   ├── 053_003_002.mp4
│   ├── 053_003_003.mp4
│   └── 053_003_004.mp4
├── Argentina
│   ├── 024_003_001.mp4
│   ├── 024_003_002.mp4
│   ├── 024_003_003.mp4
│   └── 024_003_004.mp4
└── Away
    ├── 013_003_001.mp4
    ├── 013_003_002.mp4
    ├── 013_003_003.mp4
    └── 013_003_004.mp4
```



### 2. Extracting frames

#### Command


#### Extracting frames form training videos

```bash
python "video-to-frame.py" train_videos train_frames
```
Extract frames from gestures in `train_videos` to `train_frames`.

#### Extracting frames form test videos

```bash
python "video-to-frame.py" test_videos test_frames
```
Extract frames from gestures in `test_videos` to `test_frames`.

### 3. Retrain the Inception v3 model.

- Download retrain.py.
   ```shell
   curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
   ```

- Run the following command to retrain the inception model.
  
    ```shell
    python retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train_frames
    ```

This will create two file `retrained_labels.txt` and `retrained_graph.pb`




### 4. Intermediate Representation of Videos



#### Approach 1

- Each Video is represented by a sequence of `n` dimensional vectors (probability distribution or output of softmax) one for each frame. Here `n` is the number of classes.

    **On Training Data**

    ```shell
    python predict_spatial.py retrained_graph.pb train_frames --batch=100
    ```

    This will create a file `predicted-frames-final_result-train.pkl` that will be used by RNN.

    **On Test Data**

    ```shell
    python predict_spatial.py retrained_graph.pb test_frames --batch=100 --test
    ```

    This will create a file `predicted-frames-final_result-test.pkl` that will be used by RNN. 

#### Approach 2

- Each Video represented by a sequence of 2048 dimensional vectors (output of last Pool Layer) one for each frame

    **On Training Data**

    ```shell
    python predict_spatial.py retrained_graph.pb train_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100
    ```

    This will create a file `predicted-frames-GlobalPool-train.pkl` that will be used by RNN.

    **On Test Data**

    ```shell
    python predict_spatial.py retrained_graph.pb train_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100 --test
    ```

    This will create a file `predicted-frames-GlobalPool-test.pkl` that will be used by RNN.

### 5. Train the RNN.


#### Approach 1

```bash
python rnn_train.py predicted-frames-final_result-train.pkl non_pool.model
```

This will train the RNN model on the **softmax based representation** of gestures for 10 epochs and save the model with name `non_pool.model` in a folder named checkpoints.

#### Approach 2

```bash
python rnn_train.py predicted-frames-GlobalPool-train.pkl pool.model
```

This will train the RNN model on the **pool layer based representation** of gestures for 10 epochs and save the model with name `pool.model` in a folder named checkpoints.


### 6. Test the RNN Model

#### Approach 1

```bash
python rnn_eval.py predicted-frames-final_result-test.pkl non_pool.model
```

This will use the `non_pool.model` to predict the labels of the **softmax based representation** of the test videos.
Predictions and corresponding gold labels for each test video will be dumped in to **results.txt**

#### Approach 2

```bash
python rnn_eval.py predicted-frames-GlobalPool-test.pkl pool.model
```

This will use the `pool.model` to predict the labels of the **pool layer based representation** of the test videos.
Predictions and corresponding gold labels for each test video will be dumped in to **results.txt**
