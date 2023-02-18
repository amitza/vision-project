# Font classification

This is the final project for the Open University's computer vision course 2023. 

## Use to model

### Prerequisites:

* Python 3.7+
* Pypi libraries:
  - Tensorflow
  - Keras
  - OpenCV

### How to run

Run the python file with those arguments: 

```shell
predict_font.py --dataset_file '<path>' --prediction_file '<path>'
```

Default for `--prediction_file` is `predictions.csv`

### Additional arguments

```shell
usage: predict_font.py [-h] [--images_file IMAGES_FILE] [--dataset_file DATASET_FILE] [--prediction_file PREDICTION_FILE] [--save_cropped SAVE_CROPPED]
  --images_file IMAGES_FILE                        
                        Path to the dataset file, must be .h5
  --dataset_file DATASET_FILE, -d DATASET_FILE
                        Path to the processed data
  --prediction_file PREDICTION_FILE, -o PREDICTION_FILE
                        Path to the output file
  --save_cropped SAVE_CROPPED, -s SAVE_CROPPED
                        Should save the cropped images under cropped_images
```

## Prediction file

The predictions file is a csv that has the character image name in this format: `{img_name}-{word_index}-{char_index}.jpg` and the font.

example:

```csv
Name,Font
ant+hill_102.jpg_0-0-0.jpg,Titillium Web
ant+hill_102.jpg_0-0-1.jpg,Open Sans
ant+hill_102.jpg_0-0-2.jpg,Titillium Web
ant+hill_102.jpg_0-1-0.jpg,Open Sans
ant+hill_102.jpg_0-1-1.jpg,Titillium Web
ant+hill_102.jpg_0-1-2.jpg,Titillium Web
```