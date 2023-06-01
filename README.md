# The Effect of Art Identification with Convolutional Neural Networks According to The Difference in The Dataset
This project was done as part of a deep learning course at Tel Aviv University.  In the project I performed an experiment to see the effect  of three types of datasets on art classificaion. 
in this project I'm comparing three type of dataset. The first dataset, we will refer to him as the "small" dataset, is the smallest, the data balanced contain 1,600-1,700 per class, total of 45,509 images. The second dataset, we will refer to him as the "large" dataset, is the largest, the dataset is unbalanced I use all the images that I have the smallest class has 1,592 images and the largest class has 13,060 with total of 99,167 images. The third dataset, we will refer to him as the "synthetic" dataset, I took the small dataset and increase the number of images per class with data augmentation to 3,200-3,400 per class with total of 91,018. 
these three dataset are compared with the performance of another experiment, "Artist Identification with Convolutional Neural Networks", on two model. Baseline CNN, a relatively shallow network, and ResNet-18.

## Dataset
these project files contain a file of "csv file creator" which help prepering the data for training.

The source of the data is WikiArt
Note:

1.The WikiArt dataset can be used only for non-commercial research purpose.

2.The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images.

3.By using the WikiArt dataset, you agree to obey the terms and conditions of https://www.wikiart.org/.

## Models

### Base Line CNN

![BaseLineCNN](https://github.com/Bengal1/The-Effect-of-Art-Identification-with-Convolutional-Neural-Networks-According-to-The-Difference-in-T/assets/34989887/260e557b-48d1-43f8-a16c-dfb8c03b2f3a)


### ResNet-18



## Getting Started

In order to compile this project you will need:

Pytorch - https://pytorch.org/get-started/locally/

Scikit Image - https://scikit-image.org/docs/stable/install.html

Upload the files to python environment.

There are 4 location that you would need to set. the dataset path and 3 csv files paths to there location on your computer.
If you would like to use th "CSV_creatoer" set the wanted paths as well.

In the main file " .py" set the path to the csv files of your chosen location.

Operating the "synthetic" dataset:

First set the path to the aproptiate csv file, "csv_style_synthetic", by set: csv_loc = csvs[2], on line///.
Next fill in PaintingDataset on line /// in transform argument "transform_synth", like this: PaintingDataset(csv_file=csv_loc, root_dir=data_root, transforms=transform_synth), and you are ready to go.









