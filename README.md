# The Effect of Art Identification with Convolutional Neural Networks According to The Difference in The Dataset
This project was done as part of a deep learning course at Tel Aviv University.  In the project I performed an experiment to see the effect  of three types of datasets on art classificaion. 
in this project I'm comparing three type of dataset. The first dataset, we will refer to him as the "small" dataset, is the smallest, the data balanced contain 1,600-1,700 per class, total of 45,509 images. The second dataset, we will refer to him as the "large" dataset, is the largest, the dataset is unbalanced I use all the images that I have the smallest class has 1,592 images and the largest class has 13,060 with total of 99,167 images. The third dataset, we will refer to him as the "synthetic" dataset, I took the small dataset and increase the number of images per class with data augmentation to 3,200-3,400 per class with total of 91,018. 
these three dataset are compared with the performance of another experiment, "Artist Identification with Convolutional Neural Networks", on two model. Baseline CNN, a relatively shallow network, and ResNet-18.

## Database
Art Style identification is the task of identifying the style of a painting given with no other information about it. This is an important requirement for cataloguing art, especially as art is increasingly digitized. One of the most vast and diverse datasets, WikiArt, has around 250,000 artworks from over 200 different art styles by 3,000 artists.
My dataset, sorted and modified WikiArt, consists of at list 1,600 paintings per style from 27 art style and total of 100,000 approximately. 
In this experiment I use four datasets (Control, Small, Large, Synthetic) built from the dataset I mentioned above. Every dataset has separated .csv file, and every image is labeled with its style. I split every dataset into training, validation and test sets in 3 ratio 80-10-10 respectively.
The "control" dataset will be the referenced dataset to ["Artist Identification with Convolutional Neural Networks"](http://cs231n.stanford.edu/reports/2017/pdfs/406.pdf). The "small" dataset is balanced and contain 1,600-1,700 per class, total of 45,509 images. The "large" dataset is unbalanced I have use all the images in in the database of the selected classes, the smallest class has 1,592 images, and the largest class has 13,060 with total of 99,167 images. In the "synthetic" dataset, I took the small dataset and increase the number of images per class with data augmentation to 3,200-3,400 per class with total of 91,018. which make it large and balanced.

This project contains the file "CSV_Creator.py" which help prepering the data for training.
### WikiArt Notes:

1. The WikiArt dataset can be used only for non-commercial research purpose.
2. The images in the WikiArt dataset were obtained from WikiArt.org. The authors are neither responsible for the content nor the meaning of these images.
3. By using the WikiArt dataset, you agree to obey the terms and conditions of [WikiArt.org](https://www.wikiart.org/).

## Models

### Base Line CNN
I train a simple CNN from scratch for art style identification. I took the architecture from ["Artist Identification with Convolutional Neural Networks"](http://cs231n.stanford.edu/reports/2017/pdfs/406.pdf). As the name implies, this network serves as a baseline for comparison with the other model. The network is relatively shallow compared to the other network. The network is obviously don't feet to identify complex images, and it's all purpose is to be a reference point to the deeper model. Every layer in the network down-samples the image by a small factor to reduce computational complexity. The downside of this model is that it will not analyze the image small details.

![BaseLineCNN](https://github.com/Bengal1/The-Effect-of-Art-Identification-with-Convolutional-Neural-Networks-According-to-The-Difference-in-T/assets/34989887/60a595c1-e73e-4cd7-8839-db78aca042ce)


### ResNet-18
The next network is based on the ResNet-18 architecture from [Torchvision](https://github.com/pytorch/vision). ResNet is a residual neural network that use shortcut connections, shortcut connections are those skipping one or more layers. The ResNet I am using has 18 layers and set with a new fully connected layer to allow for right number of classes predictions, ResNets use residual blocks to ensure that upstream gradients are propagated to lower network layers, aiding in optimization convergence in deep networks.

![resnet18_modified](https://github.com/Bengal1/The-Effect-of-Art-Identification-with-Convolutional-Neural-Networks-According-to-The-Difference-in-T/assets/34989887/d793daf3-cc8a-4cff-8b3d-a84647d53c91)

## Experiment & Results
All the models and experiments are implemented in [PyTorch](https://github.com/pytorch). For the setup, implementation details and evaluation metrics see the experiment file, "The Effect of Art Identification with Convolutional Neural Networks According to The Difference in The Dataset", in this repository.

### Results
The results of the experiment divided to table per dataset, so we can the effects better against the compared dataset. With the same dataset on the ResNet-18, we can notice increase in top-1 and top-3 accuracy, but F1, precision and recall score are the same. On the synthetic dataset we observe an increase of 14% in top-1 accuracy from the small dataset, However, we can notice a decrease in top-3 accuracy.
![resultssss](https://github.com/Bengal1/The-Effect-of-Art-Identification-with-Convolutional-Neural-Networks-According-to-The-Difference-in-T/assets/34989887/a2a17a62-ecaf-49ca-86c5-4331c04f9319)

My best result was when I applied the ResNet-18 model on the large dataset and it was significantly higher for top-1 and top-3 accuracy. The large dataset is highly unbalanced and still the synthetic dataset overfitted by 13% and the large dataset overfitted by 9%, when applied on them the ResMet-18.
For the results, discussion, analysis and conclusion see the experiment file, "The Effect of Art Identification with Convolutional Neural Networks According to The Difference in The Dataset", in this repository.

## Getting Started
In order to compile this project you will need [Pytorch](https://pytorch.org/get-started/locally/) and [Scikit Image](https://scikit-image.org/docs/stable/install.html).

Upload the files to python environment. There are 4 location that you would need to set. the dataset path and 3 csv files paths to there location on your computer.
If you would like to use the "CSV_creatoer" set the wanted paths as well. In the main file "Train_file.py" set the path to the csv files of your chosen location.

Operating the "synthetic" dataset:
First set the path to the aproptiate csv file, "csv_style_synthetic", by set: csv_loc = csvs[2], on the corresponding line.
Next fill in PaintingDataset on the corresponding line in transform argument "transform_synth", like this: PaintingDataset(csv_file=csv_loc, root_dir=data_root, transforms=transform_synth), and you are ready to go.

## References
1. [Artist Identification with Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2017/pdfs/406.pdf).
2. [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385).
3. [Delving deep into rectifiers: Surpassing human-level performance on imagenet classification](https://arxiv.org/abs/1502.01852).
4. [Rhythmic brushstrokes distinguish van gogh from his contemporaries: Findings via automated brushstroke extractions](https://ieeexplore.ieee.org/document/6042878).







