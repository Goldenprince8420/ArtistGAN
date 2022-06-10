# ArtistGAN<br>
## Project Summary<br>
This Repository is a **Generative Modelling Project**. <br>
Author of this Project: **Rahul Golder** <br>

Project Created as part of the Kaggle Competetion: *I'm Something of a Painter Myself*. <br>

About the Project: The model ArtistGAN will convert a real-domain image to a monet style painting domain image through a domain transfer feature mapping.<br>

Motivation of Modelling Architecture: **CycleGAN Model for Domain Features Transfer**. <br>

## Getting Started<br>
Getting Started with the Project: <br>
1. Run the following code for to import os module and also create a content directory. <br>
`import os` <br>
`os.chdir("/content")`
2. Run the code to clear the ArtistGAN directory if it exists.<br>
`!rm -r /content/ArtistGAN`
3. Run the code for cloning the Project Repository.<br>
`!git clone https://github.com/Goldenprince8420/ArtistGAN.git`
4. Set Artist Directory as the current directory.
5. Run the following script `kaggle.sh` for Kaggle Setup and Data downloads.<br>
`!bash kaggle.sh`
6. Unzip the Training Weights.<br>
`! unzip cyclegan-training-weights.zip -d weights`
7. Run the script file `run.sh` for running the model.<br>
`!bash run.sh`<br>
The Programme will run with default setup of the model running parameters.  It may take 5 to 10 minutes.

