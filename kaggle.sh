#!/bin/sh
#echo "what is your name?"
#read name
#echo "How do you do, $name?"
#read remark
#echo "I am $remark too!"

pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
kaggle datasets list
kaggle competitions download -c gan-getting-started
kaggle datasets download -d rahulgolder/cyclegan-training-weights
mkdir data
unzip gan-getting-started.zip -d data
unzip cyclegan-training-weights.zip -d data
