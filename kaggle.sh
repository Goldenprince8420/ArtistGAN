#!/bin/sh
#echo "what is your name?"
#read name
#echo "How do you do, $name?"
#read remark
#echo "I am $remark too!"

echo "Kaggle Setup and Data Installation start.."
pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list
kaggle competitions download -c gan-getting-started
kaggle datasets download -d rahulgolder/cyclegan-training-weights
mkdir data
unzip gan-getting-started.zip -d data
#mkdir weights
#unzip cyclegan-training-weights.zip -d weights
echo "Installation and preparation done!!"