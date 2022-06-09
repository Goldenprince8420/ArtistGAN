echo "Run Started..."
bash kaggle.sh
bash modules.sh
python main.py
bash git_push.sh
echo "Run Done!!"