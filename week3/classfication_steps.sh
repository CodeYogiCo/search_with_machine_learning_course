# Uncompress all xml files in week 3
#gunzip /workspace/search_with_machine_learning_course/data/*/*.xml.gz

# Run the script preparing data for fasttext
set -ex
python /workspace/search_with_machine_learning_course/week3/createContentTrainingData.py --sample_rate 0.5 --min_products 10

# Shuffle the lines
shuf /workspace/datasets/fasttext/output.fasttext > /workspace/datasets/fasttext/shuffled.fasttext

# Make sure all lines are in there
wc -l /workspace/datasets/fasttext/shuffled.fasttext

# Split into train and test
head -n 10000 /workspace/datasets/fasttext/shuffled.fasttext > /workspace/datasets/fasttext/train.fasttext
tail -n 10000 /workspace/datasets/fasttext/shuffled.fasttext > /workspace/datasets/fasttext/test.fasttext

# Pick out a validation set for autotuning
#sed '10001,15001 ! d' /workspace/datasets/fasttext/shuffled.fasttext > /workspace/datasets/fasttext/validate.fasttext

# move to where the data is
#cd /workspace/datasets/fasttext

# preprocess the files, lowercasing and 
# cat train.fasttext | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > train.preprocessed.txt
# cat test.fasttext | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > test.preprocessed.txt
# cat validate.fasttext | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > validate.preprocessed.txt

# Train a model
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/train.fasttext -output /workspace/search_with_machine_learning_course/week3/model/search_with_ml_week3_model -epoch 50 -wordNgrams 2 -lr 2

# Test the model
~/fastText-0.9.2/fasttext test /workspace/search_with_machine_learning_course/week3/model/search_with_ml_week3_model.bin  /workspace/datasets/fasttext/test.fasttext

# Try looking for hyperparameters automatically
# ~/fastText-0.9.2/fasttext supervised -input train.fasttext -output products_auto -autotune-validation validate.fasttext

# Train with one-vs-all
#~/fastText-0.9.2/fasttext supervised -input train.fasttext -output products -epoch 50 -wordNgrams 2 -lr 0.5
# -loss one-vs-all -minCount 5 -minCountLabel 5