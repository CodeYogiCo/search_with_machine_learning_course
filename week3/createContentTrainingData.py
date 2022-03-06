import argparse
import os
import random
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer



def transform_name(product_name):
  
    product_name= product_name.lower()

    # tokenize using ntlk tokenize
    #tokens= word_tokenize(product_name)

    tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    tokens= tokenizer.tokenize(product_name)

    
    stemmer = SnowballStemmer("english")

    tokens = [stemmer.stem(token) for token in tokens]
    product_name = " ".join(tokens)
    return product_name

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")
general.add_argument("--category_depth", default=0, type=int, help="The category depth for choosing categories. default is 0")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate
category_depth = args.category_depth
products_list = []




for filename in os.listdir(directory):
    if filename.endswith(".xml"):
        print("Processing %s" % filename)
        f = os.path.join(directory, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - (category_depth+1)][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                    #   output.write("__label__%s %s\n" % (cat, transform_name(name)))

                      product_dict = {}
                      product_dict['category'] = cat
                      product_dict['name'] = transform_name(name)
                      products_list.append(product_dict)


df = pd.DataFrame(products_list) 
if min_products > 0:
    print("Filtering min products per category results")
    gb = df.groupby('category')
    df_filtered = gb.filter(lambda x: x['category'].count() > min_products)
    print("{} products filtered".format(len(df)-len(df_filtered)))
    df = df_filtered

with open(output_file, 'w') as output:
    for index, row in df.iterrows():
        output.write("__label__%s %s\n" % (row['category'], row['name']))