#!/bin/bash

echo "Downloading dataset ..."
wget -O blogDataset.zip http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip

echo "Unzipping ..."
unzip -jq blogDataset.zip -d ./unprocessedBlogs/

echo "Preparing dataset ..."
python3 PrepareDataset.py

echo "Removing temporary data ..."
rm blogDataset.zip
rm -rf ./unprocessedBlogs

echo "Finished!"
