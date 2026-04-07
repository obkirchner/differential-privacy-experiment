To let this experiment run as I did, one has to:

1. Download "glove.6B.zip" from https://nlp.stanford.edu/projects/glove/
2. Extract glove.6B.100d.txt from it and place into the root of this project
3. Then create a .zip of the project (containing the glove.6B.100d.txt)
4. Upload this zip as dataset on kaggle.com
5. Create a new notebook on kaggle.com\
   **In this notebook:**
7. Enable GPU P100 via Settings > Accelerator
8. Add the previously created dataset as input (via the right-hand sidebar)
9. Issue "mv ../input/dataset/*username*/*dataset_name* ."
10. Issue "cd *into the root of the project*"
11. Issue "!python -m pip install --no-cache-dir "opacus==1.4.1" --no-deps"
12. Issue "!python main.py"
