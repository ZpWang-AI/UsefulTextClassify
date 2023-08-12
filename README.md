# UsefulTextClassify

Classify QA corpus by BERT.

### Train

1. Check data files in folder "data"
2. Reset config in file "train.py"
3. Run $python\ train.py$
4. Fetch the result from fold "saved_res", which would be automatically created by code

### Infer

1. Get parameters of the model
2. Reset config in file "inference.py"
3. Run $python\ inference.py$
4. Fetch the result from fold "data" or the path set by code

#### some useful log files are saved in folder "saved_res", if any