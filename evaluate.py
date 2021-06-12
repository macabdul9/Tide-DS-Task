import h2o
import gc
import argparse
from sklearn.metrics import classification_report
# initizlie the h2o instance
h2o.init(max_mem_size='14G')

def evaluate(model_path, file_path):
    
    data = h2o.import_file(file_path)
    
    model = h2o.load_model(model_path)
    
    y_true = h2o.as_list(data)['target'].values
    data.drop("target")
    
    preds_label = model.predict(data)
    
    preds = model.predict(data)
    
    preds_label = preds > 0.5
    
    y_pred = h2o.as_list(preds_label)['predict'].values
    
    cr = classification_report(y_true=y_true, y_pred=y_pred, digits=4)
    
    del model
    gc.collect()
    
    return cr


if __name__ == '__main__':
    
    print(evaluate(model_path="./models/StackedEnsemble_AllModels_AutoML_20210612_063344", file_path="./data/test.csv"))