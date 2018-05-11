import pandas as pd


def write_submission(pred, path):
    df = pd.DataFrame(columns = ['_ID_','_VAL_'])
    df['_ID_'] = range(len(pred))
    df['_VAL_'] = pred[:,-1]
    df.to_csv(path, index=False)