# Kernel preparation

# Warning suppression
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.warnings.filterwarnings('ignore')
np.random.seed(1001)
# Cliche
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import IPython
import pandas as pd
# import seaborn as sns

def play_audio(data):
    IPython.display.display(IPython.display.Audio(data=data))

# =========================================================================================

# Dataset preparation

# Root folder that contains entire dataset
DATAROOT = '/data/voice_zaloai/data'
df_train = pd.read_csv(DATAROOT + '/full_train_accent.csv')
df_test = pd.read_csv(DATAROOT + '/test.csv')

# =========================================================================================

def pred_geometric_mean(preds_set):
    predictions = np.ones_like(preds_set[0])
    for preds in preds_set:
        predictions = predictions*preds
    predictions = predictions**(1./len(preds_set))
    return predictions

def pred_geometric_mean_by_files(npy_pred_files):
    preds_set = np.array([np.load(file) for file in npy_pred_files])
    predictions = pred_geometric_mean(preds_set)
    return predictions

# =========================================================================================

model_types = ["alexnet", "seresnet", "vgg16"]
resolution_types = ["LH", "X"]

def get_all_prediction_files(root_path):
    for m in model_types:
        for r in resolution_types:
            for i in range(5):
                path = root_path + "/%s/%s/test_predictions_%d.npy" % (m, r, i)
                if os.path.exists(path):
                    yield path

def get_result(pred_files):
    ensembled_test_preds = pred_geometric_mean_by_files(pred_files)
    result = np.array([np.argmax(x) for x in ensembled_test_preds])
    return result

# =========================================================================================

gender_test_pred_files = list(get_all_prediction_files("./gender_result"))
accent_test_pred_files = list(get_all_prediction_files("./accent_result"))

gender_result = get_result(gender_test_pred_files)
accent_result = get_result(accent_test_pred_files)

df_test['gender'] = gender_result
df_test['accent'] = accent_result

df_test.columns = ['id', 'gender', 'accent']

df_test.to_csv("prediction_result.csv", index=False)

# =========================================================================================
# =========================================================================================
# =========================================================================================