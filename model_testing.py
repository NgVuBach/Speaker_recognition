import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from pycm import *
import matplotlib.pyplot as plt

class Models:
    file = None
    def __init__(self, file):
        with open('models\scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open('models\model_20.pkl', 'rb') as f:
            self.clf_20 = pickle.load(f)
            
        with open('models\model_10.pkl', 'rb') as f:
            self.clf_10 = pickle.load(f)
            
        with open('models\model_5.pkl', 'rb') as f:
            self.clf_5 = pickle.load(f)
        
        with open('models\model_1.pkl', 'rb') as f:
            self.clf_1 = pickle.load(f)

        self.data = pd.read_csv('Test_speaker.csv')
        self.data = self.data.drop(['Unnamed: 0'], axis = 1)
        self.X = self.scaler.transform(np.array(self.data.drop(['label'], axis=1), dtype = float))
        self.y = np.array(self.data['label'],dtype=str)

        self.file = file
        self.sample , self.sr = librosa.load(file)


def fit_model(model, X, y):
    y_pred = model.predict(X)
    return y_pred, y

def accuracy(model, X, y):
    y_pred, y = fit_model(model, X, y)
    # print(accuracy_score(y, y_pred))
    # print(precision_score(y, y_pred, average='macro'))
    # print(f1_score(y, y_pred, average='macro'))
    # print(recall_score(y, y_pred, average='macro'))

def confidence(model, scaler, data, sr):
    feature = []
    y = np.array(data)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr= np.mean(librosa.feature.zero_crossing_rate(y))

    feature.extend([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr])

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    for i, m in enumerate(mfcc):
        feature.append(np.mean(m))

    feature = np.array(feature)

    X = feature.reshape(1,-1)
    X = scaler.transform(X)
    prob = model.predict_proba(X)
    np.set_printoptions(suppress=True)
    prob = prob.flatten()
    column_name = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 
                       26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 
                       40, 41, 42, 43, 44, 45, 46, 5, 6, 7, 8, 9]
    name = np.array(column_name)

    confidence = np.column_stack((name, prob))
    confidence = confidence[confidence[:, 1].argsort()[::-1]]

    print(confidence[0, 1],"\t", confidence[0,0])

    return confidence[0, 1], int(confidence[0,0])

def model_report(model, X, y):
    y_pred = model.predict(X)
    clf_report = classification_report(y,
                                   y_pred,
                                   output_dict=True)
    fig, ax = plt.subplots(figsize=(9, 20))
    ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap='plasma', fmt=".6f")
    plt.show()

def confusion_matrix(model, X, y):
    y_pred, y = fit_model(model, X, y)
    cm = ConfusionMatrix(actual_vector=y, predict_vector=y_pred)
    cm = cm.to_array(normalized=True)
    column_name = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                   2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                   3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                   4, 40, 41, 42, 43, 44, 45, 46, 5, 6, 7, 8, 9]
    
    mtrix = pd.DataFrame(cm, columns=column_name, index= column_name)

    plt.figure(figsize=(20, 15))
    # sns.heatmap(mtrix[(mtrix > 0) & (mtrix <= 0.9)], annot=True, cmap='BrBG')
    ax = sns.heatmap(mtrix[(mtrix > 0.001)], annot=True, cmap='BrBG')
    # ax = sns.heatmap(mtrix, annot=True, cmap='BrBG')
    plt.xlabel('Predict', fontsize = 20) # x-axis label with fontsize 15
    plt.ylabel('Auctual', fontsize = 20) # y-axis label with fontsize 15
    fig = ax.get_figure()
    plt.show()

# more than 30 sec
def confidence_on_time(model, scaler, y, blocksize):
    start = 0
    count = 0
    # blocksize = sr
    accuracy = []
    result = []
    for end in range(blocksize, len(y), blocksize):
        conf, res = confidence(model, scaler, y[start:end], blocksize)
        accuracy.append(conf)
        result.append(res)
        if count <= 3:
            count += 1
        else:
            start = end - blocksize*2
        # start = end
    # print(np.mean(accuracy))
    accuracy = np.array(accuracy)
    result = np.array(result)
    # print(len(accuracy))
    # plt.ylim((0,1))
    # plt.plot(accuracy)
    # plt.ylabel("Confidence")
    # plt.xlabel("Time(sec)")
    # plt.show()
    return accuracy, result

def confidence_on_dir(dir):
    import os
    # dir = "Voice_in_noise\dish_washer"
    bigga = np.array([])
    for file in os.listdir(dir):
        md = Models(os.path.join(dir, file))
        accracy, result = confidence_on_time(md.clf_20, md.scaler, md.sample, md.sr)
        bigga = np.append(bigga, accracy)
    bigga.flatten()
    j2 = [i for i in bigga if i <= 0.8]
    print(j2)
    plt.ylim((0,1))
    plt.plot(bigga)
    plt.ylabel("Confidence")
    plt.xlabel("Time(sec)")
    plt.title(f'Mean: {np.mean(bigga)}')
    plt.show()

def noise_test(dir):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    acc = np.array([])
    res = np.array([])
    # dir = "Voice_in_noise\Room "
    for file in os.listdir(dir):
        md = Models(os.path.join(dir, file))
        accracy, result = confidence_on_time(md.clf_20, md.scaler, md.sample, md.sr)
        acc = np.append(acc, accracy)
        res = np.append(res, result)
    acc.flatten()
    res.flatten()
    df_array = {"Confidence": acc,
                "Result": res,
                "Time": [i for i in range(0, len(acc))]
    }
    df = pd.DataFrame(df_array)
    sn.scatterplot(data=df,x="Time", y="Confidence", style="Result")
    plt.ylim((0,1))
    plt.show()

if __name__ == '__main__':
    # md = Models("46_speaker_original\Test\Speaker03\Speaker03_part2.wav")
    # accuracy(md.clf_20, md.X, md.y)
    # confusion_matrix(md.clf_1, md.X, md.y)

    # confidence(md.clf_10, md.scaler, md.sample, md.sr )
    print("lmao")


