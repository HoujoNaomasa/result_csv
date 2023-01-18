# import re
import numpy as np
import csv
import glob
import pandas as pd
from natsort import natsorted


def main():
    files = getfiles('result_txt')
    epoch = 30

    columns_name = []
    # biglist = np.array([])
    idx = ['total', 'class', 'coordinate', 'angle']
    # df_all = pd.DataFrame(index=idx)
    df_min_train = pd.DataFrame(index=idx)
    df_min_test = pd.DataFrame(index=idx)

    # print(df_all)

    mode = ['train', 'test']
    find_target = '/result_'
    for f in files:
        for m in mode:
            fp = f
            target = extract(fp, epoch, m)
            min_target = search_min(target)

            tgidx = fp.find(find_target)
            columns_name = fp[tgidx+len(find_target):-4]+m

            if m=='train':
                df_min_train[columns_name] = min_target
            else:
                df_min_test[columns_name] = min_target
    
    # print(df_min_train)
    print(df_min_test.T)
    df_min_test.T.to_csv('csv/total_min_loss_test.csv')



def makecsv_from_dir():
    files = getfiles()
    epoch = 30
    
    for f in files:
        fp = f
        target_train = extract(fp, epoch, 'train')
        target_test = extract(fp, epoch, 'test')

        filename_train = fp[7:-4]+'_train.csv'
        filename_test = fp[7:-4]+ '_test.csv'

        writecsv(target_train, filename_train)
        writecsv(target_test, filename_test)


def makecsv_from_singleresult():
    fp = 'result_mobile7200.txt'
    epoch = 30
    mode = 'train'

    target = extract(fp, epoch, mode)

    filename = fp[7:-4]+'.csv'
    writecsv(target, filename)

def getfiles(home):
    files = glob.glob(home + "/*.txt")
    # print(files)
    sorted_files = natsorted(files)
    return sorted_files

def writecsv(tglist, csvname):
    target = tglist
    filename = csvname
    np.savetxt(filename, target, delimiter=',')


def extract(filepath, ep, mode):

    file_data = open(filepath, 'r')
    all_text = list(file_data)
    file_data.close()

    epoch = ep
    if mode=='train':
        start = 3 
    else:
        start = 9

    all_loss = []
    for i in range(epoch):
        tcca = all_text[start:start+4]
        loss = single(tcca)
        all_loss.append(loss)
        start += 12
    
    all_np = np.array(all_loss).T

    return all_np

def search_min(all_np):
    each_min = np.array([])
    each_min_arg = np.array([])
    for i in all_np:
        each_min = np.hstack([each_min, i.min()])
        each_min_arg = np.hstack([each_min_arg, i.argmin()+1])
    
    # return each_min, each_min_arg
    return each_min

def single(aiueo):
    loss = []
    tcca = aiueo

    target = 'loss = '
    for t in range(len(tcca)):
        idx = tcca[t].find(target)
        result = tcca[t][idx+len(target):]
        
        loss.append(float(result))
    return loss

if __name__ == "__main__":
    main()