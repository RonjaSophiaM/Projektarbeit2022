if __name__ == '__main__':
    import numpy as np
    import random
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    #!pip3
    #install - -upgrade
    #imutils
#    from imutils import paths
    import cv2
    from keras.preprocessing.image import ImageDataGenerator
    import os
    from PIL import ImageFile

    from keras.models import Sequential
    from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, concatenate
    from tensorflow.keras import layers

    from sklearn.utils import class_weight

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    import keras
    from tensorflow.keras.optimizers import Adam
    print("1")
    df_gt = pd.read_csv("D:/Melanoma_Classification/ISIC_2019_Training_GroundTruth.csv")
    #df_gt.sample(5)

    df_md = pd.read_csv("D:/Melanoma_Classification/ISIC_2019_Training_Metadata.csv")
    df_md['Melanoma'] = df_gt['MEL']# ,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK # gut: NV, BKL, DF, VASC, UNK # böse: MEL, BCC, SCC, AK
    #, 'BCC', 'SCC', 'AK'
    df_md['MelanocyticNevus'] = df_gt['NV']
    df_md['BasalCellCarcinoma'] = df_gt['BCC']
    df_md['ActinicKeratosis'] = df_gt['AK']
    df_md['BenignKeratosis'] = df_gt['BKL']
    df_md['Dermatofibroma'] = df_gt['DF']
    df_md['VascularLesion'] = df_gt['VASC']
    df_md['SquamousCellCarcinoma'] = df_gt['SCC']
    df_md['Others'] = df_gt['UNK']
    #df_md.sample(5)
    #print(df_md['target'])
    print("2")

    # Preparing directories for DataImage Generator
    os.makedirs('D:/Melanoma_Classification/Training_Data/Malignant')
    os.makedirs('D:/Melanoma_Classification/Test_Data/Malignant')
    os.mkdir('D:/Melanoma_Classification/Test_Data/NotMalignant')
    os.mkdir('D:/Melanoma_Classification/Training_Data/NotMalignant')

    # Function to populate train and test directories
    import shutil
    import sys


    def Make_Dir(src_path, dst_path, target, Data):

        Labels = Data[['image', target]]
        for imagename, target in Labels.values:
            src = src_path + '/' + imagename + '.jpg'

            if target:

                try:
                    shutil.copy(src, dst_path)
                    # print("sucessfully copied " + imagename + ' from src ' + src_path + " to dst " + dst_path)
                except IOError as e:
                    print("Unable to copy file {} to {}".format(src_path, dst_path))
                    break
                except:
                    print("when try copy file {} to {}, unexpected error: {}".format(src, dst_path, sys.exc_info()))
                    break




    # --------- Melanoma diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('Melanoma', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/Malignant',
             target = 'Melanoma',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/Malignant',
             target='Melanoma',
             Data=Test_data)

    # --------- MelanocyticNevus diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('MelanocyticNevus', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/NotMalignant',
             target = 'MelanocyticNevus',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/NotMalignant',
             target='MelanocyticNevus',
             Data=Test_data)

    # --------- BasalCellCarcinoma diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('BasalCellCarcinoma', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/Malignant',
             target = 'BasalCellCarcinoma',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/Malignant',
             target='BasalCellCarcinoma',
             Data=Test_data)

    # --------- ActinicKeratosis diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('ActinicKeratosis', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/Malignant',
             target = 'ActinicKeratosis',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/Malignant',
             target='ActinicKeratosis',
             Data=Test_data)

    # --------- BenignKeratosis diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('BenignKeratosis', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/NotMalignant',
             target = 'BenignKeratosis',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/NotMalignant',
             target='BenignKeratosis',
             Data=Test_data)

    # --------- Dermatofibroma diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('Dermatofibroma', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/NotMalignant',
             target = 'Dermatofibroma',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/NotMalignant',
             target='Dermatofibroma',
             Data=Test_data)

    # --------- VascularLesion diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('VascularLesion', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/NotMalignant',
             target = 'VascularLesion',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/NotMalignant',
             target='VascularLesion',
             Data=Test_data)

    # --------- SquamousCellCarcinoma diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('SquamousCellCarcinoma', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/Malignant',
             target = 'SquamousCellCarcinoma',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/Malignant',
             target='SquamousCellCarcinoma',
             Data=Test_data)

    # --------- Other diagnose in Ordner einsortieren ----------

    # Daten aufteilen in je ein Datenset für das Training und für das Testen
    Test_data = df_md.groupby('Others', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))
    Training_data = df_md.drop(Test_data.index)

    Make_Dir(src_path = 'D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path = 'D:/Melanoma_Classification/Training_Data/NotMalignant',
             target = 'Others',
             Data = Training_data)
    Make_Dir(src_path='D:/Melanoma_Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
             dst_path='D:/Melanoma_Classification/Test_Data/NotMalignant',
             target='Others',
             Data=Test_data)

    print("finished")