import os
import numpy as np
import matplotlib.image as mpimg
#import pandas as pd

'''
def load_dsv(filename):
    df = pd.read_csv(filename, sep=":", names= ['Path','Class'])
    return df
'''

def get_training_data(path):
   filename = os.path.join(path, 'truth.dsv')
   #df = load_dsv(filename)
   #print(df)
   
   f = open(filename, 'r')
   files = f.readlines()
   for i in range(len(files)):
       files[i] = files[i][:-1]
   #print(files)
   
   file_size = len(mpimg.imread(os.path.join(path, files[0][:-2])).flatten())
   Y = []
   X = np.zeros((len(files), file_size))
   
   for index in range(len(files)):
       p, c = files[index].split(':')
       #print(p, c)
       img_path = os.path.join(path, p)
       img = mpimg.imread(img_path).flatten()
       Y.append(c)
       for i in range(file_size):
           X[index][i] = img[i]
   
       
   return X, Y

def get_testing_data(path):
   dir_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
   #if 'classification.dsv' in files:
       #files.remove('classification.dsv')
   #df = pd.DataFrame(files, columns=(['Path']))
   #print(df)
   files = []
   for file in dir_files:
       if file.endswith('.png'):
           files.append(file)
   #print(files)
           
   
   file_size = len(mpimg.imread(os.path.join(path, files[0])).flatten())
   X = np.zeros((len(files), file_size))
   
   for index in range(len(files)):
       img_path = os.path.join(path, files[index])
       img = mpimg.imread(img_path).flatten()
       for i in range(file_size):
           X[index][i] = img[i]
           
   return X, files