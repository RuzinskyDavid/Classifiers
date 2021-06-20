import sys, os
from datetime import datetime
import get_data as gd
from Bayes import Bayes
from KNN import KNN


'''
Solutions inspired by Jason Brownlee's Naive Bayes and k-NN
algorithms from scratch. Can be found at:
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
'''

USAGE = ("Usage: classifier.py [-h] [-k K] [-b] [-o name] train_path test_path")

def parse(argv):
    
    if '-h' in argv or '--help' in argv or len(argv) < 2:
        print(USAGE)
        sys.exit(0)
    KNN = False
    K = 0
    BAYES = False
    output_path = 'classification.dsv'

    if '-k' in argv:
        KNN = True
        K = int(argv[argv.index('-k') + 1])
    if '-b' in argv:
        BAYES = True
    if not (KNN or BAYES):
        print('No classifier selected')
        sys.exit(0)

    train_path = argv[len(argv)-2]
    test_path = argv[len(argv)-1]
    
    if '-o' in argv:
        output_path = argv[argv.index('-o') + 1]
    else:
        output_path = os.path.join(test_path, output_path)
        
    parsed = [KNN, K, BAYES, output_path, train_path, test_path]
    
    return parsed

if __name__ == "__main__":
    
    
   args = parse(sys.argv[1:])
   
   '''
   print(' KNN: ', args[0], '\n',
       'K: ', args[1], '\n', 
       'BAYES: ', args[2], '\n',
       'Out_path: ', args[3], '\n',
       'Train_path: ', args[4],
       '\n', 'Test_path: ', args[5], '\n')
   '''
   
   X, Y = gd.get_training_data(args[4])
   
   if args[2]:
       model = Bayes()
   elif args[0]:
       k = args[1]
       if k == 0:
           k = 3
       model = KNN(k)
       
       
   #t0 = datetime.now()
   #print("Starting training")
   model.fit(X, Y)
   #print("Training time:", (datetime.now() - t0))
   
   #t0 = datetime.now()
   #Z = model.predict(X)
   #print("Train accuracy:", model.score(X, Y, Z))
   ##print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y))
   
   X, df = gd.get_testing_data(args[5])
   
   #t0 = datetime.now()
   #print("Starting testing")
   Z = model.predict(X)
   #print("Test accuracy:", model.score(X, Y, Z))
   #print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y))
   
   open(args[3], 'w').close()
   f = open(args[3], 'a')
   
   if f is not None:
       for index in range(len(df)):
           f.write(df[index] + ':' + str(Z[index]) + '\n')
           
   f.close()
   
   
   
   
   
   
   
   
   
   