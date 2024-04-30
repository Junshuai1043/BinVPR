'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

Datasets specifications.

'''


###########################
#### TRAINING DATASETS ####
###########################


PLACES365 = 'places365'
training_datasets = dict()


training_datasets[PLACES365] = dict()
#training_datasets[PLACES365]['training_path'] = r"/home/wjs/Datasets/GardenPoins Walking"
#training_datasets[PLACES365]['validation_path'] = r"/home/wjs/Datasets/GardenPoins Walking"
training_datasets[PLACES365]['training_path'] = r"/home/wjs/Datasets/forests/train"
training_datasets[PLACES365]['validation_path'] = r"/home/wjs/Datasets/forests/val"
training_datasets[PLACES365]['test_path'] = r"/home/wjs/Datasets/forests/val"
#training_datasets[PLACES365]['test_path'] = r"/home/wjs/Datasets/GardenPoins Walking/ground_truth_new.npy"

training_datasets[PLACES365]['nClass'] = 5

