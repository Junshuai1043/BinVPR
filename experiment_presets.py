import os
import dataset_presets as D
import copy

########################################
### working directory for presets ######
###  Change it for your convenience ####
########################################
model_save_dir = os.path.join('.','output','trained_models')
#################################################################


experiments = dict()
#########################
### TRO Paper presets ###
#########################



params = dict()

FNet = 'floppynet'
params['model_name'] = FNet
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'operation_modes'
experiments[FNet] = copy.copy(params)


params = dict()
SNet = 'shallownet'
params['model_name'] = SNet
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['val_split'] = 0.4
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.shallowNet'
experiments[SNet] = copy.copy(params)

params = dict()
BNet = 'binarynet'
params['model_name'] = BNet
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-4
params['batch_size'] = 32
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.binaryNet'
experiments[BNet] = copy.copy(params)

params = dict()
ANet = 'alexnet'
params['model_name'] = ANet
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['val_split'] = 0.5
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.alexNet'
experiments[ANet] = copy.copy(params)


params = dict()
VNet = 'vggnet'
params['model_name'] = VNet
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-5
params['batch_size'] = 32
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[VNet] = copy.copy(params)

Res34 = 'resnet34'
params['model_name'] = Res34
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-4
params['batch_size'] = 16
params['val_split'] = 0.2
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[Res34] = copy.copy(params)

Res18 = 'resnet18'
params['model_name'] = Res18
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 8e-5
params['batch_size'] = 8
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[Res18] = copy.copy(params)

BiReal34 = 'bireal34'
params['model_name'] = BiReal34
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['batch_size'] = 64
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[BiReal34] = copy.copy(params)

BinVPR18 = 'BinVPR18'
params['model_name'] = BinVPR18
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-5
params['batch_size'] = 16
params['val_split'] = 0.4
params['epochs'] = 400
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[BinVPR18] = copy.copy(params)

BinVPR34 = 'BinVPR34'
params['model_name'] = BinVPR34
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 3e-5
params['batch_size'] = 16
params['val_split'] = 0.4
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[BinVPR34] = copy.copy(params)

BiRes18 = 'bires18'
params['model_name'] = BiRes18
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-5
params['batch_size'] = 16
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[BiRes18] = copy.copy(params)

BiRes34 = 'bires34'
params['model_name'] = BiRes34
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-4
params['batch_size'] = 16
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
experiments[BiRes34] = copy.copy(params)

DoReFaNet = 'dorefanet'
params['model_name'] = DoReFaNet
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-4
params['batch_size'] = 24
params['val_split'] = 0.4
params['epochs'] = 100
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.alexNet'
experiments[DoReFaNet] = copy.copy(params)

params = dict()
XNOR = 'xnornet'
params['model_name'] = XNOR
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['val_split'] = 0.4
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.alexNet'
experiments[XNOR] = copy.copy(params)

RealToBin34 = 'realtobin34'
params['model_name'] = RealToBin34
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 8e-4
params['batch_size'] = 12
params['val_split'] = 0.4
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.alexNet'
experiments[RealToBin34] = copy.copy(params)
