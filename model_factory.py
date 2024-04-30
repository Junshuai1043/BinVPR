import experiment_presets as E
import config
import os

def get_model(
            model_name,
            training_classes,
            l_rate=1e-4, 
            resume = False, 
            model_save_dir = os.path.join('.','output','trained_models')
            ):
    
    
    model = None
    
    if model_name == E.FNet:

        from models.Lce_HybridShallow import QuantizedHShallow as floppynet_wrapper
    
        #Instantiate a model wrapper
        model = floppynet_wrapper(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 256,
            filters = (96,256,256),
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, #BNN
            kernel_precision = 1, #BNN
            enable_history = False,
            clean_history = not resume,
            optimizer = None, #Adam will be used with the l_rate as a learning rate
            loss = 'categorical_crossentropy'
            )
        
    elif model_name == E.SNet:
        
        from models.Lce_shallowNet import QuantizedShallow

        model = QuantizedShallow(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history = False
            )        
    
    elif model_name == E.ANet:
        
        from models.Z_AlexNet import AlexNet
        
        model = AlexNet(
            model_name = model_name, 
            working_dir =  model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            enable_history= False
            )
    
    elif model_name == E.BNet:
        
        from models.Lce_AlexNet import QuantizedAlexNet as BinAlex
        
        model = BinAlex(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history = False
            )
        
   
    elif model_name == E.Res34:
        
        from models.Resnet34 import ResNet34

        model = ResNet34(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )  

    elif model_name == E.Res18:
        
        from models.resnet18 import ResNet18

        model = ResNet18(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )  


    elif model_name == E.BiReal34:
        
        from models.BiReal34 import BiReal34

        model = BiReal34(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )  
    

    
    elif model_name == E.BinVPR18:
        
        from models.BinVPR18 import BinRes18

        model = BinRes18(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )  

    elif model_name == E.BinVPR34:
        
        from models.BinVPR34 import BinRes34

        model = BinRes34(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )  
    
    elif model_name == E.BiRes18:
        
        from models.Lce_resnet18 import BiRes18

        model = BiRes18(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )

    elif model_name == E.BiRes34:
        
        from models.Lce_resnet34 import BiRes34

        model = BiRes34(
            model_name = model_name,
            working_dir = model_save_dir,
            model_name_2= 'model',
            logger_lvl=config.log_lvl,
            nClasses=training_classes,
            fc_units=4096,
            l_rate=l_rate,
            save_weight_only = True,
            enable_moitoring = True,
            tensorboard = True,
            enable_history = False
        )    

    
    elif model_name == E.DoReFaNet:
        
        from models.DoReFaNet import DoReFanet
        
        model = DoReFanet(
            model_name = model_name, 
            working_dir =  model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            enable_history= False
            ) 

    elif model_name == E.XNOR:
        
        from models.XNOR import XNOR
        
        model = XNOR(
            model_name = model_name, 
            working_dir =  model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            enable_history= False
            )

    
    elif model_name == E.RealToBin34:
        
        from models.RealToBi34 import RealtoBi34
        
        model = RealtoBi34(
            model_name = model_name, 
            working_dir =  model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            enable_history= False
            )
    
    else:
        raise ValueError(f"Invalid model name (preset): {model_name}")


    return model

