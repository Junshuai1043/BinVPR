import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

import training_utility

class BinRes34(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(BinRes34, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

        self.nClasses = nClasses
        self.l_rate = l_rate
        self.units = fc_units
        self.model = self._setup_model(verbose = True)


    def _setup_model(self, **kwargs):

        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False

        self.model_name = self.model_name if self.model_name is not None else 'ResNet50'

        with training_utility.mirrored_strategy.scope():
            input_img = keras.layers.Input(shape = (224, 224, 3))
            cnn = self._cnn(input_tensor = input_img) 

            model = keras.models.Model(input_img, cnn)

            optimizer = tf.keras.optimizers.Adam( 
            learning_rate=self.l_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
            )    

            model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
             )

        self.model = model

        if verbose:
            self._display_layers()


        return self.model
    def _cnn(self, input_tensor=None, input_shape=None, activation='relu'):
        
        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )

        """x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), strides=strides, padding='same', 
                                        input_quantizer = lq.activations.leaky_tanh,
                                        kernel_quantizer = lq.activations.leaky_tanh,
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        )(img_input)"""
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same',
                                        input_quantizer=None,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv1')(img_input)
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)
        lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same',
                                        input_quantizer=None,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv35')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)
        temp = x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same',
                                        input_quantizer=None,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv36')(x)
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)
        #x = MaxPool2D(pool_size=(11,11), strides=(2,2), padding = 'same',name='pool1')(x)

        #Block1
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv2')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        residual1 = tf.keras.layers.AvgPool2D((1,1), strides=2, padding='same')(temp)

        residual1 = lq.layers.QuantConv2D(128, (1, 1), strides=1, padding='same', use_bias=False)(residual1)
        residual1 = BatchNormalization()(residual1)

        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv3')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)

        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv4')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv5')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv6')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv7')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)
        #x = MaxPool2D(pool_size=(11,11), strides=(2,2), padding = 'same',name='pool2')(x)
        
        #block1 = lq.layers.QuantConv2D(256, (1, 1), strides=2, padding='same', use_bias=False)(b1)

        #Block2
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv8')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)

        residual2 = tf.keras.layers.AveragePooling2D((1,1), strides=4, padding='same')(temp)
        residual2 = lq.layers.QuantConv2D(256, (1, 1), strides=1, padding='same', use_bias=False)(residual2)
        residual2 = BatchNormalization()(residual2)

        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv9')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)

        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv10')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=2),
                                        use_bias=False,
                                        name='conv11')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv12')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=2),
                                        use_bias=False,
                                        name='conv13')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv14')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=2),
                                        use_bias=False,
                                        name='conv15')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        
        x = Activation('relu')(x)

        #x = MaxPool2D(pool_size=(11,11), strides=(2,2), padding = 'same',name='pool3')(x)    

        #block1 = lq.layers.QuantConv2D(384, (1, 1), strides=4, padding='same', use_bias=False)(b1)
        #block1 = lq.layers.QuantConv2D(384, (1, 1), strides=2, padding='same', use_bias=False)(block1)
        # + block1
        #block2 = lq.layers.QuantConv2D(384, (1, 1), strides=2, padding='same', use_bias=False)(b2)

        #Block3
        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv16')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)

        #residual3 = tf.keras.layers.AvgPool2D((1,1), strides=8, padding='same')(temp)
        residual3 = lq.layers.QuantConv2D(384, (1, 1), strides=8, padding='same', use_bias=False)(temp)
        residual3 = BatchNormalization()(residual3)


        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv17')(x)   
        x = BatchNormalization(momentum = 0.9)(x)

        x = Activation('relu')(x)


        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv18')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv19')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv20')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)


        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv21')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv22')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)


        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv23')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv24')(x)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv25')(x)   
        x = BatchNormalization(momentum = 0.9)(x)


        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv26')(x + residual3)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv27')(x + residual3)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)
       
        #x = MaxPool2D(pool_size=(11,11), strides=(2,2), padding = 'same',name='pool4')(x)
        
        #block1 = lq.layers.QuantConv2D(512, (1, 1), strides=8, padding='same', use_bias=False)(b1)
        #block2 = lq.layers.QuantConv2D(512, (1, 1), strides=4, padding='same', use_bias=False)(b2)
        #block1 = lq.layers.QuantConv2D(512, (1, 1), strides=2, padding='same', use_bias=False)(block1)
        #block2 = lq.layers.QuantConv2D(512, (1, 1), strides=2, padding='same', use_bias=False)(block2)
        #block1 + block2 +

        #block3 = lq.layers.QuantConv2D(512, (1, 1), strides=2, padding='same', use_bias=False)(b3)

        #block4
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv28')(x + residual3)                                
        x = BatchNormalization(momentum = 0.9)(x)

        #residual4 = tf.keras.layers.AvgPool2D((1,1), strides=16, padding='same')(temp)
        residual4 = lq.layers.QuantConv2D(512, (1, 1), strides=16, padding='same', use_bias=False)(temp)
        residual4 = BatchNormalization()(residual4)


        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv29')(x + residual4)                                
        x = BatchNormalization(momentum = 0.9)(x)

        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv30')(x + residual4)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv31')(x + residual4)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x + residual4)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv32')(x + residual4)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.activations.leaky_tanh,
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        name='conv33')(x + residual4)                                
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x + residual4)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.nClasses, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(), name='fc1')(x)

        return x


    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
