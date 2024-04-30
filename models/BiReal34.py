import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
from keras.layers import AvgPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

"""from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory"""

class BiReal34(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(BiReal34, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

        self.nClasses = nClasses
        self.l_rate = l_rate
        self.units = fc_units
        self.input_quantizer = "approx_sign"
        self.kernel_quantizer = "magnitude_aware_sign"

        self.model = self._setup_model(verbose = True)


    def _setup_model(self, **kwargs):

        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False

        self.model_name = self.model_name if self.model_name is not None else 'ResNet50'

        input_img = keras.layers.Input(shape = (224, 224, 3))
        cnn = self._cnn(input_tensor = input_img) 

        model = keras.models.Model(input_img, cnn)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.l_rate,
            beta_1=0.8,
            beta_2=0.899,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
        )    

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Recall(top_k=1), tf.keras.metrics.Recall(top_k=5), tf.keras.metrics.Recall(top_k=10)]
        )

        self.model = model
        model.summary()
        if verbose:
            self._display_layers()


        return self.model
    def _cnn(self, input_tensor=None, input_shape=None, activation='relu'):
        
        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )

        shortcut = x = lq.layers.QuantConv2D(filters = 64, kernel_size = (7,7), strides=(2,2), padding='same', 
                                        input_quantizer = None,
                                        kernel_quantizer = 'magnitude_aware_sign',
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        )(img_input)
        """x = lq.layers.QuantConv2D(filters = 64, kernel_size = (7,7), 
                                        strides=(2,2), padding='same', 
                                        kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv1')(img_input)"""                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        residual = x = MaxPool2D(pool_size = (2,2), strides=(2,2), name='pool1')(x)

        #Block1
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                         name='conv2')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv3')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)
        #x = residual + x
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv4')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv5')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv6')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv7')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block2
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv8')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(128, (1, 1), strides=1, padding='same',
                                        use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv9')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)        
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv10')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv11')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv12')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv13')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv14')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv15')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block3
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv16')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(256, (1, 1), strides=1, padding='same',
                                        use_bias=False)(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv17')(x)   
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv18')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv19')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv20')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv21')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv22')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)


        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv23')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv24')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv25')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv26')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv27')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #block4
        b1 = x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv28')(x)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        shortcut = lq.layers.QuantConv2D(512, (1, 1), strides=16, padding='same',
                                         use_bias=False)(shortcut)
        x = keras.layers.add([x, shortcut])

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(512, (1, 1), strides=1, padding='same',
                                         use_bias=False)(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv29')(x + shortcut)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv30')(x + shortcut)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv31')(x + shortcut)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv32')(x + shortcut)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'approx_sign',
                                         kernel_quantizer = 'magnitude_aware_sign',
                                         kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv33')(x + shortcut)                                
        x = BatchNormalization(momentum = 0.8)(x)
        x = Activation('relu')(x)

        #x = tf.keras.layers.Concatenate()([x, b1])
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x + shortcut)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.nClasses, activation='softmax', name='fc1')(x)

        return x


    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
