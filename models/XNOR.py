import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

class XNOR(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(XNOR, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

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

        """x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), strides=strides, padding='same', 
                                        input_quantizer = 'ste_sign',
                                        kernel_quantizer = 'ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        )(img_input)"""
        x = lq.layers.QuantConv2D(filters = 96, kernel_size = (11,11), 
                                        strides=(4,4), padding='same',
                                        input_quantizer=None,
                                        kernel_quantizer=None,
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv1')(img_input)
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3,3), strides=(2,2), padding = 'same',name='pool1')(x)
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)

        #block1
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (5,5), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv2')(x)                                
        x = MaxPool2D(pool_size=(3,3),  strides=(2,2), padding = 'same',name='pool2')(x)
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)


        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv3')(x)                                
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)


        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv4')(x)                                
        x = MaxPool2D(pool_size=(3,3),  strides=(2,2), padding = 'same',name='pool5')(x)
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)

        b1 = x = lq.layers.QuantConv2D(filters = 4096, kernel_size = (6,6), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv5')(x)                                
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)
        
        
        x = lq.layers.QuantConv2D(filters = 4096, kernel_size = (1,1), 
                                        strides=(1,1), padding='valid', 
                                        input_quantizer='ste_sign',
                                        kernel_quantizer='ste_sign',
                                        kernel_constraint ='weight_clip',
                                        use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(5e-7),
                                        name='conv6')(x)                                
        x = BatchNormalization(momentum=0.9, scale=False, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(self.nClasses, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(), name='fc1')(x)

        return x


    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
