import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

import numpy

from zookeeper import Field

class DoReFanet(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(DoReFanet, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

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

        input_img = keras.layers.Input(shape = (227, 227, 3))
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
    
    def magnitude_aware_sign_unclipped(x):
        """
        Scaled sign function with identity pseudo-gradient as used for the weights
            in the DoReFa paper. The Scale factor is calculated per layer.
        """
        w = x
        w = w.numpy()
        w = tf.math.abs(w)
        x = tf.convert_to_tensor(w)
        scale_factor =  tf.convert_to_tensor(tf.math.abs(x))
        #scale_factor = tf.reduce_mean(x)
        #scale_factor = 1.3
        #scale_factor =  tf.keras.layers.Lambda(tf.stop_gradient(tf.reduce_mean(tf.abs(x))))
        #scale_factor =  tf.keras.layers.Lambda(lambda scale_factor : scale_factor)(scale_factor)

        @tf.custom_gradient
        def _magnitude_aware_sign(x):
            return tf.sign(x) * scale_factor, lambda dy: dy
        x = _magnitude_aware_sign(x)
        #x = tf.convert_to_tensor(x)
        return x

    @lq.utils.register_keras_custom_object
    def clip_by_value_activation(x):
        return tf.clip_by_value(x, 0, 1)

    @lq.utils.register_keras_custom_object
    def clip_by_value_activation(x):
        return tf.clip_by_value(x, 0, 1)
    
    activations_k_bit: int = Field(2)
    
    def input_quantizer(self):
        return lq.quantizers.DoReFaQuantizer(k_bit=2.0)

    

    def _cnn(self, input_tensor=None, input_shape=None, activation='relu'):
        
        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )

        """x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), strides=strides, padding='same', 
                                        input_quantizer = 'ste_sign',
                                        kernel_quantizer = 'ste_sign',
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        )(img_input)"""
        x = lq.layers.QuantConv2D(filters = 96, kernel_size = (12,12), 
                                        strides=(4,4), padding='valid', 
                                        use_bias=False,
                                        name='conv1')(img_input)                                

        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (5,5), 
                                        strides=(1,1), 
                                        padding='same', 
                                        input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint=None,use_bias=False,
                                        name='conv2')(x)                                
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint=None,use_bias=False,
                                        name='conv3')(x)                                
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)


        x = lq.layers.QuantConv2D(filters = 384, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint=None,use_bias=False,
                                        name='conv4')(x)                                
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
        
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
                                        kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
                                        kernel_constraint=None,
                                        use_bias=False,
                                        name='conv5')(x)                                
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = tf.keras.layers.Flatten()(x)
        x = lq.layers.QuantDense(
            4096,
            input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
            kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)

        x = lq.layers.QuantDense(
            4096,
            input_quantizer=lq.quantizers.DoReFaQuantizer(k_bit=2.0),
            kernel_quantizer=lq.quantizers.DoReFa(mode="weights"),
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-4)(x)

        x =  tf.keras.layers.Activation("clip_by_value_activation")(x)
        x = tf.keras.layers.Dense(self.nClasses, activation='softmax', name='fc1')(x)

        return x


    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
