import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
from keras.layers import AvgPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

from larq_zoo.core import utils

"""from larq_zoo.core import utils
from larq_zoo.core.model_factory import ModelFactory"""


class RealtoBi34(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(RealtoBi34, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

        self.nClasses = nClasses
        self.l_rate = l_rate
        self.units = fc_units
        self.kernel_regularizer = None
        self.scaling_r = 8
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
            beta_1=0.99,
            beta_2=0.9999,
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

        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (7,7), strides=(2,2), padding='same', 
                                        input_quantizer = None,
                                        kernel_quantizer = 'ste_sign',
                                        kernel_constraint = 'weight_clip',
                                        kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        )(img_input)
        """x = lq.layers.QuantConv2D(filters = 64, kernel_size = (7,7), 
                                        strides=(2,2), padding='same', 
                                        kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv1')(img_input)"""                                
        x = BatchNormalization(momentum = 0.99)(x)
        x = Activation('relu')(x)
        residual = x = MaxPool2D(pool_size = (2,2), strides=(2,2), name='pool1')(x)

        #Block1
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                         name='conv2')(x)
        x = self._scale_binary_conv_output(residual, x)                                
        
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv3')(x)                                
        
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)
        #x = residual + x
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv4')(x)                                
        
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv5')(x)                                
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv6')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 64, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv7')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block2
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv8')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(128, (1, 1), strides=1, padding='same',
                                        use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv9')(x)                                
        x = self._scale_binary_conv_output(residual, x)        
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv10')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv11')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv12')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv13')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv14')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 128, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv15')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block3
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv16')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(256, (1, 1), strides=1, padding='same',
                                        use_bias=False)(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv17')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv18')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv19')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv20')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv21')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv22')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)


        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv23')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv24')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv25')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv26')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 256, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv27')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #block4
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(2,2), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv28')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        residual = AvgPool2D((2,2), strides=2, padding="same")(residual)
        residual = lq.layers.QuantConv2D(512, (1, 1), strides=1, padding='same',
                                         use_bias=False)(residual)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv29')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv30')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv31')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv32')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = BatchNormalization(momentum = 0.99)(x)
        x = lq.layers.QuantConv2D(filters = 512, kernel_size = (3,3), 
                                        strides=(1,1), padding='same', 
                                        input_quantizer = 'ste_sign',
                                         kernel_quantizer = 'ste_sign',
                                         kernel_constraint = 'weight_clip',
                                         kernel_initializer="glorot_normal",
                                         use_bias=False,
                                        name='conv33')(x)                                
        x = self._scale_binary_conv_output(residual, x)
        x = Activation('relu')(x)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.nClasses, activation='sigmoid', kernel_initializer="he_normal", name='fc1')(x)

        return x


    class LearnedRescaleLayer(tf.keras.layers.Layer):
        """Implements the learned activation rescaling XNOR-Net++ style.
        This is used to scale the outputs of the binary convolutions in the Strong
        Baseline networks. [(Bulat & Tzimiropoulos,
        2019)](https://arxiv.org/abs/1909.13863)
        """

        def __init__(
            self,
            
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            

        def build(self, input_shapes):
            self.scale_h = self.add_weight(
                name="scale_h",
                shape=(input_shapes[1], 1, 1),
                initializer="ones",
                trainable=True,
            )
            self.scale_w = self.add_weight(
                name="scale_w",
                shape=(1, input_shapes[2], 1),
                initializer="ones",trainable=True,
            )
            self.scale_c = self.add_weight(
                name="scale_c",
                shape=(1, 1, input_shapes[3]),
                initializer="ones",
                trainable=True,
            )

            super().build(input_shapes)

        def call(self, inputs, **kwargs):
            return inputs * (self.scale_h * self.scale_w * self.scale_c)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            return {
                **super().get_config(),
                "regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            }

    def _scale_binary_conv_output(
        self, conv_input: tf.Tensor, conv_output: tf.Tensor
    ) -> tf.Tensor:
        """Data-dependent convolution scaling.
        Scales the output of the convolution in the (squeeze-and-excite
        style) data-dependent way described in Section 4.3 of Martinez at. al.
        """
        in_filters = conv_input.shape[-1]
        out_filters = conv_output.shape[-1]

        z = utils.global_pool(conv_input)
        dim_reduction = tf.keras.layers.Dense(
            int(in_filters // self.scaling_r),
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            
            use_bias=False,
        )(z)
        dim_expansion = tf.keras.layers.Dense(
            out_filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            
            use_bias=False,
        )(dim_reduction)
        scales = tf.keras.layers.Reshape(
            (1, 1, out_filters)
        )(dim_expansion)

        return tf.keras.layers.Multiply()(
            [conv_output, scales]
        )
    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
