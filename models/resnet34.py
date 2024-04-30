import config
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Activation, MaxPool2D
import larq as lq
from prettytable import PrettyTable
from models.abstract import ModelWrapper

class ResNet34(ModelWrapper):

    def __init__(self, model_name, working_dir, nClasses,
                 fc_units = 4096,
                 model_name_2 = "checkpoint",
                 logger_lvl = config.log_lvl,
                 l_rate = 0.0008, **kwargs) -> None:
        super(ResNet34, self).__init__(model_name, working_dir, model_name_2, logger_lvl=logger_lvl, **kwargs)

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
                                        kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                        use_bias=False,
                                        )(img_input)"""
        x = self._conv_block(filters=64, kernel_size=(7,7), strides=(4,4), activation='relu', padding='same', name='conv1')(img_input)
        residual = x = MaxPool2D(pool_size = (2,2), strides=(2,2), name='pool1')(x)

        #Block1
        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv2')(x)


        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv3')(x)


        #residual = Activation('relu')(residual)
        #x = residual + x
        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv4')(x)
        
        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv5')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv6')(x)


        x = self._conv_block(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv7')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block2
        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv8')(x)


        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv9')(x)


        residual = lq.layers.QuantConv2D(128, (1, 1), strides=2, padding='same', use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        

        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv10')(x)


        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv11')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv12')(x)


        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv13')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv14')(x)


        x = self._conv_block(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv15')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #Block3
        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv16')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv17')(x)



        residual = lq.layers.QuantConv2D(256, (1, 1), strides=2, padding='same', use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        #residual = Activation('relu')(residual)        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv18')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv19')(x)
        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv20')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv21')(x)
     

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv22')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv23')(x)
        

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv24')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv25')(x)
 

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv26')(x)


        x = self._conv_block(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='conv27')(x)
      

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        #block4
        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv28')(x)

        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv29')(x)


        residual = lq.layers.QuantConv2D(512, (1, 1), strides=4, padding='same', use_bias=False)(residual)
        residual = BatchNormalization()(residual)
        #residual = Activation('relu')(residual)

        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv30')(x)


        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv31')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)

        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv32')(x)


        x = self._conv_block(filters=512, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same', name='conv33')(x)


        x = keras.layers.add([x, residual])
        residual = x = Activation('relu')(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.nClasses, activation='softmax', name='fc1')(x)

        return x

    def _conv_block(self, filters, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name=None):

        def layer_wrapper(inp):
            x = lq.layers.QuantConv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(inp)
            x = BatchNormalization(momentum=0.9)(x)
            x = Activation(activation)(x)
            return x
        
        return layer_wrapper

    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)            
