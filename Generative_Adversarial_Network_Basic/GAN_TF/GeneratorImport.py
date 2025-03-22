import tensorflow as tf

def GeneratorModel():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256,input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7,7,256)))
    model.add(tf.keras.layers.Conv2DTranspose(128,(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(1,(3,3),strides=(2,2),padding="same"))
    return model

def GeneratorLoss(fake_pred):
    fake_pred=tf.sigmoid(fake_pred)
    fake_loss=tf.keras.losses.binary_crossentropy(tf.ones_like(fake_pred),fake_pred)
    return fake_loss