import tensorflow as tf

def DiscriminatorModel():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(7,(3,3),padding="same",input_shape=(28,28,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(50,activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    return model

def DiscriminatorLoss(real_pred,fake_pred):
    real_pred=tf.sigmoid(real_pred)
    fake_pred=tf.sigmoid(fake_pred)
    real_loss=tf.losses.binary_crossentropy(tf.ones_like(real_pred),real_pred)
    fake_loss=tf.losses.binary_crossentropy(tf.zeros_like(fake_pred),fake_pred)
    return fake_loss+real_loss