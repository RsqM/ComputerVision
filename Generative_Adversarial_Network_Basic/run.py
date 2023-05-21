import tensorflow as tf
import numpy as np

from DiscriminatorImport import DiscriminatorLoss, DiscriminatorModel
from GeneratorImport import GeneratorLoss, GeneratorModel

genopt=tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer=tf.optimizers.Adam(1e-3)

generator = GeneratorModel()
discriminator = DiscriminatorModel()

batch_size=100

def TrainingStep(images):
    noise=np.random.randn(batch_size,100).astype("float32")
    with tf.GradientTape() as generatorTape,tf.GradientTape() as discriminatorTape:
        generated_images=generator(noise)
        realOutput=discriminator(images)
        fakeOutput=discriminator(generated_images)
        
        #GenLoss = Sigmoid ---> BinaryCrossEntropy -----> Returns Loss
        generatorLoss=GeneratorLoss(fakeOutput)
        discriminatorLoss=DiscriminatorLoss(realOutput,fakeOutput)
        
        #BackProp - step1 - Pull Gradient Tapes from the model.
        #use the tapes to find the loss using above. Update gradients
        #Back prop and zip gradients over the model.
        gradientsGenerator=generatorTape.gradient(generatorLoss,generator.trainable_variables)
        gradientsDiscriminator=discriminatorTape.gradient(discriminatorLoss,discriminator.trainable_variables)
        
        #apply gradients in the Generator and Discriminator networks.
        genopt.apply_gradients(zip(gradientsGenerator,generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradientsDiscriminator,discriminator.trainable_variables))
        
        print("Generator Loss:",np.mean(generatorLoss),"\n")
        print("Discriminator Loss:",np.mean(discriminatorLoss),"\n")

def Train(dataset,epochs):
    for _ in range(epochs):
        for images in dataset:
            images=tf.cast(images,tf.dtypes.float32)
            TrainingStep(images)


(train_X,train_y),(test_X,test_y)=tf.keras.datasets.mnist.load_data()

train_X=train_X.reshape(train_X.shape[0],28,28,1)
train_X=(train_X-127.5)/127.5

Buffer=train_X.shape[0]
train_X=tf.data.Dataset.from_tensor_slices(train_X).shuffle(Buffer).batch(100)

#call Train
Train(train_X,1)