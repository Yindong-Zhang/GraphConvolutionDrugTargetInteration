from tensorflow.python.keras.losses import mean_squared_error
import tensorflow as tf


def loop_dataset(model, dataset, optimizer= None, print_every = 32):
    mean_loss = 0
    for it, (mols, props) in enumerate(dataset):
        with tf.GradientTape() as tape:
            props_pred = model(mols)
            loss = mean_squared_error(props_pred, props)

        loss_value = loss.numpy().mean()
        mean_loss = (it * mean_loss + loss_value) / (it + 1)

        if optimizer:
            variables = model.variables
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

        if it % print_every == 0:
            print("%d: loss %.4f." %(it, mean_loss))
    return mean_loss