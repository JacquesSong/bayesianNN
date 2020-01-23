import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import seaborn as sns
import os
import pandas as pd
import warnings

from matplotlib.backends import backend_agg
from tensorflow.contrib.learn.python.learn.datasets import mnist
from absl import flags

import tensorflow_probability as tfp

warnings.simplefilter(action="ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float("learning_rate",
                   default=0.005,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=200,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getcwd(), "data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir",
                    default=os.path.join(os.getcwd(), "frequentist"),
                    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=100,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("print_steps",
                     default=100,
                     help="Frequency at which print logs")
flags.DEFINE_integer("run_steps",
                     default=50,
                     help="Number of graph runs to bootstrap weights probabilities")
flags.DEFINE_bool("fake_data",
                  default=True,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.
      Args:
        names: A Python `iterable` of `str` variable names.
        qm_vals: A Python `iterable`, the same length as `names`,
          whose elements are Numpy `array`s, of any shape, containing
          posterior means of weight variables.
        qs_vals: A Python `iterable`, the same length as `names`,
          whose elements are Numpy `array`s, of any shape, containing
          posterior standard deviations of weight varibles.
        fname: Python `str` filename to save the plot to.
      """
    fig = plt.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)  # plot posterior means
    for n, qm in zip(names, qm_vals):
        sns.distplot(qm.flatten(), ax=ax, label=n)
    ax.set_title("weight means")
    ax.set_xlim([-.2, .2])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)  # plot posterior standard deviation
    for n, qs in zip(names, qs_vals):
        sns.distplot(qs.flatten(), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, .4])

    fig.tight_layout()  # print figures
    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=""):
    """Save a PNG plot visualizing posterior uncertainty on n first heldout data.
  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
    fig = plt.Figure(figsize=(9, 3 * n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3 * i + 1)  # plot heldout input images
        ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3 * i + 2)  # plot barplots
        for prob_sample in probs:
            sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3 * i + 3)
        sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs")
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))


def build_input_pipeline(mnist_data, batch_size, heldout_size, noisy_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))
    training_batches = training_dataset.shuffle(
        50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = tf.data.make_one_shot_iterator(training_batches)

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
         np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).
                      repeat().batch(heldout_size))
    heldout_iterator = tf.data.make_one_shot_iterator(heldout_frozen)

    # Build a iterator over the noisy heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    noisy_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.noisy.images,
         np.int32(mnist_data.noisy.labels)))
    noisy_frozen = (noisy_dataset.take(noisy_size).
                      repeat().batch(noisy_size))
    noisy_iterator = tf.data.make_one_shot_iterator(noisy_frozen)

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator, noisy_iterator


def build_fake_data(num_examples=10):
    """Build fake MNIST-style data for unit testing."""

    class Dummy(object):
        pass

    mnist_data = Dummy()
    true_mnist = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

    # set-up training data
    mnist_data.train = Dummy()
    mnist_data.train.images = true_mnist.train.images
    mnist_data.train.labels = true_mnist.train.labels
    mnist_data.train.num_examples = len(mnist_data.train.images)

    # set-up validation data
    mnist_data.validation = Dummy()
    mnist_data.validation .images = true_mnist.validation.images[0:num_examples]
    mnist_data.validation.labels = true_mnist.validation.labels[0:num_examples]
    mnist_data.validation.num_examples = len(mnist_data.validation.images)

    # set-up noisy data
    mnist_data.noisy = Dummy()
    mnist_data.noisy.images = np.float32(np.random.randn(  # create random images
        num_examples, *IMAGE_SHAPE))
    mnist_data.noisy.labels = np.int32(np.random.permutation(  # create random labels
        np.arange(num_examples)))
    mnist_data.noisy.num_examples = num_examples
    return mnist_data  # fake dataset


def main(argv):
    del argv  # unused

    # setup saving directories
    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.logging.warning(
            "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    # setup iterators
    if FLAGS.fake_data:
        mnist_data = build_fake_data()
    else:
        mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

    (images, labels, handle,
     training_iterator, heldout_iterator, noisy_iterator) = build_input_pipeline(
        mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples, mnist_data.noisy.num_examples)

    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
    with tf.name_scope("frequentist_neural_net", values=[images]):
        neural_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4,  # optimal is 6
                                   kernel_size=3,  # optimal is 5
                                   padding="SAME",
                                   activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding="SAME"),
            tfp.layers.Convolution2DFlipout(16,
                                            kernel_size=3,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding="SAME"),
            tf.keras.layers.Conv2D(16,  # optimal is 120
                                   kernel_size=3,  # optimal is 5
                                   padding="SAME",
                                   activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),  # optimal is 84
            tf.keras.layers.Dense(10)
        ])

        logits = neural_net(images)
        labels_pred = tfp.distributions.Categorical(logits=logits)

    # Compute the cross-entropy (same as negative log-likelihood), averaged over the batch size.
    cross_entropy = -tf.reduce_mean(input_tensor=labels_pred.log_prob(labels))

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight for layers with weight distributions
    # for later visualization.
    indices = []
    names = []
    for i, layer in enumerate(neural_net.layers):
        try:
            layer.get_weights()[0]
        except IndexError:
            continue
        indices.append(i)
        names.append("Layer {}".format(i))

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(cross_entropy)

    # model wide variables
    weights_moments = np.empty((FLAGS.run_steps, FLAGS.max_steps // FLAGS.viz_steps, len(indices), 2))
    probs_heldout = np.empty((FLAGS.run_steps, FLAGS.max_steps // FLAGS.viz_steps, 10, 10))
    probs_noisy = np.empty((FLAGS.run_steps, FLAGS.max_steps // FLAGS.viz_steps, 10, 10))

    for run_index in range(FLAGS.run_steps):

        print("Begin run {:03d}".format(run_index))

        # initialize graph and set all weights to zero (?)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)

            # Run the training loop.

            # build handles
            train_handle = sess.run(training_iterator.string_handle())
            heldout_handle = sess.run(heldout_iterator.string_handle())
            noisy_handle = sess.run(noisy_iterator.string_handle())

            # train network on training set
            for step in range(FLAGS.max_steps):

                # update CNN
                _ = sess.run([train_op, accuracy_update_op],
                             feed_dict={handle: train_handle})

                # show logs every 10 steps
                if step % FLAGS.print_steps == 0:
                    loss_value, accuracy_value = sess.run(
                        [cross_entropy, accuracy], feed_dict={handle: train_handle})
                    print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                        step, loss_value, accuracy_value))

                # store data for visual
                if step == 0 or (step + 1) % FLAGS.viz_steps == 0:

                    # plot probs for last model only
                    probs_heldout[run_index, (step + 1) // FLAGS.viz_steps - 1] = np.array(sess.run(labels_pred.probs, feed_dict={handle: heldout_handle}))

                    # noisy data
                    probs_noisy[run_index, (step + 1) // FLAGS.viz_steps - 1] = np.array(sess.run(labels_pred.probs, feed_dict={handle: noisy_handle}))

                    # store weight moments for each model
                    nn_moments = np.empty((len(indices), 2))
                    for position, index in enumerate(indices):
                        layer = neural_net.layers[index]
                        try:
                            weights = layer.get_weights()[0]
                            nn_moments[position, :] = np.array([np.mean(weights), np.std(weights)])
                        except IndexError:
                            continue
                    weights_moments[run_index, (step + 1) // FLAGS.viz_steps - 1, :, :] = nn_moments
            image_vals, label_vals = sess.run((images, labels),
                                              feed_dict={handle: heldout_handle})
            image_noisy, label_noisy = sess.run((images, labels),
                                              feed_dict={handle: noisy_handle})

    probs_heldout = np.transpose(probs_heldout, axes=(1, 0, 2, 3))

    for t, probs in enumerate(probs_heldout):
        plot_heldout_prediction(image_vals, probs,
                                fname=os.path.join(
                                    FLAGS.model_dir,
                                    "step{:05d}_pred.png".format((t + 1) * FLAGS.viz_steps - 1)))

    probs_noisy = np.transpose(probs_noisy, axes=(1, 0, 2, 3))

    for t, probs in enumerate(probs_noisy):
        plot_heldout_prediction(image_noisy, probs,
                                fname=os.path.join(
                                    FLAGS.model_dir,
                                    "step{:05d}_noisy.png".format((t + 1) * FLAGS.viz_steps - 1)))

    weights_moments = np.transpose(weights_moments, axes=(1, 3, 2, 0))

    for t, value_at_time in enumerate(weights_moments):
        pd.DataFrame(value_at_time[0]).to_csv(os.path.join(FLAGS.model_dir,
                                                         "step{:05d}_means.csv").format((t + 1) * FLAGS.viz_steps - 1))
        pd.DataFrame(value_at_time[1]).to_csv(os.path.join(FLAGS.model_dir,
                                                         "step{:05d}_stds.csv").format((t + 1) * FLAGS.viz_steps - 1))
        plot_weight_posteriors(names, value_at_time[0], value_at_time[1],
                               fname=os.path.join(
                                   FLAGS.model_dir,
                                   "step{:05d}_weights.png".format((t + 1) * FLAGS.viz_steps - 1)))




    return 0


if __name__ == '__main__':
    tf.app.run()
