import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import seaborn as sns
import os
import time
import warnings

from matplotlib.backends import backend_agg
from tensorflow.contrib.learn.python.learn.datasets import mnist
from absl import flags

import tensorflow_probability as tfp

warnings.simplefilter(action="ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float("learning_rate",
                   default=0.005,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getcwd(), "data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir",
                    default=os.path.join(os.getcwd(), "bayesian"),
                    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=100,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.
      Args:
        names: A Python `iterable` of `str` variable names.
        qm_vals: A Python `iterable`, the same length as `names`,
          whose elements are Numpy `array`s, of any shape, containing
          posterior means of weight varaibles.
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
    ax.set_xlim([-.5, .5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)  # plot posterior standard deviation
    for n, qs in zip(names, qs_vals):
        sns.distplot(qs.flatten(), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, .15])

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


def build_input_pipeline(mnist_data, batch_size, heldout_size):
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

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator


def build_fake_data(num_examples=10):
    """Build fake MNIST-style data for unit testing."""

    class Dummy(object):
        pass

    mnist_data = Dummy()
    mnist_data.train = Dummy()
    mnist_data.train.images = np.float32(np.random.randn(
        num_examples, *IMAGE_SHAPE))  # create random image
    mnist_data.train.labels = np.int32(np.random.permutation(
        np.arange(num_examples)))  # create random labeling
    mnist_data.train.num_examples = num_examples
    mnist_data.validation = Dummy()
    mnist_data.validation.images = np.float32(np.random.randn(
        num_examples, *IMAGE_SHAPE))
    mnist_data.validation.labels = np.int32(np.random.permutation(
        np.arange(num_examples)))
    mnist_data.validation.num_examples = num_examples
    return mnist_data  # fake dataset


def main(argv):
    del argv  # unused
    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.logging.warning(
            "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)

    if FLAGS.fake_data:
        mnist_data = build_fake_data()
    else:
        mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

    (images, labels, handle,
     training_iterator, heldout_iterator) = build_input_pipeline(
        mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
    with tf.name_scope("bayesian_neural_net", values=[images]):
        neural_net = tf.keras.Sequential([
            tfp.layers.Convolution2DFlipout(4,  # optimal is 6
                                            kernel_size=3,  # optimal is 5
                                            padding="SAME",
                                            activation=tf.nn.relu),
            # tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
            #                              strides=[2, 2],
            #                              padding="SAME"),
            # tfp.layers.Convolution2DFlipout(16,
            #                                 kernel_size=5,
            #                                 padding="SAME",
            #                                 activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding="SAME"),
            tfp.layers.Convolution2DFlipout(16,  # optimal is 120
                                            kernel_size=3,  # optimal is 5
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(32, activation=tf.nn.relu),  # optimal is 84
            tfp.layers.DenseFlipout(10)
        ])

        logits = neural_net(images)
        labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(
        input_tensor=labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_data.train.num_examples
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for layers with weight distributions
    # for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
        try:
            q = layer.kernel_posterior
        except AttributeError:
            continue
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        for step in range(FLAGS.max_steps):  # train network on training set
            _ = sess.run([train_op, accuracy_update_op],
                         feed_dict={handle: train_handle})

            if step % 100 == 0:  # show logs every 100 steps
                loss_value, accuracy_value = sess.run(
                    [elbo_loss, accuracy], feed_dict={handle: train_handle})
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                    step, loss_value, accuracy_value))

            if step == 0 or (step + 1) % FLAGS.viz_steps == 0:
                # Compute log prob of heldout set by averaging draws from the model:
                # p(heldout | train) = int_model p(heldout|model) p(model|train)
                #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
                # where model_i is a draw from the posterior p(model|train).
                probs = np.asarray([sess.run((labels_distribution.probs),
                                             feed_dict={handle: heldout_handle})
                                    for _ in range(FLAGS.num_monte_carlo)])
                mean_probs = np.mean(probs, axis=0)

                image_vals, label_vals = sess.run((images, labels),
                                                  feed_dict={handle: heldout_handle})
                heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                       label_vals.flatten()]))
                print(" ... Held-out nats: {:.3f}".format(heldout_lp))

                qm_vals, qs_vals = sess.run((qmeans, qstds))

                plot_weight_posteriors(names, qm_vals, qs_vals,
                                       fname=os.path.join(
                                           FLAGS.model_dir,
                                           "step{:05d}_weights.png".format(step)))

                plot_heldout_prediction(image_vals, probs,
                                        fname=os.path.join(
                                            FLAGS.model_dir,
                                            "step{:05d}_pred.png".format(step)),
                                        title="mean heldout logprob {:.2f}"
                                        .format(heldout_lp))


if __name__ == "__main__":
    tf.app.run()

