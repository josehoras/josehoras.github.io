---
layout: post
title: "LSTM in TensorFlow"
excerpt: "Implementation of a LSTM recurrent neural network using TensorFlow."
author: "Jose"
date: 2019-05-05
---

In this post I tell about how I designed a LSTM recurrent network in TensorFlow.

The goal of this post is not to explain the theory of recurrent networks. There are already amazing posts and resources on that topic that I could not surpass. I specially recommend: 

- Andrej Karpathy's [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Christopher Olah's [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Stanford's [CS231n lecture](https://youtu.be/6niqTuYFZLQ)

Instead in this post I want to give a more practical insight. I'm also doing the same, in two separate posts, for pure Python and Keras. The aim is to have the same program written in three different frameworks to highlight the similarities and differences between them. Also, it may make easier to learn one of the frameworks if you already know some of the others. 

My starting point is Andrej Karpathy code [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086), described in his post linked above. I first modified the code to make a LSTM out of it, using what I learned auditing the CS231n lectures (also from Karpathy). So, I started from pure Python, and then moved to TensorFlow and Keras. You may, however, come here after knowing Keras, or just want to have a practical look at recurrent networks in TensorFlow.

The three frameworks have different philosophies, and I wouldn't say one is better than the other, even for learning. The code in pure Python takes you down to the mathematical details of LSTMs, as it programs the backpropagation explicitly. Keras, on the other side, makes you focus on the big picture of what the LSTM does, and it's great to quickly implement something that works. Going from pure Python to Keras feels almost like cheating. Suddenly everything is so easy and you can focus on what you really need to get your network working. Going from Keras to pure Python feels, I would think, enlightening. You will look under the hood and things that seemed like magic will now make sense.

----
# LSTM in TensorFlow

You find this implementation in the file `tf-lstm-char.py` in the [GitHub repository](https://github.com/josehoras/LSTM-Frameworks)

As in the other two implementations, the code contains only the logic fundamental to the LSTM architecture. I use the file `aux_funcs.py` to place functions that, being important to understand the complete flow, are not part of the LSTM itself. These include functionality for loading the data file, pre-process the data by encoding each character into one-hot vectors, generate the batches of data that we feed to the neural network on training time, and plotting the loss history along the training. These functions are (mostly) reused in the pure Python and Keras versions. I will not explain in detail these auxiliary functions, but the type of inputs that we give to the network and its format will be important.

## Data

The full data to train on will be a simple text file. In the repository I uploaded the collection on Shakespeare works (~4 MB) and the Quijote (~1 MB) as examples. 

We will feed the model with sequences of letters taken in order from this raw data. The model will make its prediction of what the next letter is going to be in each case. To train it will compare its prediction with the true targets. The data and labels we give the model have the form:

<div class="post_img">
<img src="../../../assets/lstm-recurrent-networks/data_batch.png" /> 
</div>

However, we don't give the model the letters as such, because neural nets operate with numbers and one-hot encoded vectors, not characters. To do this we give each character an unique number stored in the dictionary `char_to_idx[]`. Each of these number is a class, and the model will try to see in which class the next character belongs. A neural network outputs the probability of each class, that is, a vector of a length equal to the number of classes, or characters we have. Then it will compare this probability vector with a vector representing the true class, a one-hot encoded vector (that's its name) where the true class has probability 1, and all the rest probability 0. That's the kind of vectors we get from the encode function.

So, when we pass a sequence of **seq_length** characters and encode them in vectors of lengths **vocab_size** we will get matrices of shape (seq_length, vocab_size). It's very important to keep track of the dimensions of your data as it goes from input through the several layers of your network to the output. When we define our model in TensorFlow we have to specify the shape of our input's size. But TensorFlow expects something else, as it is able to do the training using entire batches of the input data at each step. And it actually expects you to feed a batch of data. Although this is pretty cool, we will feed one sequence and its targets at a time to keep it simple. This would be a batch of one element, and the corresponding matrix TensorFlow will have is one of shape **(seq_length, 1, vocab_size)**, 1 being our batch size.

If you have read the Keras implementation you have seen that the matrix shape there is (1, seq_length, vocab_size), batch size being the first dimension. TensorFlow accepts also this format, and it would actually be easier. However, for some reason I began feeding (seq_length, batch_size, vocab_size) data, and that caused a bug in my model. I got crazy until I found the way to solve it, and now I'm making a point to keep it like this. This way I can show you below where to look to keep your dimensions right, and you can learn from my mistakes.

So, in our case we specify (seq_length, vocab_size) and pass a batch of (seq_length, 1, vocab_size).

## Model architecture

After having cleared what kind of inputs we pass to our model, we can look without further delay at the model itself, defined in `keras-lstm-char.py`. Our model is composed of:

- one LSTM layer, that process sequentially the temporal input series (our characters sequence), and outputs a sequence of hidden states
- one dense layer, that transforms each hidden state into a vector of scores or logits for each character in our dictionary
- a softmax transformation that normalizes our logits in probabilities (the sum of probabilities for all characters equals 1)

In TensorFlow we define first the input variables to our model:

```
x = tf.placeholder("float", [None, batch_size, vocab_size])
y = tf.placeholder("float", [None, batch_size, vocab_size])
init_state = tf.placeholder(tf.float32, [2, batch_size, hidden_dim])
```

The "placeholder" keyword means that the tensor is not given any value now. It's values will be fed at training or test time. The first dimension, the length of the input sequence is not fixed, which is denoted by the "None" keyword. "x" and "y" will be our input and target sequences respectively.

Our third input, "init_state" is the initialization given to the LSTM hidden state at the beginning of each training loop or test run. We will initialize it with zeros.

Next we define the parameters in the network layers: 

```
# lstm layer
lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, state_is_tuple=True)
rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
# dense layer parameters
dense_weights = tf.get_variable("out_w", shape=[hidden_dim, vocab_size])
dense_bias = tf.get_variable("out_b", shape=[vocab_size])
```

The "get_variable()" method will create and initialize new variables to contain the weights and biases of the model's dense layers. These are trainable variables that will be modified in the training process to improve the model's predictions. The LSTM layer also has trainable weights and biases, but they are internally defined by the class "LSTMCell()". Additionaly, we provide the initial hidden and cell states with our "init_state" placeholder. However we need to convert it to the correct format using the "LSTMStateTuple()" class.

Finally we define the operations in our model with:

```
h_states, current_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=rnn_tuple_state,
                                           time_major=True, dtype=tf.float32)
logits = tf.matmul(h_states[:, 0, :], dense_weights) + dense_bias
probabilities = tf.nn.softmax(logits, name="probabilities")

```

Now, this is the step where I was not careful the with dimensions and took me a while to debug. In the recurrent operation command, "dynamic_rnn()", you have the option `time_mayor=True`, that tell TensorFlow in the input "x" the first dimension will be the temporal sequence, instead of the batch size. If you pass your input in the format (batch_size, seq_length, vocab_size), you have to set `time_mayor=False`, which is the default actually...

The LSTM layer output is a sequence of states as long as our input sequence. Each of these states should predict the following character. These states are the input to the dense layer. We will now get rid of the batch dimension, that we know is only one. We collapse that dimension passing `h_states[:, 0, :]`.

We express these scores in probabilities, that is, we normalize the series of scores so that their sum is equal to one, with the softmax function.

Once the model architecture is defined, we compare its results with the true targets, obtaining the cross-entropy and the loss:

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy, name="loss")
```

As analogy to the program in pure Python in the previous section, we can say up until here we have defined the forward pass, that can be represented in the following diagram:

<div class="post_img">
<img src="../../../assets/lstm-recurrent-networks/forward_diagram.png" width="550" height="170" /> 
</div>

## Optimization 

The equivalent of our backward pass in pure Python is where TensorFlow really takes matters into its hands. Under the hood the way you optimize a model, to adjust to the ground truth inputs you feed, is to calculate the gradients of the loss with respect to the model parameters and use this values to modify these parameters in a way the loss will be minimized a little on each loop iteration. 

If you haven't really understood the last sentence and you don't care now much the mathematical details, that's totally fine. TensorFlow will do it for you with two lines:

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training = optimizer.minimize(loss, name="training")
```

Here we first define the optimizer, or the algorithm used to minimize the loss. There are several optimization methods. Here we use `AdamOptimizer()`, that works better than the simple Stochastic Gradient Descent (SGD) of the Python version.

With the `minimize()` method, and passing our "loss" as input, we have finally defined the full training flow:

<div class="post_img">
<img src="../../../assets/lstm-recurrent-networks/sgd_diagram.png" width="550" height="300" /> 
</div>

But up to now we have just defined the flow, or Graph in TensorFlow jargon. We haven't begun the training process or done actual calculations yet. We need to open a TensorFlow session to do that.

## Training in a TensorFlow session

The code for our TensorFlow session is: 

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _current_state = np.zeros((2, batch_size, hidden_dim))
    for p in range(4001):
        # show progress
        if p % 100 == 0:
            print('\niter %d, loss: %f' % (p, smooth_loss[-1]))  # print progress
            print(sample(600, sess))
            aux.plot(loss_hist, smooth_loss, it, it_per_epoch, base_name="tensor")

        inputs, targets = (next(data_feed))
        l, _, _current_state = sess.run([loss, training, current_state],
                                        feed_dict={x: inputs,
                                                   y: targets,
                                                   init_state: _current_state})
        loss_hist.append(l)
        smooth_loss.append(smooth_loss[-1] * 0.999 + loss_hist[-1] * 0.001)
```

The session begins by initializing weights, biases and internal variables of our model. We also define our first states that will be passed in our "init_state" input. With that we enter into the training loop.

First, we check for every 100 steps to print out information of the current status of our model. This is a sample of the text the model is able to generate (described below) and a plot defined in the auxiliary functions. 

Next we get the next batch of inputs and targets from the Python generator.

And the magic happens in `sess.run([variables], feed_dict={...})`. The first input, [variables], tells TensorFlow the variables within our model that we want it to calculate. The second input, feed_dict={...}, are the inputs to feed the model, our placeholders above, inputs, targets, and initial states.

On each loop, calculating "training" will mean that we perform a gradient descent step. The result of this will be reflected in the variable loss, that we also calculate and keep on "l" and the corresponding loss series "loss_hist", and "smooth_loss". We will also get the final state on this batch as output from `sess.run()`, and keep it on the "_current_state" variable to feed it back in the next loop cycle.

This is all needed to define and train a recurrent LSTM network in TensorFlow. Of course we want to use the model and see how good it generates texts similar to the input data character by character. This is done in the `sample()`function

## Sample

The sample function will use the process depicted in this diagram:

<div class="post_img">
<img src="../../../assets/lstm-recurrent-networks/test_diagram.png" width="550" height="143" /> 
</div>

This time the inputs are a single character instead of a sequence. As we are not training now, we don't make use of any target character. We just want to see what the network comes up with. Here we use the probabilities for the out of the model  to sample out a next character. this predicted character will be passed as input to the next prediction.

To do that the code below follows a similar process as our training loop:

```
def sample(sample_length, session):
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    _char_state = np.zeros((2, batch_size, hidden_dim))
    txt = ''
    for i in range(sample_length):
        char_probs, _char_state = session.run([probs, current_state],
                                           feed_dict={x: seed, init_state: _char_state})
        pred = np.random.choice(range(vocab_size), p=char_probs[0])
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        character = idx_to_char[pred]
        txt = txt + character
    return txt
```

This time the input is a single encode character chosen randomly. Initial states are again initiated to zero and passed to the `sess.run()` as the "init_state" element in the "feed_dict={}". As we don't do any training here, the variables to calculate are the next states, and the probabilities vector to use to predict the next character. For the prediction we use the numpy function `random.choice()` that chooses elements in an array based on assigned probabilities. If we just choose the maximal probability the texts turn out with less variability and less interesting.

Note that here we use the probabilities variable for the first time. We didn't really need it in the training because the function `tf.nn.softmax_cross_entropy_with_logits_v2()` takes the logits as inputs and calculates the softmax distribution itself. We defined it just to use it here. 

## Summary

TensorFlow is a middle way between the full automation of Keras and the detailed implementation done in the pure Python program. I think the trade-off between knowing the model in deep detail and automatizing most of its declarations is mainly relevant, in a practical sense, when your program does not work and you want to debug and change things to experiment. 

In TensorFlow you still care about some details in the model, like defining the weights and biases yourself. That gives you a deeper understanding of the model, and more work to do. But you get away with the need to care about backpropagation and gradient descent. As in the Keras post, I want to link to this Andrej Karpathy post where he explains why it is useful to [understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b). I think it is great to begin learning neural nets using TensorFlow, but if you want to do more and more with it, at some point, you may want to have a look at those more basic concepts. 




