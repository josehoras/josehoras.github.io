---
layout: post
title: "LSTM implementation in Keras"
excerpt: "Implementation of a LSTM recurrent neural network using Keras."
author: "Jose"
date: 2019-05-05
---

In this post I tell about how I designed a LSTM recurrent network in Keras.

The goal of this post is not to explain the theory of recurrent networks. There are already amazing posts and resources on that topic that I could not surpass. I specially recommend: 

- Andrej Karpathy's [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- Christopher Olah's [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Stanford's [CS231n lecture](https://youtu.be/6niqTuYFZLQ)

Instead in this post I want to give a more practical insight. I'm also doing the same, in two separate posts, for pure Python and TensorFlow. The aim is to have the same program written in three different frameworks to highlight the similarities and differences between them. Also, it may make easier to learn one of the frameworks if you already know some of the others. 

My starting point is Andrej Karpathy code [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086), described in his post linked above. I first modified the code to make a LSTM out of it, using what I learned auditing the CS231n lectures (also from Karpathy). So, I started from pure Python, and then moved to TensorFlow and Keras. You may, however, come here after knowing TensorFlow or Keras, or having checked the other posts.

The three frameworks have different philosophies, and I wouldn't say one is better than the other, even for learning. The code in pure Python takes you down to the mathematical details of LSTMs, as it programs the backpropagation explicitly. Keras, on the other side, makes you focus on the big picture of what the LSTM does, and it's great to quickly implement something that works. Going from pure Python to Keras feels almost like cheating. Suddenly everything is so easy and you can focus on what you really need to get your network working. Going from Keras to pure Python feels, I would think, enlightening. You will look under the hood and things that seemed like magic will now make sense.

---
# LSTM in Keras

You find this implementation in the file `keras-lstm-char.py` in the [GitHub repository](https://github.com/josehoras/LSTM-Frameworks).

As in the other two implementations, the code contains only the logic fundamental to the LSTM architecture. I use the file `aux_funcs.py` to place functions that, being important to understand the complete flow, are not fundamental to the LSTM itself. These include functionality for loading the data file, pre-process the data by encoding each character into one-hot vectors, generate the batches of data that we feed to the neural network on training time, and plotting the loss history along the training. These functions are (mostly) reused in the TensorFlow and Python versions. I will not explain in detail these auxiliary functions, but the type of inputs that we give to the network and its format will be important.

## Data

The full data to train on will be a simple text file. In the repository I uploaded the collection on Shakespeare works (~4 MB) and the Quijote (~1 MB) as examples. 

We will feed the model with sequences of letters taken in order from this raw data. The model will make its prediction of what the next letter is going to be in each case. To train it will compare its prediction with the true targets. The data and labels we give the model have the form:

<div class="post_img">
<img src="../../../assets/lstm-recurrent-networks/data_batch.png" /> 
</div>

However, we don't give the model the letters as such, because neural nets operate with numbers and one-hot encoded vectors, not characters. To do this we give each character an unique number stored in the dictionary `char_to_idx[]`. Each of these number is a class, and the model will try to see in which class the next character belongs. A neural network outputs the probability for this of each class, that is, a vector of a length equal to the number of classes, or characters we have. Then it will compare this probability vector with a vector representing the true class, a one-hot encoded vector (that's its name) where the true class has probability 1, and all the rest probability 0. That's the kind of vectors we get from the encode function.

So, when we pass a sequence of **seq_length** characters and encode them in vectors of lengths **vocab_size** we will get matrices of shape (seq_length, vocab_size). It's very important to keep track of the dimensions of your data as it goes from input through the several layers of your network to the output. When we define our model in Keras we have to specify the shape of our input's size. But Keras expects something else, as it is able to do the training using entire batches of the input data at each step. And it actually expects you to feed a batch of data. Although this is pretty cool, we will feed one sequence and its targets at a time to keep it simple. This would be a batch of one element, and the corresponding matrix Keras will have is one of shape **(1, seq_length, vocab_size)**, 1 being our batch size.

So, in our case we specify (seq_length, vocab_size) and pass a batch of (1, seq_length, vocab_size).

## Model architecture

After having cleared what kind of inputs we pass to our model, we can look without further delay at the model itself, defined in `keras-lstm-char.py`. Our model is composed of:

- one LSTM layer, that process sequentially the temporal input series (our characters sequence), and outputs a sequence of hidden states
- one dense layer, that transforms each hidden state into a vector of scores or logits for each character in our dictionary
- a softmax transformation that normalizes our logits in probabilities (the sum of probabilities for all characters equals 1)

I will define this model in Keras using the `Model()` API:

```
inputs = Input(shape=(None, vocab_size))
lstm_layer = LSTM(hidden_dim, return_sequences=True, return_state=True)
lstm_output, _, _ = lstm_layer(inputs)
dense_layer = Dense(vocab_size, activation='softmax')
probabilities = dense_layer(lstm_output)
model = Model(inputs=inputs, outputs=probabilities)
```

This model could be defined as well using the Sequential() method. However the Model() API gives the flexibility to reuse layers or parts of the model to define a second model, which I will do next to check the text generation that the model is able at every N iteration on the training process. 

The network consists of one LSTM layer that process our inputs in a temporal sequence, and delivers hidden states of `hidden_dim` length. This second sequence of hidden states are passed through a Dense layer with softmax activation that converts each hidden state in a probability vector on same length as our `vocab_size`, or the number of characters in our dictionary. This represents the more likely output character `t` given all the previous input characters from `0` to `t-1`.

## Optimization and training

With the model definition done, we have to compare the model outputs with the real targets. Then we use this comparison to optimize the model in a training loop, where batch after batch of data will be feed to the model. This is done in the following lines:

```
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate), metrics=['accuracy'])
print(model.summary())
epochs_log = model.fit_generator(data_feed, steps_per_epoch=it_per_epoch, shuffle=False,
                                 epochs=epochs, callbacks=[history], verbose=0)
```

Before training we have to compile our model. This step mainly defines the way we calculate our loss, and the optimizer method to the gradient descent (or optimization). 

To calculate the loss the model will compare the results of the last step, the probabilities of each character for the prediction, with the input targets. The comparison will result in a certain loss, quite high at the beginning, as the first predictions are totally random.

To reduce this loss and optimize our predictions, Keras use internally a method called Gradient Descent. As we are describing the Keras framework we don't really need to understand this process. It is, on the contrary, described in the Python section above. For us here the optimization is a magic that Keras use on the model to make it improve as it goes through the training data we feed it. There are several optimization methods. Here we use Adam, that works better than the simple Stochastic Gradient Descent (SGD) of the Python version.

The next line `print(model.summary())` is self explanatory. In this summary you can see the model layers, their dimensionality, and number of parameters. It's very useful to check that the model is what you meant it to be.

Finally `model.fit_generator()` does the actual training. We use the `fit_generator()` method because we provide the data using a Python generator function ( `data_feed`). Otherwise we could use the equivalent `fit()` method. We also define the amount of batches to be found in an epoch and the number of epochs we want to train. On each epoch the generator is reset. So, if we define less batches per epoch than the full data for some reason, the data feed will not continue until the end on the next epoch, but will start from the beginning of the data again. We also set `shuffle` to false as we want Keras to keep the time dependency.

## Keras Callbacks

Training will take a long time, depending on how much you want or need to train to see meaningful results. If we set `verbose=1` Keras provides information on how our training is doing. This is good, but I wanted to get something more done at the same time the model is training. To do that Keras let you define callbacks. These are functions that will be called when some condition is true. I have done that defining a class called `LossHistory()`. This class inherits from its parent class "Callback", a Keras class. It has two procedures that will be activated at the beginning of the training and after each batch has been processed.

```
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = [-np.log(1.0 / vocab_size)]
        self.smooth_loss = [-np.log(1.0 / vocab_size)]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.smooth_loss.append(self.smooth_loss[-1] * 0.999 + logs.get('loss') * 0.001)
        if batch % 1000 == 0:
            print(batch, " ", self.smooth_loss[-1])
            aux.plot(self.losses, self.smooth_loss, it, it_per_epoch, base_name="keras")
            test(0, logs)
```

And is instantiated on the line `history = LossHistory()`. As you see this class keeps track of the loss after each batch in the arrays `self.losses` and `self.smooth_loss`. Every 1000 batches it will use them to call our auxiliary function and plot the loss history. That will give you a nice graphical insight on what is actually happening as you train. Also every 1000 batches we call the function test, that will generate a sample of the text the model is able to generate at this point in the training.

Before explaining how we do the sampling I should mention that Keras callbacks where probably not thought for that many content. I took this callback from the Keras documentation and it limits itself to keep track of the loss, assuming you can save or plot it after the training is done. As my callback takes a while to perform all I want it to do, Keras monitors this and gives me a warning: "Method on_batch_end() is slow compared to the batch update". Well, as I know why this happens and I want it this way, so be it.

## Test

Now, the method we use to sample a new text is the following. We input to the model a single character, and the model will make a prediction of the probabilities for each character in the dictionary to be the next one after this input. We choose our next character based on this prediction, which we save as part of the text we are building. This character will be passed to the model again, that will generate another prediction. In this way, we loop over the number of characters we want for our text.

But this process still lacks one important component. Doing as just explained each character will be predicted based on one input character. But the power of the recursive neural networks is to take into account the history of all previous characters to make its prediction. To do this the network saves two internal states (in a LSTM, just one in a regular RNN). These states will change on each loop iteration and, somehow, will keep the relevant information of all characters that the network has seen so far. So, to make the prediction we need to pass not just the last character, but also these two states for the network to know what has been going on so far.

This two states are the reason we define a second model for testing. In our first model we where passing long character sequences for training. Keras kept track of these states internally as it passed the sequence through the network. We didn't need to explicitly worry about them, but now we want them as output of each prediction step to pass it forward into the next prediction step. We need these states to be defined as input and outputs. This second model look like this:

```
state_input_h = Input(shape=(1, hidden_dim))
state_input_c = Input(shape=(1, hidden_dim))
outputs, state_h, state_c = lstm_layer(inputs, initial_state=[state_input_h, state_input_c])
pred_outputs = dense_layer(outputs)
pred_model = Model(inputs=[inputs, state_input_h, state_input_c], outputs=[pred_outputs, state_h, state_c])
```

It looks similar to a new model definition, but if you pay attention we used the layers that we defined in our first model, `lstm_layer`, and `dense_layer`. These layers will be modified (optimized) as we train. When we call this second model, `pred_model`, it will use the layer of the first model in their current state, partially optimized by the training routine. So, as we have defined it, the second model is basically the first one arranged in a way that makes its internal states explicit as inputs and outputs.

Now, the way we use this model is encapsulated in the `test()` function:

```
def test(txt_length):
    txt = ""
    seed = aux.encode([int(np.random.uniform(0, vocab_size))], vocab_size)
    seed = np.array([seed])
    init_state_h = np.zeros((1, batch_size, hidden_dim))
    init_state_c = np.zeros((1, batch_size, hidden_dim))
    for i in range(txt_length):
        prob, init_state_h, init_state_c = pred_model.predict([seed, init_state_h, init_state_c])
        pred = np.random.choice(range(vocab_size), p=prob[-1][0])
        character = idx_to_char[pred]
        seed = np.expand_dims(aux.encode([pred], vocab_size), axis=0)
        txt += character
    return txt
```

In this step we don't train the model, so we don't need to compile or fit against the target data. Instead we use the `predict()` method that will simply evaluate the model for some input and deliver our defined outputs. Before the loop we don't have previous internal states, so we initialize them with zeros. As you see they will keep updating inside the loop on each new prediction. For the prediction we use the numpy function `random.choice()` that chooses elements in an array based on assigned probabilities. If we just choose the maximal probability the texts turn out with less variability and less interesting.

With this you will have fun watching your network improves as it learns to generate text in the same style as the input, character by character. 

## Summary

As you see the Keras framework is the most easy and compact of the three I have used for this LSTM example. You can put together a powerful neural network with just a few lines of code. And the example shown here is even relatively complex, I would say. I wanted to test as I train, and do the test character by character,  for a direct comparison with the two other versions. To achieve that I used the Model() API instead the sequential model to define two versions of the same model. If you want to try out a more simple convolutional network, for example, you'll be fine using one model defined with the `Sequential()` method.

However, easy as it looks like, I want to mention some drawbacks. To program it and find the right information was actually quite difficult for me, and the reason is that when you try something not standard you don't have much insight of the inner workings of Keras in order to debug. Maybe more experience than I have helps of course. But I found in TensorFlow, and of course in pure Python, I had many variables to inspect and see what was going wrong with my code. When I had just five lines of Keras functions for my model and that was not working, it was not clear to me where to begin changing and tweaking. Also, just the understanding of how this really works is quite rewarding for me, and in the long run that effort may pay off. As in the TensorFlow post, I want to link to this Andrej Karpathy post where he explains why it is useful to [understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b).






