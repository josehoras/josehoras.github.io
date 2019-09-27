---
layout: post
title: "Save and restore RNN / LSTM models in TensorFlow"
excerpt: "How to save a model in TensorFlow using the Saver API (tf.train.Saver)"
author: "Jose"
date: 2019-09-27
---


If you have been learning TensorFlow for a while, you have probably trained some models, check that they work as intended, and then forgotten all about then. It is when you want to use this knowledge for a real-world problem that you realize you need to save the trained model to use it later. Saving a restoring a model may sound easy compared with grasping the concepts necessary to train a neural net. However, I found it wasn't that trivial for me. Particularly I found two difficulties:

- Restoring my model in a different program to the one where I trained it (two files)
- Restoring a recurring net (RNN / LSTM) 

Most of my problems came from using the easy **Saver** API (tf.train.Saver). This is good and simple to do your training in different sessions and load your variables each time. For more complex tasks, like packaging your model for use with different users, devices, or languages the **SavedModel** API is your solution.

Here I will go on explaining what can you do with the Saver API. Even though it gets complex at some point, I think understanding this also serves to better understand and use TensorFlow concepts like graph, session, operations, tensors...

So, let's start with the easy stuff.

### 1) Save and restore variable values within a program

As working example I will use my [previous program](https://github.com/josehoras/LSTM-Frameworks/blob/master/tf-lstm-char.py) to train a LSTM network on character level with TensorFlow. This program is inspired by Karpathy's [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086), and explained in [this other post](https://josehoras.github.io/lstm-in-tensorflow/).

To save and restore models in TensorFlow we use the `tf.train.Saver()` class. This class creates a saver object.

```
saver = tf.train.Saver()
```

The object constructor adds save and restore operations to the graph for the variables defined in the graph. Within a session, you can use the function `saver.save(session, filename)` to save the variables values. This command creates the following files:

```
checkpoint
model.ckpt.data-00000-of-00001
model.ckpt.index
model.ckpt.meta
```

After saving the model in these files, you can restore the trained variables by using `saver.restore(session, filename)`, again within a session.

You can find an example on [tf-lstm-char_save.py](https://github.com/josehoras/LSTM-Frameworks/blob/master/tf-lstm-char_save.py). It is quite easy. At the end of the program, when the model is trained, a new session is opened, the variables are restored, and we can sample a new text again.

```
with tf.Session() as sess:
    saver.restore(sess, save_path + "model")
    print(sample(600, sess))
```

`save_path` contains the folder name where I saved my model. `sample()` is the function that uses the model to generate a new text, similar to the text corpus the model was trained in. In the new session the trained variables are not even initialized, and is we skip the restore command we get an error on trying to use an uninitialized value. We can initialize the variables with `tf.global_variables_initializer()`, but all previous training will be lost.

### 2) Restore variable values within a different program

The thing is, `saver.restore()` will restore the values of our trained variables within the existing graph. It assumes the graph contains these variables, and if the current graph is empty, or differs with the graph of the saved model in some variable name, it will give an error. If the current graph's names of placeholders or operations is changed, the restore function still works.

The file [tf-lstm-char_restore-1.py](https://github.com/josehoras/LSTM-Frameworks/blob/master/tf-lstm-char_restore-1.py) is a copy of last section's example, but all code regarding to training is deleted. The same graph is defined at the beginning, and the session only contains the code to restore and test the saved model. 

This is an example of a code that loads a model trained somewhere else, maybe by somebody else. But you still have to redefine the exact graph originally used.

### 3) Restore graph and variable values within a different program

But we can load the model's graph from the saved files. Concretely, the graph information is contained in `model.ckpt.meta`. You can find the example for this section on [tf-lstm-char_restore-2.py](https://github.com/josehoras/LSTM-Frameworks/blob/master/tf-lstm-char_restore-2.py).

We can import the graph to the current default graph with the function `tf.train.import_meta_graph()`. This function also returns a saver object, like the one we used before.

```
saver = tf.train.import_meta_graph(save_path + "model.meta")
```

At this point the default graph contains all the operations that were used in the model we just loaded. However, the tensors we will used are not defined yet. To understand this let's look at a piece of code in the function `sample()`:

```
char_probs, _char_state = session.run([probabilities, current_state],
                                      feed_dict={x: seed, init_state: _char_state})
```

Here we want TensorFlow to evaluate the operations "probabilities" and "current_state" given than we are feeding certain values to the placeholders "x" and "seed". But these are the names that remain from our implementation from the sections above, and we have deleted all the code that defined the graph. Now "probabilities" is just an undefined Python variable.

We have to redefine the variable "x", or whatever we want to call it now, and make it point to the original placeholder. This placeholder, and the rest of the saved graph, is now in our default graph. We have to get the tensor from the graph and place it in the Python variable "x". When we defined our placeholder before training our model we were clever and did:

```
x = tf.placeholder("float", [None, batch_size, vocab_size], name="x")
```

So, the name of the tensor is the same as the name of the Python variable we used. Otherwise the name of this tensor would have been "Placeholder". This will allow us to recover the next tensors form our current default graph:

```
x = sess.graph.get_tensor_by_name('x:0')
init_state = sess.graph.get_tensor_by_name('init_state:0')
probabilities = sess.graph.get_tensor_by_name('probabilities:0')

```

However the tensor stored in "current_state" comes from:

```
h_states, current_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=rnn_tuple_state,
                                            time_major=True, dtype=tf.float32)
```

And `dynamic_rnn()` does not have a name option. It was not obvious to me how to get this tensor name, until I found this post: [Understanding TensorFlow's rnn inputs, outputs and shapes](https://www.damienpontifex.com/2017/12/06/understanding-tensorflows-rnn-inputs-outputs-and-shapes/). Then I went back and printed this tensor when I created it before training, the output being:

```
LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(1, 500) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(1, 500) dtype=float32>)
```

So, this tensor is actually a LSTMStateTuple composed of two other tensors. Without going into detail on what this means here, the way to obtain back this tensor into the "current_state" variable is:

```
current_state_c = sess.graph.get_tensor_by_name('rnn/while/Exit_3:0')
current_state_h = sess.graph.get_tensor_by_name('rnn/while/Exit_4:0')
current_state = tf.nn.rnn_cell.LSTMStateTuple(current_state_c, current_state_h)
```

And finally we have all the tensors we need into our variables that allow us to run `sample()` successfully.

### Conclusion

You cannot save and restore a model with the basic functionality without thinking about your model graph. This basic functionality is good to restore the trained variables values when you have the graph defined in your program already, as shown in sections 1 and 2.

In section 3 I showed a way to extract from the graph the tensors you need without knowing much of the graph's details. Still, you need to have been careful creating the graph to know the variables' names. And for the TensorFlow RNN / LSTM functions you need to explicitly print out the tensor to obtain their names within the graph.

Although this methods are not straightforward, discussing them is a learning opportunity to understand the working inner of TensorFlow.

    
    
    
    
    
