---
layout: post
title: "PyTorch is great"
excerpt: "Introduction to PyTorch using a char-LSTM example"
tags: ["PyTorch", "LSTM"]
author: "Jose"
date: 2020-02-05
---

I have found PyTorch while following the CS224n course on NLP imparted by Christopher Manning, and learning it has been a great experience. 

There are several characteristics that make PyTorch so compelling. Using an imperative programming style, it feels very Pythonic and intuitive. It also features dynamic computation graphs, which provides more flexibility. 

But honestly, at this point the feature I'm most excited about is the easy integration of GPU. You define your GPU with just one line (`device = torch.device("cuda:0")`) . At any time afterwards you can easily move data and variables to CPU or GPU as you see fit. Compared with all the fine-tuning needed to use my GPU in TensorFlow, that I extensively described [here](/../tensorflow-with-gpu-using-docker-and-pycharm/), this is bliss.

To showcase some of the advantages and ease of use, I will replicate here my favorite neural network: a char LSTM. Last year, I got deep into this architecture and implemented it in [Python](/../lstm-pure-python), [TensorFlow](/../lstm-in-tensorflow), and [Keras](/../lstm-in-keras), as I wanted to understand better both the LSTM and the different deep learning frameworks. It's only natural that I would go now and translate it to PyTorch.

I will not go here into the theory of LSTM, as I did in the previous posts. The following is just a description of the simplest program I could come up in PyTorch to set up and train a char-LSTM model. Also, I won't explain every function and detail, but instead insert an hyperlink to the relevant documentation. You can see the complete code [here](https://github.com/josehoras/LSTM-Frameworks/blob/master/pytorch-nn-lstm-char.py)

I hope this is useful to have a first look and test the advantages of PyTorch. For a more in-depth understanding I recommend [this post](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e).

## Overview

The steps to train a model are summarized in the following diagram:

<div class="post_img">
<img src="/../../../assets/lstm-recurrent-networks/sgd_diagram.png" width="550" height="300" />
</div>

The way PyTorch works, we will first define the following components:

- [Model](#model)
- [Dataset](#dataset)
- [Loss Function](#loss-function)
- [Optimizer](#optimizer)

Then we'll put it all together to work in the training loop. 

Finally, we will set up a function to [test](#test) our network and sample a text out of it that, hopefully, will resemble the style of the input text we feed into the network. In the example below I'll use the full collection of Shakespeare works.

## Model

In PyTorch a model is defined by a Python class that inherits from the nn.Module class. The model itself (its layers) is defined in the `__init__(self)` method. The computation and the model outputs, or predictions, is defined in the `forward(self, x)` method.

Let's go through my simple implementation of a LSTM model:

```
import torch.nn as nn

class char_lstm(nn.Module):
    def __init__(self, vocab, hidden_size, n_layers = 1):
        super(char_lstm, self).__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab
        self.lstm = nn.LSTM(self.vocab_size, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab, bias=True)

    def forward(self, input, states_0=None):
        output, (hn, cn) = self.lstm(input, states_0)
        scores = self.linear(output)
        return scores, (hn, cn)
```

Our model has just two layers, a recurrent LSTM and a Linear (feed forward) layer that takes the results of the LSTM and output scores for each step.

The forward function just applies these two layers, and returns the scores for each character as well as the final hidden and cell states.

The model is an object that we will declare below

```
model = char_lstm(vocab_size, hidden_dim, n_layers=n_layers).to(device)
```

`vocab_size` is the length of our character dictionary, that is, the number of possible characters in the data. We will get this number in the Dataset function below. `hidden_dim` and `n_layers` are defined hyperparameters.

## Dataset

You can feed your data to your model in different ways, for example using a Python iterator. But PyTorch provides a convenient Dataset class for that. A dataset is represented by a regular Python class that inherits from the Dataset class. It has three main methods:

- `__init__(self)`: Defines our data. The delivery of data will be provided in the next method, so you don't need to place all your data in memory here. In our case, we pass the file name where the data is. Our data is a big text file with all Shakespeare's works. We also define some parameters to use in the next method
- `__getitem__(self, index)`: This method returns a tuple (feature, label) for the index element of the dataset. The DataLoader will use this method to fetch the wanted minibatches from the dataset. In our case, as we are training the model to predict the next character, the feature is the character at the given position, and the label will just be the next character.
- `__len__(self)`: It returns the size of the entire dataset

```

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_name):
        self.data = open(data_name + '.txt', 'r').read()
        chars = sorted(set(self.data))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        print('data has %d characters, %d unique.' % (len(self.data), self.vocab_size))

    def __getitem__(self, index):
        x = self.char_to_idx[self.data[index]]
        x = torch.tensor([x])
        x = F.one_hot(x, num_classes=self.vocab_size)
        x = x.type(torch.FloatTensor)
        t = self.char_to_idx[self.data[index + (index < (self.__len__() - 1))]]
        t = torch.tensor([t])
        return (x.to(device), t.to(device))

    def __len__(self):
        return len(self.data)

    def params(self):
        return self.vocab_size, self.char_to_idx, self.idx_to_char
```

Some words on the `__getitem__(self, index)` method. Our raw data is just text, but we have to pre-process these text characters to a format that our model can use. Basically, as the model will do mathematical operations with the input, it expects numbers instead of letters. First we use our "character to index" dictionary, that assigns a number to each letter. But the model is expecting a [one_hot](https://pytorch.org/docs/stable/nn.functional.html#one-hot) encoded vector of `vocab_size` length. Furthermore, to compute gradients and whatnot, it wants this vector in a float format.

Meanwhile, we will use the targets to compute the loss. It turns out [the loss function](https://pytorch.org/docs/stable/nn.html#crossentropyloss) we will use does not require one hot encoding. So, the corresponding number will suffice. The piece of logic in the dictionary index avoid the DataLoader to look for a character over the dataset length.

In this case, some parameters of my model are to be extracted from the text I choose. For instance, the vocabulary size, the number of possible characters, is not fixed but depends of the text in question. That is also the case for the dictionaries to convert from character to index and viceversa. To be able to use these parameters somewhere else in the program, I have also added a method `params(self)` that passes over these parameters. 

Remember, this is only the definition of our dataset object. Next, in the main body of our program, we instantiate this object and define the data loader function.

```python
train_data = CustomDataset('shakespeare')
train_loader = DataLoader(dataset=train_data, batch_size=seq_length, shuffle=False)
```

The DataLoader is defined by the dataset it uses to grab the data batches, the batch size to use, and whether we shuffle this data. In many applications shuffling the data is very useful to get rid of some biases appearing in the data collection. However we cannot do that here. Our recurrent LSTM network will follow the character sequence to predict the next one. If we shuffle the data we will obviously loss this sequence dependence.

Until that we have just defined and instantiate our dataset. Now it is ready to use, which we'll do in the training loop below.

## Loss Function

PyTorch has defined several loss functions that you can use. You can review them in the [official documentation](https://pytorch.org/docs/stable/nn.html). I pick the function [CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) that according to the documentation "combines nn.LogSoftmax() and nn.NLLLoss() in one single class". 

This means that the function applies Softmax to the scores we get out of the model, and then compare these probabilities with the labels to compute our loss. Thus, in our scheme above, we go from scores to loss. Here we just define the loss function that we later use in the training loop.

```
loss_fn = nn.CrossEntropyLoss()
```

## Optimizer

Once we have the gradients of our weights with respect to the loss, we need to do a gradient descent. This process adjust our weights in a way that the loss would have been a little less, and consequently the predictions would have been a little better. As we continue this loop with further pairs of inputs an targets, our network will learn to spell out characters in a Shakespearean style.

But first, as we did with the model and the loss function, we have to define our optimization algorithm of choice. Instead of going for the vanilla [stochastic gradient descent](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) we choose the fancier [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) algorithm, because in PyTorch is the same work to type down one function or the other.

```
optimizer = Adam(model.parameters(), lr=lr)
```

## Training loop

Here is where all elements we defined (model, data loader, loss function, and optimization algorithm) come into use to train our model.

```
# Initialize initial hidden and cell state
h = torch.zeros(n_layers, 1, hidden_dim).to(device)
c = torch.zeros(n_layers, 1, hidden_dim).to(device)
states = (h, c)

for inputs, targets in train_loader:

    # Forward run the model and get predictions
    scores, (h, c) = model(inputs, states)
    states = (h.detach(), c.detach())
    loss = loss_fn(scores.squeeze(dim=1), targets.squeeze(dim=1))

    # Backpropagate the loss and update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The job of the training loop is:
- get a batch of data
- pass the inputs through the model obtaining the target predictions
- compute the loss of these predictions compared to the real target
- backpropagate the loss to the model parameters
- update these parameters minimizing the loss

And of course, do this over and over again until the models gets good at predicting the target characters.

Before starting the loop, we initialize the LSTM states to zero.

The training loop itself is defined by the DataLoader, and it will go once over all the data in our DataSet. 

Next we feed the input to the network, together with the hidden and cell states, by simply calling the model. The resulting scores assign for each prediction a list of length `vocab_size` with a measure of the probability of each character in the vocabulary to be chosen the next one. To obtain the normalized probability we should use softmax, but this step is automatically included in the loss function.

We update the new states. We do not want to compute the loss over the states variable. This PyTorch could do in the first loop, but afterwards it will try to backpropagate all the way over to the first states. As we continually update the states, we loss the previous values on each loop and PyTorch will give an error trying to backpropagate over these. The method `.detach()` tells PyTorch to not compute the loss over the states.

The scores and the targets are all the loss function needs to compute the loss. 

Next we want to obtain the gradients of the loss with respect to the model's weights. What would be a complex backpropagation routine, is the part where deep learning framework perform their magic. PyTorch delivers it wil the line `loss.backward()`.

Finally we make one gradient descent step, updating the network parameters, just calling `optimizer.step()`. Before going to the next loop iteration we have to remember to zero the parameter gradients we just computed. Honestly, this is the only step where PyTorch kind of bugs me a little. It happens that in PyTorch gradients are accumulated. Thus, if we again backpropagate the gradients will sum to the previous ones. This sum will not be the correct gradient for that particular gradient descent step. To avoid that, we zero the gradients every time we apply an optimizer step.

That's all! The loop will go on until the DataLoader runs out of data in the dataset. The network is training, but of course we want to visualize how good this is working.

## Test

I have added one more method to our model class in order to produce a sample of text with the current state of the network. We call this method within the loop every 500 iterations to see how the text quality evolves from random gibberish to a decent Shakespeare imitation.

Similar to the training scheme shown above, the scheme for this routine is:

<div class="post_img">
<img src="/../../../assets/lstm-recurrent-networks/test_diagram.png" width="550" height="143" />
</div>

So, the same as in training, we call our forward function to obtain the scores. But this time we don't have targets to calculate the loss, and, thus, we don't have a loss function that also takes care of computing scores into normalized probabilities. We use then [nn.funtional.softmax](https://pytorch.org/docs/stable/nn.functional.html#softmax) to obtain the probabilities, and [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) to pick a character according to its probability. 

Finally, similar as we did in the DataLoader, we pre-process this character to be the next input to our net, re-scaling to the correct dimensions with the [view](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view) method.

```
    def sample(self, x, txt_length=500):
        x = x.view(1, 1, self.vocab_size)
        h = torch.zeros(self.n_layers, 1, hidden_dim).to(device)
        c = torch.zeros(self.n_layers, 1, hidden_dim).to(device)
        txt = ""
        for i in range(txt_length):
            scores, h, c = self.forward(x, h, c)
            probs = nn.functional.softmax(scores, dim=2).view(self.vocab_size)
            pred = torch.tensor(list(WeightedRandomSampler(probs, 1, replacement=True)))
            x = F.one_hot(pred, num_classes=self.vocab_size)
            x = x.view(1, 1, self.vocab_size).type(torch.FloatTensor).to(device)
            next_character = idx_to_char[pred.item()]
            txt += next_character
        return txt
```

After I went over the full dataset, here is an example of the kind of text my trained network was able to deliver:

```
Bases of the petition,
Must born so living beauty and this abold,
A side, rich, and sut thee sick, that first be bethink whre's
Marvel in presedd to my injury ia this thief,
May ilf he's this winged nounctian?
It is that those sfright
Her money, where he knows me, on eArster again, and love,
King the quarreen! shall lay then,
Not optress hLefortrate mad bed:
once the purpose faith. Knought, I rage,
Which a dreamer that this joil only,
That I be foos but with all supply of both as men,
That thoug
```





