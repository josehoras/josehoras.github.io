---
layout: post
title: "TensorFlow with GPU using Docker (and PyCharm)"
excerpt: "Where I describe all the non-trivial steps I took to have my GPU working."
author: "Jose"
date: 2019-03-11
---

I have acquired a laptop with a Nvidia GPU to help me in my learning of AI and Neural Nets. So, I turn quickly to try and run my old models on the GPU and see how fast it is. 

Well... it turns out that running TensorFlow with your local GPU is not really straightforward. Eventually, I needed to go through many tutorials and Stack Overflow questions to finally get my model running on GPU. After all that effort, I want to share here my errors and the steps that eventually lead me to success.
## The wrong way

The main problem is that, as of now, TensorFlow needs Nvidia CUDA 9.0 Toolkit to run, but I am using Linux Mint 19 which, being based on Ubuntu 18.04, installs CUDA 9.1. It is possible to install the previous version on this system, but doing this is way more complex than you would think and, in my case, after one full day of trying, the configuration that allowed me to use the GPU crashed my system when I restarted the computer.

So, I do not recommend installing the dependencies that Ubuntu 18.04 needs to run TensorFlow on your system.
## The right way

As I have learned, there is a solution for this problem: *Docker!*

Docker allows you to run your program within a container that has the environment and dependencies you need. This way you don’t have to mess up your host system. These containers are like lightweight virtual machines, as they use the operating system of the host machine to run, and don’t need another full operating system inside them. A great idea indeed.

So, this is the way to go. But as I said, it is not straightforward. I will walk through the following steps:

* Install Docker CE
* Install Nvidia Docker 2.0
* Pull a TensorFlow Docker image
* Create a new image for your program with a Dockerfile

And additionally twopoint that were quite important for me:

- Set up the tweaks I needed for my program: Bind mounts, and Matplotlib
- Run your container in PyCharm

Of course I didn’t create all this info, and for each step I used some good documentation to learn what to do. I will put these links on each section. You should follow them, as I cannot improve on these tutorials. I just give you here a summary of what, for me, were the most important points and some issues I had. This way I hope I make it easier for you , having all the links together and recounting my experience going through them.

## Install Docker CE
The installation steps are well described [under this link](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

I will just add that for me, using Linux Mint Tessa, I had to change the way to add the docker repository. In the section “Set up the repository”, point 4, instead of

```
 ~$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $ (lsb_release -cs) stable"
```

I changed `(lsb_release -cs)` by `bionic`, as this is the Ubuntu version name Mint Tessa is based on. Maybe your system is based on `xenial`, or `trusty`.

After installing Docker and before testing it I needed to restart the computer, otherwise I would get an error as the /var/lib/docker/ directory is not yet created

Don’t forget to follow the “Post-installation steps for Linux” in order to add your user to the new docker group: 
- Create the docker group `~$ sudo groupadd docker`
- Add your user to the docker group `~$ sudo usermod -aG docker $USER`
- Restart
- Verify that you can run docker commands without sudo `~$ docker run hello-world`

## Install Nvidia Docker 2.0
From [this post](https://devblogs.nvidia.com/nvidia-docker-gpu-server-application-deployment-made-easy/), “nvidia-docker is essentially a wrapper around the docker command that transparently provisions a container with the necessary components to execute code on the GPU”.  In short, we need to install nvidia-docker to use the GPU in Docker.

Again you can follow the instructions on: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

And again, if you work on Linux Mint as I do, you should change the line

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
```

to `distribution=ubuntu18.04` or the Ubuntu distribution your Mint system is based on.

In the last step we test that the installation works with the command:

```
~$ docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

The first part of this command, `docker run –runtime=nvidia`, tells Docker to use the CUDA libraries. If we skip `–runtime=nvidia`, Docker alone will not be able to run the imega. We can also use `nvidia-docker run` and it will work too.

The second part tells Docker to use an image (or download it if it doesn’t exist locally) and run it, creating a container. It runs the command `nvidia-smi` on this container. The `–rm` flag tells Docker to delete the container after it has run.

## Pull a TensorFlow Docker image
Now that you have Docker, you can download, or pull, the images you need from the web. There are all kind of images uploaded to the official Docker repository (where you can also upload your own images). From there we pull the latest stable TensorFlow image with gpu support and python3. You can find more details [here](https://www.tensorflow.org/install/docker), or directly type the command:

`~$ docker pull tensorflow/tensorflow:latest-gpu-py3`

Now that we have the Tensorflow image and the Docker wrapper for CUDA 9.0, we will create another, personalized, image to run our program.

## Create a new image for your program with a Dockerfile
This process is described [here](https://docs.docker.com/get-started/part2/). I will summarize it in four steps:

- Get into the directory of your Python program, in my case “Tensorflow.py”
- Create a new file called “Dockerfile”. If you need further dependencies for your app, create also “requirements.txt” with the names of the Python libraries you need.  As I do not need extra libraries at this moment, my Dockerfile looks like this:

```python
FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run app.py when the container launches
CMD ["python3", "/app/TensorFlow.py"]
```

- Create your image with this command (I give the name “mytensorflow-py3” to my image):

```
~$ docker build -t mytensorflow-py3 .
```

- And your image with:

```shell
~$ docker run --runtime=nvidia --rm mytensorflow-py3
```

The `–rm` flag will remove the container when the run is finished

## Bind mounts
Now you have managed to run your Tensorflow Python program in a GPU. But I still needed two more improvements on this.

With this setup I needed to create the image again every time I was changing something in my program and wanted to run and check the new code. That is very cumbersome if you are developing and are changing and checking your code every minute. It is possible to get rid of this by using a bind mount.  A bind mount mounts a file or directory on the host machine into a container. The full docu for binds mounts is [here](https://docs.docker.com/storage/bind-mounts/). As you see in this link, Docker recommends using volumes for data storage. As I understand volumes are good for storing data your app is creating, but volumes do not easily link the local Python file you are working on. So, I go for bind mounts.

First I delete the copy command on the Dockerfile. I don’t need this anymore because now instead of copying, I will bind my local directory there. I keep the WORKDIR command though.

Next I build my image just as before (for the last time in a while). And now my command to run my image looks like this:
```
~$ docker run --runtime=nvidia -v $(pwd):/app --rm mytensorflow-py3
```
After changing my code I just run this command again without having to rebuild the image.

## Matplotlib
It happens that my program uses Matplotlib because I like to plot and see how well my neural net is training, by plotting the value of the loss function with every iteration. But for doing that in Docker I found two difficulties.

First, I needed the Tkinter dependency. In the [link above](https://docs.docker.com/get-started/part2/), they teach you how to install additional Python dependencies using a “requirements.txt” file and this command inside the Dockerfile:
```
RUN pip install --trusted-host pypi.python.org -r requirements.txt
```
But this will not do with Tkinter. Instead, you need to install it with using “apt-get install”. And you can do this in an image too, at the moment you create it with the Dockerfile. Instead of the line above, place the following in you Dockerfile:
```
RUN apt-get -y update
RUN apt-get -y install python3-tk
```
With the update command you may solve some dependency problems I had.

The second problem is that my container does not have access to the x11 socket of the host. We will actually use another bind mount to give our image access to X11. We do that adding these flags to the command line:
```
-v /tmp/.X11-unix:/tmp/.X11-unix
-e DISPLAY=unix$DISPLAY
```
And by now my command to run my program looks like this:
```
~$ docker run --runtime=nvidia -v $(pwd):/app -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -t --rm mytensorflow-py3
```
The -t flag allocates a pseudo-TTY in the container, so you will see your program output as it comes.

The -e flag sets an environment variable. You can put this in the Dockerfile instead, and skip it in the command line. But you need to place the literal name of your DISPLAY environment variable
```
ENV DISPLAY unix:0
```
We still need to allow all users to print on X11 with:
```
~$ sudo apt-get install x11-xserver-utils
~$ xhost +
```
and your plots should pop up 🙂

## Run your container in PyCharm
Until now we have been running the container using the console and the command line. But if you are as lazy as me, you would prefer to have it integrated in your IDE. You can have it on PyCharm Community, and do not even need the full blown (and not free) PyCharm Professional. At least from the version PyCharm 2018.3.2.

I could not find the last version from the beginning and needed to install it via snap

```
$ sudo apt install snapd
$ sudo snap install pycharm-community --classic
```

But maybe now your app store has the last version too

You have a great description in [this documentation](https://www.jetbrains.com/help/pycharm/docker.html). Basically:

- Install the Docker Integration plugin on (File - Settings - Plugins). I did not need, neither found the Python Docker plugin.
- Add a Docker configuration on (File - Settings - Build, Execution, Deployment - Docker)
- Use the Docker daemon in the Docker tool window (View - Tool Windows - Docker). This GUI is very easy and you can see and manage your images and containers with a few mouse clicks
- Use (Run - Edit Configurations) to define how to run your container. Set the “Image ID” (name) you want to run, and the additional command line flags (like bind mounts) on “Run options”

After that the GUI is pretty easy and the Jetlabs documentation above very complete.

Have fun!
