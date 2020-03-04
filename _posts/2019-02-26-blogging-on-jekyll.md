---
layout: post
title: "Blogging on Jekyll"
excerpt: "Why I decided to blog in Jekyll on GitHub instead of WordPress."
author: "Jose"
date: 2019-02-26
---

Well, I am finally starting a blog. There have been times in the past that I entertained the thought of starting a blog, but didn’t got to do it. For example, as I started my year backpacking around South East Asia I really though a blog was a great way of documenting my travels and keep in touch with friends and family. However, as I started traveling, I found a better use of my time was to focus on living the places I was visiting instead of writing about them.


So, why start now? Do I think I’ll have more exciting adventures to tell, and more followers, writing about learning artificial intelligence instead of about surfing in Bali? Not really...


But I do think writing can help me more on this travel than on my backpacking. Writing is a great way to really fix new concepts in your mind. I found the best way to deeply understand something is to explain it. Frequently, we think we understand something up until the time we have to put that concept into words. Then new questions and ideas arise, and when the explaining is done, we may have new insights on that old concept. As I don’t expect to find many volunteers willing to be schooled into AI by me, writing a blog would be a better option to achieve this.

Also, a blog is a good complement to my GitHub, where I want to place personal projects and the projects of my Udacity course. Because even better than explaining is actually doing and programming those neural nets from scratch. That follows the Richard Feynman quote, "What I cannot create, I do not understand"

## Learning Basic Web Development

And starting the blog itself has already been a good dose of learning. This is the first time I start a web page and didn't have any knowledge about HTML, CSS, or blog hosting services. Just making up my mind on which way to go has taken a lot of thought. For simplicity, I had a first go at WordPress.com, but never got to really like it. It can well be my own ability, but no matter how much time I spent clicking around the different themes and options, I never got the clean format I wanted. And there was that feeling of not being in really in control of what I was doing that quite annoyed me.
### Jekyll

And then I stumbled upon Jekyll. I heard about it first when creating my GitHub profile. Some days after I began realizing that some of the best blogs I was following were actually implemented on Jekyll. That was enough to give it a try, and soon I was convinced this was the way to go.

According to its homepage, Jekyll is a simple, extendable, static site generator. It is also blog aware, meaning that it automatically manages your posts just by placing them on the _**_posts**_ folder. The idea is quite simple. Just start from scratch defining your page, starting easy and improving as you learn. You have everything under control and learn much more in contrast to just choosing a WordPress theme.

### Courses, courses...
To get started I used two tutorials on [GitHub Training](https://lab.github.com/githubtraining/): [GitHub Pages](https://lab.github.com/githubtraining/github-pages), and [Introduction to HTML](https://lab.github.com/githubtraining/introduction-to-html). They go to the point and just focus on learning by doing, which I find great. The CSS basics I got from this free [Udacity course](https://www.udacity.com/course/intro-to-html-and-css--ud001). It is almost impossible to be more concise and practical than the GitHub courses, but Udacity does a good job too.
### ...and start doing!
To give a feeling of what all this implies, I'll sketch of I went about the first steps to go live with a web page on GitHub Pages with Jekyll.

I began by creating the index.html file in the root of your project's folder. After learning HTML basics, I was good to place a basic scafolding of the page: title on a header, a place for posts listing in the middle, and some information on a footer. 

Next I had to put some formatting with a CSS file. This file just refers to each element on your index.html and defines the way you want that element to show on the browser. Things like font style, size, colors... The basics of it can be also learned quickly with a starting tutorial. But I have to admit that I also learned a lot by inspecting the format of my favourite blogs using the browser's "Inspect" tool and copying, or _"versioning"_, some of it.

Finally just placed my [first post]({{site.posts.last.url}}) on the _**_posts**_ folder and integrate on index.thml using [Liquid](https://jekyllrb.com/docs/liquid/) syntax and tags. You can just copy some examples from the documentation to understand it. For example 

{% raw %}`{% for post in site.posts %}...{% endfor %}`{% endraw %} 

defines a loop over all posts you have in your _**_posts**_ folder. To include content use double brackets, like  {% raw %}`{{post.url}}`, or `{{post.title}}`{% endraw %}.

As the content increases you should keep the Jekyll [directory structure](https://jekyllrb.com/docs/structure/) to keep your site modular and well organized. I have already placed some content in the _**assets**_ and _**_layouts**_ folders. But the simple steps above are enough to start posting with Jekyll.

This is still very simple, and you can see on the current design of the page. However, It is a great platform to start posting and learning in a clean way, and improve from here.