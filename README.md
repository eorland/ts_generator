# A NLP-based model to generate descriptions for Taylor Stitch products

## Background

If you know the clothing company [Taylor Stitch](https://www.taylorstitch.com/), then some first thoughts about them might include distinct product photography, a connection to California, and some really unique fabrics -- at least that what's I tend to think of. After a while I also started to notice a very consistent set of product names and a quite recognizable writing style used for their marketing. Taking all of this in, I wondered if I could train a NLP model to write product descriptions for real or made up Taylor Stitch product names. 

Take the product description for the following product as an example:

> *The Jack in Prussian Blue Oxford*

> The Oxford button-down is a stylish wardrobe staple with endless versatility, so it’s no surprise that our signature Oxford, The Jack, has become a mainstay of our roster. Its burly 100% organic cotton basket weave and double-needle felled construction make it sturdy enough to stand up to the roughest workday, while peerless tailoring and a clean, pleatless design means it’s snazzy enough for your next night on the town. We’ve given this classic Prussian Blue iteration a thorough garment dye and wash for a soft, lived-in feel on the very first wear.

A few things here: the product naming conventions are always ***product name*** *in* ***colorway***, and note how the description focuses on the durability of the product, its history within the company, and the wide variety of the product's cases. I won't pretend that these sentiments are exclusive to just Taylor Stitch's marketing, but I've found that the tone and consistency of these product descriptions make for what could be a fun modeling problem. Put more broadly: *with enough examples is it possible to train a NLP model to generate fake marketing descriptions in same writing style?* 

In addition to the consistent product names and descriptions, Taylor Stitch's website made gathering product descriptions to be relatively easy. They have a very accessible [archive](https://www.taylorstitch.com/collections/mens-archive?sorted=best-selling-sales-count) of nearly 1000 of their products, and otherwise a simple website to scrape. With just a few URLs, I was able to scrape all the info I needed and create nearly 1200 training examples from old and in-stock products alike. I've made my current version of the training data file available in ```ref_files/product_info.csv```; the original script I used to create this file is titled ```
info_scrape.py``` and I created an interactive notebook which allows for regular updates of the training data, titled ```update_text_file.ipynb``` 

## Model Info

For the actual model training, I decided to take advantage of many of the already sophisticated NLP models out there, and focused on fine-tuning a light version of [GPT-2](https://huggingface.co/gpt2) provided by Hugging Face. In other words, this became a transfer learning problem. I'd also like to give credit to François St-Amant's great [article](https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272) and whose code served as a foundation which I then adapted to my needs. A while there are more powerful versions of GPT2 available, the lightest version is nonetheless >500mb! As such, having a functional model which was also on the lighter side was important to me. 

OK--**so how good is the model?** Why don't we see with some examples, all of which you can find in ```ref_files/sample_output.txt```...

For a real product that was not used for training: 

> [The Jack in Chestnut Oxford](https://www.taylorstitch.com/collections/mens-shirts/products/jack-in-chestnut-oxford-2207):

>Originally conceived as a companion piece to our
signature Heavy Bag Polo, our Oxford jacket has
developed an incredibly soft, supportive softness
that can be trusted to stay on for years to come.
Built from rugged 100% organic cotton and
beautifully textured with a subtle speckling at
the collar, The Jack is perfect for weekends,
camp, or any other time of the day. With a button
down collar and button up sleeves, The Jack is
just as cozy as ever—tough enough to dress down to
your favorite sweater, jeans, or whatever you wear
them for at night.

Not bad? Note that the model got quite a few things wrong. It was asked to write about a shirt, and I don't know about "dressing down to your favorite sweater". Still, I'm quite pleased to see how it hit several of the original beats in the real product description above: there was a mention 100% organic cotton in the, talk about durability, and even a mention of different use cases! All while having a tone consistent to their actual writing. Think it's over fitting? Here's another take: 

> The Jack in Chestnut Oxford:

> Inspired by early pulp and hardwood shirting, The
Jack in Chestnut Oxford is our take on the classic
British shirt. This ultra-soft, comfortable oxford
iteration is made from 100% organic cotton and is
exceptionally soft—so you won't feel the cold at
the edges, for example. Our collaboration with the
renowned camp collar expert Dan Taylor of
Pantyhose to develop the chunky Oxford stitch
means it's constructed from a very special blend
of organic cotton and 100% recycled yarns that’ll
hold up well for years to come.

All right, the mention of the cotton is back (and more than once) and the durability is mentioned again. It's even cool to see the attempt to chacterize Oxford cloth as a "British" shirt (Even if it that isn't [exactly true](https://www.gazman.com.au/blogs/journal/history-of-the-oxford-shirt).) This example is definitely not as clean so it's safe to say that while the tone and some main points are there, I wouldn't exactly call this one a compelling descripton. 
