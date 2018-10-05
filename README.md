# nonechucks

**nonechucks** is a library for PyTorch that provides wrappers for PyTorch's Dataset and Sampler objects to allow for dropping unwanted or invalid samples dynamically during dataset iteration.

- [Introduction](#Introduction)
- [Use Cases](#UseCases)
- [Installation](#Installation)
- [Examples](#Examples)
- [Contributing](#Contributing)
- [Licensing](#Licensing)

---


<a name="Introduction"/>

## Introduction
What if you have a dataset of 1000s of images, out of which a few dozen images are unreadable because the image files are corrupted? Or what if your dataset is a folder full of scanned PDFs that you have to OCRize, and then run a language detector on the resulting text, because you want only the ones that are in English? Or maybe you have an `AlternateIndexSampler`, and you want to be able to move to `dataset[6]` after `dataset[4]` fails while attempting to load!

PyTorch's data processing module expects you to rid your dataset of any unwanted or invalid samples before you feed them into its pipeline, and provides no easy way to define a "fallback policy" in case such samples are encountered during dataset iteration.    

#### Why do I need it?
You might be wondering why this is such a big deal when you could simply `filter` out samples before sending it to your PyTorch dataset or sampler! Well, it turns out that it can be a huge deal in many cases:
1. When you have a small fraction of undesirable samples in a large dataset, or
2. When your sample-loading operation is expensive, or
3. When you want to let downstream consumers know that a sample is undesirable, or
4. When you want your dataset and sampler to be decoupled.

In such cases, it's either simply too expensive to have a separate step to weed out bad samples, or it's just plain impossible because you don't even know what constitutes as "bad", or worse - both!



**nonechucks** allows you to wrap your existing datasets and samplers with "safe" versions of them, which can fix all these problems for you!



<a name="UseCases"/>

## Use Cases
Coming soon

<a name="Installation" />

## Installation
To install nonechucks, simply use pip:

`$ pip install nonechucks`

or clone this repo, and build from source with:

`$ python setup.py install`.

<a name="Examples"/>

## Examples

Coming soon


<a name="Contributing"/>

## Contributing
All PRs are welcome.

<a name="Licensing"/>

## Licensing

**nonechucks** is [MIT licensed](https://github.com/msamogh/nonechucks/blob/master/LICENSE).
