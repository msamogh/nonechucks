# nonechucks

**nonechucks** is a library that provides wrappers for PyTorch's datasets, samplers, and transforms to allow for dropping unwanted or invalid samples dynamically.

- [Introduction](#Introduction)
- [Examples](#Examples)
- [Installation](#Installation)
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
3. When you want to let downstream consumers know that a sample is undesirable (with nonechucks, transforms are not restricted to modifying samples; they can drop them as well),
4. When you want your dataset and sampler to be decoupled.

In such cases, it's either simply too expensive to have a separate step to weed out bad samples, or it's just plain impossible because you don't even know what constitutes as "bad", or worse - both!

**nonechucks** allows you to wrap your existing datasets and samplers with "safe" versions of them, which can fix all these problems for you.



<a name="Examples"/>

## Examples

### 1. Dealing with bad samples
Let's start with the simplest use case, which involves wrapping an existing `Dataset` instance with `SafeDataset`.

#### Create a dataset (the usual way)
Using something like torchvision's <a href='https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=folder#torchvision.datasets.ImageFolder'>ImageFolder</a> dataset class, we can load an entire folder of labelled images for a typical supervised classification task.

```python
import torchvision.datasets as datasets
fruits_dataset = datasets.ImageFolder('fruits/')
```
#### Without nonechucks
Now, if you have a sneaky `fruits/apple/143.jpg` (that is corrupted) sitting in your `fruits/` folder, to avoid the entire pipeline from surprise-failing, you would have to resort to something like this:
```python
import random

# Shuffle dataset
indices = list(range(len(fruits_dataset))
random.shuffle(indices)

batch_size = 4
for i in range(0, len(indices), batch_size):
    try:
        batch = [fruits_dataset[idx] for idx in indices[i:i + batch_size]]
        # Do something with it
        pass
    except IOError:
        # Skip the entire batch
        continue
```
Not only do you have to put your code inside an extra `try-except` block, but you are also forced to use a for-loop, depriving yourself of PyTorch's built-in `DataLoader`, which means you can't use features like batching, shuffling, multiprocessing, and custom samplers for your dataset.

I don't know about you, but not being able to do that kind of defeats the whole point of using a data processing module for me.


#### With nonechucks
You can transform your dataset into a `SafeDataset` with a single line of code.
```python
import nonechucks as nc
fruits_dataset = nc.SafeDataset(fruits_dataset)
```
That's it! Seriously.

And that's not all. You can also use a `DataLoader` on top of this.
```python
dataloader = nc.SafeDataLoader(fruits_dataset, batch_size=4, shuffle=True)
for i_batch, sample_batched in enumerate(dataloader):
    # Do something with it
    pass
```
In this case, `SafeDataset` will skip the erroneous image, and use the next one in the place of it (as opposed to dropping the entire batch).

### 2. Use Transforms as Filters!
The function of transorms in PyTorch is restricted to *modifying* samples. With nonechucks, you can simply return `None` (or raise an exception) from the transform's `__call__` method, and nonechucks will drop the sample from the dataset for you, allowing you to use transforms as filters!

For the example, we'll assume a `PDFDocumentsDataset`, which reads PDF files from a folder, a `PlainTextTransform`, which transforms the files into raw text, and a `LanguageFilter`, which retains only documents of a particular language.
```python
class LanguageFilter:
    def __init__(self, language):
        self.language = language
        
    def __call__(self, sample):
        # Do machine learning magic
        document_language = detect_language(sample)
        if document_language != self.language:
            return None
        return sample

transforms = transforms.Compose([
                PlainTextTransform(),
                LanguageFilter('en')
            ])
en_documents = PDFDocumentsDataset(data_dir='pdf_files/', transform=transforms)
en_documents = nc.SafeDataset(en_documents)
```




<a name="Installation" />

## Installation
To install nonechucks, simply use pip:

`$ pip install nonechucks`

or clone this repo, and build from source with:

`$ python setup.py install`.

<a name="Contributing"/>

## Contributing
All PRs are welcome.

<a name="Licensing"/>

## Licensing

**nonechucks** is [MIT licensed](https://github.com/msamogh/nonechucks/blob/master/LICENSE).
