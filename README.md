# Topic Modelling Competition Submission

The competition was run as part of the internal EYC3 Data Science Reading Club initiative.


### Competition Instructions

The dataset provided are snippets of four books mixed up. Each snippet of 200 words can be considered to be a document, and each book can be considered as a class.

The four books are:
  * Frankenstein (FS)
  * Les Miserables (LM)
  * Walden (WD)
  * The Bible - New Testament (NT)

This is a classification problem where the aim is to assign each snippet to a book using a topic modelling algorithm (LDA).

You should:
  1. Download the data and instructions from
  2. Train an LDA model using the training documents provided. Every document in the training set has a class (i.e. book name), and a 200 word snippet.
  3. Using the testing snippet data set, predict the book of each testing document (i.e. predict the class of each document).

This competition will be ranked on whether or not each document is assigned to the right book.


### My Submission

I have submitted two different approaches and you will find the code for both in this repository:

  1. Official submission created using LDA with `k = 32` (number of topics) and a neuronal network classifier
  2. Secondary and unranked submission created using a custom algorithm based on normalized term frequencies
  

### Rankings

My official submission scored `0.6222` and came out first among 10 participants. My secondary submission scored slightly better with `0.6277`.
