# Harmonia
Harmonia is the novel B+ tree data-structure specifically designed for GPU to achieve high throughput for basic database operations like search, range queries etc. For more details please refer the following link.
https://dl.acm.org/doi/abs/10.1145/3293883.3295704

# Motivation
B+tree is one of the most important data structures, which has been widely used in different fields such as web
indexing, database, data mining and file system but as the data size increase it puts tremendous pressures on applications
which use B+tree as index data structure. Concurrent B+tree queries involve many global memory accesses and different divergences, 
which mismatch with GPU features. Hence, in order to get best out of GPUs authors have designed novel B+ tree data-structure which bridges the several gaps 
between the characteristics of GPU and B+ tree.

# B+ Trees
A B+ tree is an m-ary tree with a variable but often large number of children per node. A B+ tree consists of a root, internal nodes and leaves.
For more info please refer to the following link:
https://en.wikipedia.org/wiki/B%2B_tree

# Built in
Google Colab

# Technology used
C/C++ and CUDA

# Features Supported
Current implemention support the following operations on the database 
* Insertions - Add the Tuple to the Database.
* Search - Find the particular Tuples in the Database with given Key.
* Range Queries - Find all the Tuples with key values falling in particular range.
* Updates - Update the value of particular Tuples specified by key from old value to new value.

# Testing and Sample Example
The link given below contains some of the small and large datasets/testcases on which you can test the code.
https://drive.google.com/drive/folders/1sid4JR1GmnBQ1VAN2pIcYM9sAXdnrHC0?usp=sharing
explain.pdf contains the grapical results showing the time taken by the code on some arbitary GPUs on the above testcases.

# How to run the code?

**nvcc -O3 -Xcompiler -fopenmp -Igomp filename.cu**

**./a.out input.txt A**

* Compile the code with given flags to get the best performance.
* At command Line there are two arguments that has to be provided
  1. Input file path
  2. Mode i.e 'A' or 'B'. In 'B' mode you will be able to see the visual output of performing various operations on the database and in mode 'A' you will be able to see the Time taken(performance) to execute each batch of operation. 

# Example

** nvcc -O3 -Xcomplier -fopenmp -Igomp code.cu**

** ./a.out input1.txt B**

