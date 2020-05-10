# Harmonia
Harmonia is the novel B+ tree data-structure specifically designed for GPU to achieve high throughput for basic database operations like search, range queries etc. For more details please refer the following link.
https://dl.acm.org/doi/abs/10.1145/3293883.3295704

# Motivation
B+tree is one of the most important data structures, which has been widely used in different fields such as web
indexing, database, data mining and file system but as the data size increase it puts tremendous pressures on applications
which use B+tree as index data structure. To enhance the performance of B+ tree on GPUs we use novel B+ tree data-structure which bridges the several gaps between the characteristics of GPU and B+ tree.

# Bulit in
Google Colab

# Features Supported
Current implemention support the following operations on the database 
* Insertions
* Search
* Range Queries
* Updates

# Testing
The link given below contains some of the small and large datasets/testcases on which you can test the code.
https://drive.google.com/drive/folders/1sid4JR1GmnBQ1VAN2pIcYM9sAXdnrHC0?usp=sharing

# How to run the code?
**
nvcc -O3 -Xcompiler -fopenmp -Igomp filename.cu

./a.out input.txt A
**

* Compiler the code with given flags to get the best performance.
* At command Line there are two arguments that has to be provided
  1. Input file path
  2. Mode i.e 'A' or 'B'. In 'B' mode you will be able to see the visual output of performing various operations on the database and in mode 'A' you will be able to see the Time taken(performance) to execute each batch of operation. 


