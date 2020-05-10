%%cuda --name filename.cu
#include<cuda.h>
#include<stdio.h>
#include<omp.h>
#include<bits/stdc++.h>
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/generate.h>
#include <thrust/sort.h>
#include<thrust/copy.h>
using namespace std;
#define MAX 33
#define COLUMN 20

//number of pointers or number of child blocks [numberOfPointers = numberOfNodes + 1]
int numberOfPointers = 4;

struct Block{
    //number of nodes
    int tNodes;

    //for parent Block and index
    Block *parentBlock;
    
    //values
    int value[MAX];
    
    //child Blocks
    Block *childBlock[MAX];

    //record pointers
    int *recptr[MAX];

    Block(){    //constructor to initialize a block
        tNodes = 0;
        parentBlock = NULL;
        for(int i=0; i<MAX; i++){
            value[i] = INT_MAX;
            childBlock[i] = NULL;
            recptr[i] = NULL;
        }
    }

};

struct Range{
  int start;
  int end;   
};

struct Updatetuples{
  int key;
  int col;
  int uval;  
};

struct Node{
   int keys[MAX][COLUMN]; 
  int count;

    Node() {
        count = 0;
        for(int i=0 ; i<MAX; i++){
            for(int j=0 ; j<COLUMN ; j++){
            keys[i][j] = INT_MAX;            
            }
        }
    
    }

};

__host__ __device__ 
bool operator<(const Range &lhs, const Range &rhs) 
{
 return ( lhs.start < rhs.start ); 
 };

__host__ __device__ 
bool operator<(const Updatetuples &lhs, const Updatetuples &rhs) 
{
 return ( lhs.key < rhs.key ); 
 };

// creating hamonia structure here
vector <struct Node> key_region;
vector <int> child_prefix_sum;
int psum;                         //psum has to initialize to zero before each insertion batch
// before insertion reorganizing harmonia do 3 things 1. clear key_region 2. clear child_prefix_sum 3. set psum to zero
//end of Harmonia D.S


//declare root Block

Block *rootBlock = new Block();

//function to split the leaf nodes
void splitLeaf(Block *curBlock){
    int x, i, j;

    if(numberOfPointers%2)
        x = (numberOfPointers+1)/2;
    else x = numberOfPointers/2;

    Block *rightBlock = new Block();

    curBlock->tNodes = x;
    rightBlock->tNodes = numberOfPointers-x;
    rightBlock->parentBlock = curBlock->parentBlock;

    for(i=x, j=0; i<numberOfPointers; i++, j++){
        rightBlock->value[j] = curBlock->value[i];
        rightBlock->recptr[j] = curBlock->recptr[i];
        curBlock->value[i] = INT_MAX;
    }
    int val = rightBlock->value[0];
    int *rp = rightBlock->recptr[0];

    if(curBlock->parentBlock==NULL){
        Block *parentBlock = new Block();
        parentBlock->parentBlock = NULL;
        parentBlock->tNodes=1;
        parentBlock->value[0] = val;
        parentBlock->recptr[0] = rp;
        parentBlock->childBlock[0] = curBlock;
        parentBlock->childBlock[1] = rightBlock;
        curBlock->parentBlock = rightBlock->parentBlock = parentBlock;
        rootBlock = parentBlock;
        return;
    }
    else{   
        curBlock = curBlock->parentBlock;

        Block *newChildBlock = new Block();
        newChildBlock = rightBlock;

        for(i=0; i<=curBlock->tNodes; i++){
            if(val < curBlock->value[i]){
                swap(curBlock->value[i], val);
                curBlock->recptr[i] =  rp;
            }
        }

        curBlock->tNodes++;

        for(i=0; i<curBlock->tNodes; i++){
            if(newChildBlock->value[0] < curBlock->childBlock[i]->value[0]){
                swap(curBlock->childBlock[i], newChildBlock);
            }
        }
        curBlock->childBlock[i] = newChildBlock;

        for(i=0;curBlock->childBlock[i]!=NULL;i++){
            curBlock->childBlock[i]->parentBlock = curBlock;
        }
    }

}

//function to split the non leaf nodes
void splitNonLeaf(Block *curBlock){
    int x, i, j;

    x = numberOfPointers/2;

    Block *rightBlock = new Block();

    curBlock->tNodes = x;
    rightBlock->tNodes = numberOfPointers-x-1;
    rightBlock->parentBlock = curBlock->parentBlock;


    for(i=x, j=0; i<=numberOfPointers; i++, j++){
        rightBlock->value[j] = curBlock->value[i];
        rightBlock->recptr[j] = curBlock->recptr[i];
        rightBlock->childBlock[j] = curBlock->childBlock[i];
        curBlock->value[i] = INT_MAX;
        if(i!=x)curBlock->childBlock[i] = NULL;
    }

    int val = rightBlock->value[0];
    int *rp = rightBlock->recptr[0];
    memcpy(&rightBlock->value, &rightBlock->value[1], sizeof(int)*(rightBlock->tNodes+1));
    memcpy(&rightBlock->recptr, &rightBlock->recptr[1], sizeof(int *)*(rightBlock->tNodes+1));
    memcpy(&rightBlock->childBlock, &rightBlock->childBlock[1], sizeof(rootBlock)*(rightBlock->tNodes+1));

    for(i=0;curBlock->childBlock[i]!=NULL;i++){
        curBlock->childBlock[i]->parentBlock = curBlock;
    }
    for(i=0;rightBlock->childBlock[i]!=NULL;i++){
        rightBlock->childBlock[i]->parentBlock = rightBlock;
    }

    if(curBlock->parentBlock==NULL){
        Block *parentBlock = new Block();
        parentBlock->parentBlock = NULL;
        parentBlock->tNodes=1;
        parentBlock->value[0] = val;
        parentBlock->recptr[0] = rp;
        parentBlock->childBlock[0] = curBlock;
        parentBlock->childBlock[1] = rightBlock;

        curBlock->parentBlock = rightBlock->parentBlock = parentBlock;

        rootBlock = parentBlock;
        return;
    }
    else{   
        curBlock = curBlock->parentBlock;

        Block *newChildBlock = new Block();
        newChildBlock = rightBlock;

        for(i=0; i<=curBlock->tNodes; i++){
            if(val < curBlock->value[i]){
                swap(curBlock->value[i], val);
                curBlock->recptr[i] = rp ;
            }
        }

        curBlock->tNodes++;

        for(i=0; i<curBlock->tNodes; i++){
            if(newChildBlock->value[0] < curBlock->childBlock[i]->value[0]){
                swap(curBlock->childBlock[i], newChildBlock);
            }
        }
        curBlock->childBlock[i] = newChildBlock;

         for(i=0;curBlock->childBlock[i]!=NULL;i++){
            curBlock->childBlock[i]->parentBlock = curBlock;
        }
    }

}

void insertNode(Block *curBlock, int val, int *rp){

    for(int i=0; i<=curBlock->tNodes; i++){
        if(val < curBlock->value[i] && curBlock->childBlock[i]!=NULL){
            insertNode(curBlock->childBlock[i], val, rp);
            if(curBlock->tNodes==numberOfPointers)
                splitNonLeaf(curBlock);
            return;
        }
        else if(val < curBlock->value[i] && curBlock->childBlock[i]==NULL){
            swap(curBlock->value[i], val);
            curBlock->recptr[i] =  rp;
            if(i==curBlock->tNodes){
                    curBlock->tNodes++;
                    break;
            }
        }
    }

    if(curBlock->tNodes==numberOfPointers){

            splitLeaf(curBlock);
    }
}

void print(vector < Block* > Blocks){
    vector < Block* > newBlocks;
    for(int i=0; i<Blocks.size(); i++){ //for every block
        Block *curBlock = Blocks[i];
        cout <<"[|";
        int j;
        for(j=0; j<curBlock->tNodes; j++){  
            cout << curBlock->value[j]; 
            cout << "|";
            if(curBlock->childBlock[j]!=NULL)
            newBlocks.push_back(curBlock->childBlock[j]);
        }
        if(curBlock->value[j]==INT_MAX && curBlock->childBlock[j]!=NULL)
            newBlocks.push_back(curBlock->childBlock[j]);

        cout << "]  ";
    }

    if(newBlocks.size()==0){ 

        puts("");
        puts("");
        Blocks.clear();
        //exit(0);
    }
    else {                    
        puts("");
        puts("");
        Blocks.clear();
        print(newBlocks);
    }
}


void createHarmonia(vector < Block* > Blocks, int n){
    vector < Block* > newBlocks;

    for(int i=0; i<Blocks.size(); i++){ //for every block
        Block *curBlock = Blocks[i];
        struct Node t;
        t.count = curBlock->tNodes;
        for(int j =0 ; j<curBlock->tNodes ;j++ ){
          t.keys[j][0] = curBlock->value[j];
          
          for(int k=0; k < n-1 ; k++){
              t.keys[j][k+1] = *(curBlock->recptr[j] + k);
          }
           
        }

        key_region.push_back(t);
        
        int j;
        for(j=0; j<curBlock->tNodes; j++){ 

            if(curBlock->childBlock[j]!=NULL)
            {
            newBlocks.push_back(curBlock->childBlock[j]);
            psum++;
            }

            if(j==0){
                child_prefix_sum.push_back( psum );
            }
        
        }
        if(curBlock->value[j]==INT_MAX && curBlock->childBlock[j]!=NULL){
            newBlocks.push_back(curBlock->childBlock[j]);
            psum++;
        }
    }

    if(newBlocks.size()==0){ 
        Blocks.clear();
    }
    else {                 
        Blocks.clear();
        createHarmonia(newBlocks,n);
    }

}

//search code also includes the implementation of NTG 

__global__ void search( struct Node *a , int *b , int asize , int bsize , int *search_keys, int *mutex , int n , char mode){
    
    // task 1 is to assign individual searches to each threads
    int key = search_keys[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    
    // task 2 is to perform search using harmonia
    while(true){
       
        if( index > asize-1 ){
            break;
        }
        prev_index = index;
        // need to divide this for loop among several threads to implement NTG.
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == key ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < key  && key < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > key ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < key ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }
 
  if(threadIdx.x == 0) {
      index = prev_index;
    int flag_found = 0;
    for(int i=0; i< a[index].count; i++){
        if( a[index].keys[i][0] == key  ){
                flag_found = 1;
      if(mode == 'B'){
           while( atomicCAS(mutex,0,1) != 0  );
                printf("\n"); 
                for(int k=0 ; k < n ; k++)
                printf("%d " , a[index].keys[i][k]  );
                printf("\n"); 
       atomicExch(mutex,0);                
      }
      
        }
    }
    if(!flag_found){
        if(mode == 'B'){
            printf("\nRecord doesn't exists with given key %d\n" , key);
        }
    }

  }
   
   

}

//update code also includes the implementation of NTG 
__global__ void update( struct Node *a , int *b , int asize , int bsize , struct Updatetuples *update_tuples , int *mutex , char mode ){
    
    // task 1 is to assign individual updates to each threads
    struct Updatetuples tp = update_tuples[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    // task 2 is to perform search using harmonia
    while(true){
       
        if( index > asize-1 ){
            break;
        }
        prev_index = index;
        // need to divide this for loop among several threads to implement NTG.
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == tp.key ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < tp.key  && tp.key < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > tp.key ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < tp.key ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }

    // task 3 is to update the value by new one

    if( threadIdx.x == 0){
        index = prev_index;
        int flag_found = 0;
    
        for(int i=0; i< a[index].count; i++){
          if( a[index].keys[i][0] == tp.key  ){
                flag_found = 1;
                
                while( atomicCAS(mutex,0,1) != 0  );   //atomicity is needed while updation.
                a[index].keys[i][tp.col-1] = tp.uval;
                atomicExch(mutex,0);

               if(mode == 'B'){
                   printf("\nrecord updated successfully\n");
               }
        }
    }

    if(!flag_found){
        if(mode == 'B'){
        printf("\nRecord doesn't exists with given key %d\n" , tp.key);    
        }
    }
  }
   
    
}

//rangeQuery code also includes the implementation of NTG 
__global__ void rangeQuery( struct Node *a , int *b , int asize , int bsize , struct Range *range_arr , int *mutex ,int n , char mode ){
    
    // task 1 is to assign individual search Ranges to each threads
    struct Range r = range_arr[blockIdx.x];
    __shared__ int index ;
    __shared__ int prev_index;

    index = 0;
    prev_index = 0;

    __syncthreads();
    
    
    // task 2 is to perform search using harmonia
    while(true){
       
        if( index > asize-1 ){
            break;
        }
        prev_index = index;
        // need to divide this for loop among several threads to implement NTG.
        
        if( threadIdx.x < a[index].count ){
            if( a[index].keys[ threadIdx.x][0] == r.start ){
                index = b[index] + threadIdx.x + 1;
                goto bottom;   
            }
            if(threadIdx.x != a[index].count - 1)
            if( a[index].keys[ threadIdx.x][0] < r.start  && r.start < a[index].keys[threadIdx.x + 1][0]  ){
                index = b[index] + threadIdx.x +1;
                goto bottom;
            }
            if(threadIdx.x == 0){
                 if( a[index].keys[0][0] > r.start ){
                index = b[index] + 0;
                goto bottom;
                }
                if( a[index].keys[ a[index].count-1 ][0] < r.start ){
                index = b[index] + a[index].count;
                goto bottom;
                }

            }
           
        }
      bottom:  __syncthreads();
   }

  // print tuples in the given range

  if(threadIdx.x == 0){
      int i = prev_index;
      int j = 0;

      if(mode == 'B'){
          while( atomicCAS(mutex,0,1) != 0  );  // atomicity is needed to ensure that all print from one range are together else output will be jumbled
        printf("----------------------------------------------------");
      }
        
      while(true){
      
        if( a[i].keys[j][0] > r.end )
          break;     

        if( i > asize - 1 )
          break;

        if(  r.start <= a[i].keys[j][0] && a[i].keys[j][0] <= r.end ){
           
           if(mode == 'B'){
              printf("\n");
                 for(int k=0 ; k < n ; k++){
                  printf("%d " , a[i].keys[j][k] );   
                 }
                printf("\n");      
           }
              
        }
        j++;

        if( j >= a[i].count ){
        i++;
        j=0;   
        }
      
    }
    if(mode == 'B'){
     atomicExch(mutex,0);   
    }
  }
  
}


template<class T>
__device__ T plus_scan(T *x)
{
    unsigned int i =  threadIdx.x; // id of thread executing this instance
    unsigned int n =  blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        T t;

        if ( i >= offset ) 
            t = x[i-offset];

        __syncthreads();

        if ( i >= offset ) 
            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]

       __syncthreads();
    }
    return x[i];
}


__device__ void partition_by_bit(int *values, int bit)
{
    int i = threadIdx.x ;
    
    int size = blockDim.x;
    int x_i = values[i];          // value of integer at position i
    int p_i = (x_i >> bit) & 1;   // value of bit at position bit

    values[i] = p_i;  

   __syncthreads();

    int T_before = plus_scan(values);
  
    int T_total  = values[size-1];
   
    int F_total  = size - T_total;

    __syncthreads();

    if ( p_i )
        values[T_before-1 + F_total] = x_i;
    else
       values[i - T_before] = x_i;

}


//perform radix sort to get partial sorted aggregation (PSA)
 __global__ void radix_sort( int *values ,int loop )
{   
    int bit;
    for(int j=0 ; j<loop ; j++){
    for( bit = 5; bit < 32; ++bit )
    {
        
        partition_by_bit(values+(j*1024) , bit);
        __syncthreads(); 
    }
  }
        
}

int main(int argc , char **argv){
    
    FILE *filePointer;
    char *filename = argv[1]; 
    char mode = *argv[2];
    filePointer = fopen( filename , "r") ; 
    
    if ( filePointer == NULL ) 
    {
        printf( "input.txt file failed to open." ) ; 
    }
    freopen("output.txt", "w", stdout);

    vector < Block* > Blocks;

    int totalValues = 0;
    int m,n;
    
    fscanf(filePointer, "%d", &m );
    fscanf(filePointer, "%d", &n );

    int **database = (int **)malloc( 5 * m * sizeof(int *)); 
    for(int i=0; i< 5*m; i++) 
         database[i] = (int *)malloc(n * sizeof(int)); 


    //-------------------Initial Insertions form DB-----------------------
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            fscanf(filePointer, "%d" , &database[i][j] );
        }
    }

    // creating cpu b+ tree here
    for(int i=0;i<m;i++){
        
            insertNode(rootBlock, database[i][0] , &database[i][1] );
            totalValues++;
    }

            // create the harmonia 
            key_region.clear();
            child_prefix_sum.clear();
            psum=0;
        
            Blocks.clear();
            Blocks.push_back(rootBlock);
            createHarmonia(Blocks,n);
            puts("");
            for(int i=0 ; i<child_prefix_sum.size() ; i++  )
            {
                if( child_prefix_sum[i] == child_prefix_sum.size() - 1 )
                {
                    child_prefix_sum[i] = child_prefix_sum.size() ;
                }
            }
    //---------------------Initial Insertions end---------------------------

    int *mutex;
    struct Node *gpuA;
    int *gpuB;
    cudaMalloc(&mutex , sizeof(int) );
    cudaEvent_t start, stop;
    float milliseconds;

    int numofops;
    int ch;

    fscanf(filePointer, "%d" , &numofops );    

   for(int q=0; q < numofops ; q++ ){
        fscanf(filePointer, "%d" , &ch );
            
        if(ch == 0){
             //---------------------Prints------------------------------------------
  

             // print the B+ tree
            Blocks.clear();
            Blocks.push_back(rootBlock);
            print(Blocks);
            puts("");
            cout << "\n";

            for(int i=0; i<key_region.size() ; i++){

                 for(int j=0 ; j<key_region[i].count ; j++ ){
                
                    for(int k=0 ; k<n ; k++){
                    cout << key_region[i].keys[j][k]  << " "; 
                    }
                
                }
                cout << "\n";
            }
            

            cout << "\n";
            for(int i=0 ; i<child_prefix_sum.size() ; i++  )
            {
                cout << child_prefix_sum[i]  << " ";
            }
        //---------------------------prints end--------------------------------
        }
         
         else if(ch == 1){
             //-----------------------------search op---------------------------------------
        int Ssize;
        fscanf(filePointer, "%d" , &Ssize );
        int *search_arr = (int *)malloc( Ssize*sizeof(int) );            
            
            for(int i=0 ; i< Ssize ; i++){
               fscanf(filePointer, "%d" , &search_arr[i] );  
            }   
            
           //thrust::sort( search_arr, search_arr + Ssize  );
            int *gpuC;
            int threads;
            int loop;

            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuC , Ssize*sizeof(int) );

            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuC , search_arr , Ssize*sizeof(int) , cudaMemcpyHostToDevice ); 

            if(Ssize<=1024){
                threads = Ssize;
                loop = 1;
            }
            else{
                threads = 1024;
                loop= Ssize/1024;
            }
           radix_sort<<<1,threads>>>(gpuC,loop);

            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);    
            }
                
            search<<< Ssize, (numberOfPointers-1) >>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuC , mutex , n , mode );
            //cudaDeviceSynchronize();

            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }

            //------------------------search end--------------------------------------------

         }

        else if(ch == 2)
        {
            //--------------------------Range Query----------------------------------------
            
            int Rsize;
            fscanf(filePointer, "%d" , &Rsize );             
            //Range range_arr[Rsize];
            Range *range_arr = (Range *)malloc( Rsize*sizeof(Range) );
         
            for(int i=0 ; i< Rsize ; i++){
               fscanf(filePointer, "%d" , &range_arr[i].start );
               fscanf(filePointer, "%d" , &range_arr[i].end );
            }
         
            //thrust::sort(range_arr, range_arr + Rsize );
         
            struct Range *gpuD;
            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuD, Rsize*sizeof(struct Range) );

            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy(gpuD , range_arr , Rsize*sizeof(struct Range) , cudaMemcpyHostToDevice );

            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);    
            }
            
            rangeQuery<<<Rsize,(numberOfPointers-1)>>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuD , mutex , n , mode );
           // cudaDeviceSynchronize();
         
            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);    
            }
            
            //-------------------------Range Query end-------------------------------------

        }

        else if( ch == 3)
        {
        //------------------------update operation-------------------------------------

            int Usize;
            fscanf(filePointer, "%d" , &Usize ); 
            //Updatetuples tp[Usize]; 
            Updatetuples *tp = (Updatetuples *)malloc( Usize*sizeof(Updatetuples) );

            for(int i=0 ; i< Usize ; i++){
               fscanf(filePointer, "%d" , &tp[i].key );
               fscanf(filePointer, "%d" , &tp[i].col );
               fscanf(filePointer, "%d" , &tp[i].uval );  
            }
         
            //thrust::sort(tp, tp + Usize );
            
            Updatetuples *gpuE;
            cudaMemset(mutex,0,sizeof(int));

            cudaMalloc(&gpuA , key_region.size() * sizeof(struct Node) );
            cudaMalloc( &gpuB , child_prefix_sum.size() * sizeof(int) );
            cudaMalloc( &gpuE, Usize*sizeof(struct Updatetuples) );
            
            cudaMemcpy( gpuA , &key_region[0] , key_region.size() * sizeof(struct Node) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuB , &child_prefix_sum[0] , child_prefix_sum.size() * sizeof(int) , cudaMemcpyHostToDevice);
            cudaMemcpy( gpuE , tp , Usize*sizeof(struct Updatetuples) , cudaMemcpyHostToDevice );
            
            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            }
    
            update<<<Usize,(numberOfPointers-1)>>>( gpuA , gpuB , key_region.size() , child_prefix_sum.size() , gpuE ,mutex, mode );
            cudaMemcpy( &key_region[0], gpuA , key_region.size() * sizeof(struct Node) , cudaMemcpyDeviceToHost);
            //cudaDeviceSynchronize();
         
            //update the original database after the update operation and before insertion.
         
         // parallelize 2 loops using openMP 

            #pragma omp parallel for
            for(int i=0 ; i < Usize ; i++ ){ 
                for(int j=0 ; j< m ; j++){
                    if( tp[i].key == database[j][0] ){
                       #pragma omp critical
                          {
                          database[j][ tp[i].col - 1 ] = tp[i].uval;
                          }
                        
                    }
                }
            }
         
            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }
            

        //------------------------update operation end---------------------------------
        }

        else if(ch == 4)
        {
         //-------------------------Insertion opertions----------------------------------

            int Isize;
            fscanf(filePointer, "%d" , &Isize ); 

            //step 1 perform the series of insertions i.e batch insertion
            // replace 5 with Isize later
            for(int i=m;i<m+Isize;i++){
              for(int j=0;j<n;j++){
                fscanf(filePointer, "%d" , &database[i][j] );
              }
            }

            if(mode == 'A'){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
            }
            
            for(int i=m;i<m+Isize;i++){
            insertNode(rootBlock, database[i][0] , &database[i][1] );
            totalValues++;
            }

             //last thing to be done before step 2 is to update value of m
             m = m+Isize;

            //step 2 is to reconstruct the harmonia after insertion
            
            //firstly clear the whole harmonia D.S
            key_region.clear();
            child_prefix_sum.clear();
            psum=0;

            Blocks.clear();
            Blocks.push_back(rootBlock);
            createHarmonia(Blocks,n);
            puts("");
            for(int i=0 ; i<child_prefix_sum.size() ; i++  )
            {
                if( child_prefix_sum[i] == child_prefix_sum.size() - 1 )
                {
                    child_prefix_sum[i] = child_prefix_sum.size() ;
                }
            }
         
            if(mode == 'A'){
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("\n%f\n",milliseconds);
            }
            
            //-------------------------Insertion End----------------------------------------
        
        }
    
    }
       
      return 0;
}

