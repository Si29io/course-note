[TOC]

# About

This note is for the class "成功前進科技巨頭，百萬年薪面試術" at https://hahow.in/cr/getoffer

It covers the most common algorithms to solve most interview questions. 
It include notes, homework, solutions, and some advanced materials that is optional for you to read. 
Please make sure you understand the materials in the video first, because it appears in the interview more frequently.

Friendly reminder: 
Extend this note to make your own note as you like
review your own note before the interview
If you're not sure how to solve problem, write down ideas before write code
Write down acutal number/data helps you debug
You can use this edior [Typora](https://typora.io/) to read and edit this note easily.

By Dev in Si29

# Data Structure

## Python3 built-in CheatSheet

```python
#global (no import needed) 
list
tuple # like an immutable list
dict
set
frozenset # immutable set
str
int
ord # ord('a') is 97
chr # chr(97) is 'a'
bin # bin(2) is '0b10'
hex # hex(15) is '0xf'
type #show what class the obj is
iter 
next
dir # list object's methods, if you forget the methods

from collections import *
deque
Counter # similar to defaultdict(int)
defaultdict # dict with default value
OrderDict # dict that has an order, HashMap + Linked List

from queue import *
Queue
PriorityQueue

import heapq
heapq # the heap impliment PriorityQueue. In most cases, PriorityQueue is enough.

from random import *
random
randrange
choice 

# operator
/	  # Division
//	# Floor division

# bit operation
& 	# AND	Sets each bit to 1 if both bits are 1
|	  # OR	Sets each bit to 1 if one of two bits is 1
^	  # XOR	Sets each bit to 1 if only one of two bits is 1
~ 	# NOT	Inverts all the bits
<<	# Zero fill left shift	
>>	# Signed right shift	

* in function # fun(*A) = fun(A[0], A[1], A[3],....)

# to impliment built-in compare in python3
__lt__; stands for the less-than sign ( < )
__gt__; stands for the greater-than sign ( > )
__eq__: stands for the equal (==)

```



## Time and Space Complex CheatSheet

Common Data Structure Operations

| Data Structure                                               | Time Complexity |             |             |             |        |        |           |          | Space Complexity |
| :----------------------------------------------------------- | :-------------- | :---------- | :---------- | :---------- | :----- | :----- | :-------- | :------- | :--------------- |
|                                                              | Average         |             |             |             | Worst  |        |           |          | Worst            |
|                                                              | Access          | Search      | Insertion   | Deletion    | Access | Search | Insertion | Deletion |                  |
| [Array](http://en.wikipedia.org/wiki/Array_data_structure)   | `Θ(1)`          | `Θ(n)`      | `Θ(n)`      | `Θ(n)`      | `O(1)` | `O(n)` | `O(n)`    | `O(n)`   | `O(n)`           |
| [Stack](http://en.wikipedia.org/wiki/Stack_(abstract_data_type)) | `Θ(n)`          | `Θ(n)`      | `Θ(1)`      | `Θ(1)`      | `O(n)` | `O(n)` | `O(1)`    | `O(1)`   | `O(n)`           |
| [Queue](http://en.wikipedia.org/wiki/Queue_(abstract_data_type)) | `Θ(n)`          | `Θ(n)`      | `Θ(1)`      | `Θ(1)`      | `O(n)` | `O(n)` | `O(1)`    | `O(1)`   | `O(n)`           |
| [Singly-Linked List](http://en.wikipedia.org/wiki/Singly_linked_list#Singly_linked_lists) | `Θ(n)`          | `Θ(n)`      | `Θ(1)`      | `Θ(1)`      | `O(n)` | `O(n)` | `O(1)`    | `O(1)`   | `O(n)`           |
| [Doubly-Linked List](http://en.wikipedia.org/wiki/Doubly_linked_list) | `Θ(n)`          | `Θ(n)`      | `Θ(1)`      | `Θ(1)`      | `O(n)` | `O(n)` | `O(1)`    | `O(1)`   | `O(n)`           |
| [Hash Table](http://en.wikipedia.org/wiki/Hash_table)        | `N/A`           | `Θ(1)`      | `Θ(1)`      | `Θ(1)`      | `N/A`  | `O(n)` | `O(n)`    | `O(n)`   | `O(n)`           |
| [Binary Search Tree](http://en.wikipedia.org/wiki/Binary_search_tree) | `Θ(log(n))`     | `Θ(log(n))` | `Θ(log(n))` | `Θ(log(n))` | `O(n)` | `O(n)` | `O(n)`    | `O(n)`   | `O(n)`           |



Sorting Algorithms

| Algorithm                                                    | Time Complexity |               | Space Complexity |             |
| :----------------------------------------------------------- | :-------------- | :------------ | :--------------- | :---------- |
|                                                              | Best            | Average       | Worst            | Worst       |
| [Quicksort](http://en.wikipedia.org/wiki/Quicksort)          | `Ω(n log(n))`   | `Θ(n log(n))` | `O(n^2)`         | `O(log(n))` |
| [Mergesort](http://en.wikipedia.org/wiki/Merge_sort)         | `Ω(n log(n))`   | `Θ(n log(n))` | `O(n log(n))`    | `O(n)`      |
| [Heapsort](http://en.wikipedia.org/wiki/Heapsort)            | `Ω(n log(n))`   | `Θ(n log(n))` | `O(n log(n))`    | `O(1)`      |
| [Bubble Sort](http://en.wikipedia.org/wiki/Bubble_sort)      | `Ω(n)`          | `Θ(n^2)`      | `O(n^2)`         | `O(1)`      |
| [Insertion Sort](http://en.wikipedia.org/wiki/Insertion_sort) | `Ω(n)`          | `Θ(n^2)`      | `O(n^2)`         | `O(1)`      |
| [Selection Sort](http://en.wikipedia.org/wiki/Selection_sort) | `Ω(n^2)`        | `Θ(n^2)`      | `O(n^2)`         | `O(1)`      |
| [Bucket Sort](http://en.wikipedia.org/wiki/Bucket_sort)      | `Ω(n+k)`        | `Θ(n+k)`      | `O(n^2)`         | `O(n)`      |
| [Radix Sort](http://en.wikipedia.org/wiki/Radix_sort)        | `Ω(nk)`         | `Θ(nk)`       | `O(nk)`          | `O(n+k)`    |
| [Counting Sort](https://en.wikipedia.org/wiki/Counting_sort) | `Ω(n+k)`        | `Θ(n+k)`      | `O(n+k)`         | `O(k)`      |



## Heap

1. Insert: insert at bottom, sift-up. O(log n)
2. Delete: swap top one with bottom one, sift-up (when sift-up, swap with its smaller child in a min-heap and its larger child in a max-heap.) O(log n)

```
Notice heap is an array considered as a tree, so when append a node to the array, it is alway at bottom right, so don't need to worry which node to append.
```

## Iterator

The reason to impliment an iterator `iter()` is to average the time complexity to every `next()` operation. e.g To pring an very large tree

## Number of nodes in balanced binary tree

N = number of all nodes
N = 1 + 2 + .... + 2^height
N*2 = 2 + ... + 2^(height+1)
2N-N=N= 2^(height+1) - 1
(N + 1)/2 = 2^heigh
**height = log (N+1)/2 ~= log N**
**width = 2^height = (N+1)/2**

## LRU

use DLL + dict, See 146 LRU Cache 

## LFU

Each count (freqency) has a Linked list, that can be done by `a dict of OrderedDicts`. Remember to record min count to know which node to remove. Define a `dict` count2nodes, contain `OrderedDict`, so you can look up by `count2nodes[count][key]` to remove/update the node in O(1), or `count2nodes[count].popitem(last=True)` to remove the oldest node in O(1).

```python
from collections import defaultdict
from collections import OrderedDict

class Node:
    def __init__(self, key, val, count):
        self.key = key
        self.val = val
        self.count = count
    
class LFUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity
        self.key2node = {}
        self.count2node = defaultdict(OrderedDict)
        self.minCount = None
        
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.key2node:
            return -1
        
        node = self.key2node[key]
        del self.count2node[node.count][key]
        
        # clean memory
        if not self.count2node[node.count]:
            del self.count2node[node.count]
        
        node.count += 1
        self.count2node[node.count][key] = node
        
        # NOTICE check minCount
        if not self.count2node[self.minCount]:
            self.minCount += 1
            
            
        return node.val
        
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if not self.cap:
            return 
        
        if key in self.key2node:
            self.key2node[key].val = value
            self.get(key) # NOTICE, put makes count+1 too
            return
        
        if len(self.key2node) == self.cap:
            # popitem(last=False) is FIFO, like queue
            # it return key and value!!!
            k, n = self.count2node[self.minCount].popitem(last=False) 
            del self.key2node[k] 
        
        self.count2node[1][key] = self.key2node[key] = Node(key, value, 1)
        self.minCount = 1
        return
```

## Homework

```
706 Design HashMap
146 LRU Cache 
215 Kth Largest Element in an Array
705	Design HashSet    
```



## Solutions

### 706 Design HashMap

Chaining: add node in linked list 
Linear probing: put to next empty slot in the array

```python
from collections import deque

class Node:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
    
    def __eq__(self, n):
        return self.key == n.key

class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """        
        self.SIZE = 1000
        self.table = [deque() for _ in range(self.SIZE)]
        
    def hash(self, key):
        # 123 -> 1*33^2 + 2*33^1 + 3*33^0
        r = 0
        for ch in str(key):
            r = (int(ch) + r*33) % self.SIZE
        return r #key % self.SIZE

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        index = self.hash(key)
        if Node(key) in self.table[index]:
            self.table[index].remove(Node(key))
            
        self.table[index].append(Node(key, value))

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        index = self.hash(key)
        if Node(key) in self.table[index]:
            #dq.index(n2)  dq=[n1->n2->n3] 1
            i = self.table[index].index(Node(key))
            return self.table[index][i].value
 
        return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        index = self.hash(key)
        if Node(key) in self.table[index]:
            self.table[index].remove(Node(key))


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```

### 146 LRU Cache 

```python
class Node:
    def __init__(self, k=None, v=None):
        self.key = k
        self.value = v
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.dic = {}
        self.head = Node()
        self.tail = Node()
        self.head.next, self.tail.prev = self.tail, self.head
    
    def _add(self, n):
        l, r = self.tail.prev, self.tail
        l.next = r.prev = n
        n.prev, n.next = l, r
        
    def _remove(self, n):
        l, r = n.prev, n.next
        l.next, r.prev = r, l
        
    def _pop(self):
        n = self.head.next
        self._remove(n)
        return n

    def get(self, key: int) -> int:
        if key in self.dic:
            n = self.dic[key]
            self._remove(n)
            self._add(n)
            return n.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self._remove(self.dic[key])
        n = Node(key, value)
        self._add(n)
        self.dic[key] = n
        if len(self.dic) > self.cap:
            del self.dic[self._pop().key]


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



### 215 Kth Largest Element in an Array

```python
from queue import PriorityQueue

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return self.quickSelect(nums, 0, len(nums)-1, len(nums)-k)
        
    # Quick Select
    #[...]pivot[...]
    #[...]
    def quickSelect(self, nums, start, end, k):
        pivot = nums[(start+end)//2]
        l, r = self.partition3way(nums, start, end, pivot)
        if l <= k <= r: return nums[k] #[..l.k..r....]
        if k < l: return self.quickSelect(nums, start, l-1, k)
        if k > r: return self.quickSelect(nums, r+1, end, k)
        
    def partition3way(self, A, l, r, t):
        m = l
        while m <= r:
            if A[m] < t:
                A[m], A[l] = A[l], A[m]
                m += 1
                l += 1
            elif A[m] > t:
                A[m], A[r] = A[r], A[m]
                r -= 1
            else:
                m += 1
        return l, r
    
   
    # Heap
    # T: O(n log k)
    # S: O(k)
    def findKthLargestPriorityQueue(self, nums: List[int], k: int) -> int:
        pq = PriorityQueue()#min heap
        for n in nums:
            pq.put(n)
            if pq.qsize() > k:
                pq.get()
```

# Array

## Sliding Window

See 3 Longest Substring Without Repeating Characters

## Binary Search (BS)

See 34 Find First and Last Position of Element in Sorted Array

## Three Way Partition

See 75 Sort Colors  

```
# Three way partition output analysis
l is the index of the left side boundary of the target
r is the index of the right side boundary of the target
m is the traveling index at middle (most of the time)

in: A=[3, 2, 2, 1] t=2
out: A=[1, 2, 2, 3] (l,r)=(1, 2) m=3

in: A=[2, 2, 2] t=2
out: A=[2, 2, 2] (l,r)=(0, 2) m=3

in: A=[3, 2, 1] t=2
out: A=[1, 2, 3] (l,r)=(1, 1) m=2

Notice when target is NOT in A, 
It makes l > r and the l or the r will be out of the boundary (at the side t could belongs to). 

in: A=[3, 1, 2] t=4
out: A=[3, 1, 2] (l,r)=(3, 2) m=3

in: A=[3, 1, 2] t=0
out: A=[1, 2, 3] (l,r)=(0, -1) m=0
```



## Scan overlays of intervals 

253	Meeting Rooms II (locked by leetcode)

```
Given intervals [[start, end], [start, end], [start, end]...]
To calculate the number of overlay meetings at a given time, 
build start and end events and sort them
[(start, 1), (start, 1), (end, -1), (end, -1), (start, 1), (end, -1)....]
Add up the values to get number of avaliable meeting rooms 
```

## Start at end

Remember to think about if starting from end is better
Ex [merge-sorted-array](https://leetcode.com/problems/merge-sorted-array/) to one of the arrays. Inserting at front needs to move the whole elements. Starting at the end is much better. Or [daily-temperatures](https://leetcode.com/problems/daily-temperatures/), that start at end too.

## Kadane's algorithm (optional)

  https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

  ```python
  '''121. Best Time to Buy and Sell Stock'''
  class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # turn prices to delta prices
        prices = [prices[i]-prices[i-1] for i in range(1, len(prices))]

        # Kadane's algorithm
        maxHere = maxSoFar = 0
        for p in prices:
            maxHere = max(p, maxHere + p)
            maxSoFar = max(maxHere, maxSoFar)

        return maxSoFar
  ```

  ```python
  '''152. Maximum Product Subarray'''
  def maxProduct(self, nums):
        minHere = maxHere = maxSoFar = nums[0]
        for n in nums[1:]:
            maxHere, minHere= max(n, maxHere * n, minHere * n), min(n, maxHere * n, minHere * n)
            maxSoFar = max(maxHere, maxSoFar)

        return maxSoFar
  ```

## Homework

```
3 Longest Substring Without Repeating Characters
567 Permutation in String
209 Minimum Size Subarray Sum

34 Find First and Last Position of Element in Sorted Array
278 First Bad Version
33 Search in Rotated Sorted Array

75 Sort Colors  
912 Sort an Array

253	Meeting Rooms II (locked by leetcode)
```



## Solutions

### 3 Longest Substring Without Repeating Characters

```python
from collections import deque

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        Sliding Window
        1. Eat: If food is new food, eat.
        2. Diguest: Record the the maximum of food in the body.
        3. Check condition to poop:
            If the food is eaten, poop till the eaten food come out.
        '''
        q = deque()
        maxLen = 0
        for ch in s:
            #3. poop
            while ch in q:
                q.popleft()
            
            #1. eat
            q.append(ch)
            #2. diguest
            maxLen = max(maxLen, len(q))
        
        return maxLen
```

### 34 Find First and Last Position of Element in Sorted Array

```python
class Solution:
    def searchRange(self, A: List[int], target: int) -> List[int]:
        if not A:
            return [-1, -1]
        
        l, r = self.binarySearch(A, target, True)
        if target not in (A[l], A[r]):
            return [-1, -1]
        
        left = l if A[l]==target else r
        
        l, r = self.binarySearch(A, target, False)
        right = r if A[r]==target else l
        
        return [left, right]
    
    def binarySearch(self, A, target, isLeft):
        l, r = 0, len(A) - 1
        while l + 1 < r: #A=[....l,r...]
            m = (l + r) // 2
            if A[m] < target:
                l = m
            elif A[m] > target:
                r = m
            else:
                if isLeft:
                    r = m
                else:
                    l = m
        return l, r
```



### 75 Sort Colors

```python
class Solution:
    def sortColors(self, A: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        Three Way Partition
        A=[-1,9,2,0,1,1,1] t=1
              l       r
              m
        '''
        l, m, r = 0, 0, len(A)-1
        t = 1
        while m <= r: #
            if A[m] < t:
                A[m], A[l] = A[l], A[m]
                l += 1
                m += 1
            elif A[m] > t:
                A[m], A[r] = A[r], A[m]
                r -= 1
            else:
                m += 1
```

### 912 Sort an Array

```python
class Solution:
    def sortArray(self, A: List[int]) -> List[int]:
        '''
        Qucik Sort
        [5,2,3,1] pivot=2
        [1]2[3,5]
        1,2[3,5] pivot=3
        1,2,3,5 
        '''
        self.quickSort(A, 0, len(A)-1)
        return A
        
    def quickSort(self, A, L, R):
        if L >= R: return
        t = A[(L+R)//2]
        l, r = self.partition(A, L, R, t)
        self.quickSort(A, L, l-1)
        self.quickSort(A, r+1, R)
       
    
    def partition(self, A, l, r, t):
        m = l
        while m <= r: #
            if A[m] < t:
                A[m], A[l] = A[l], A[m]
                l += 1
                m += 1
            elif A[m] > t:
                A[m], A[r] = A[r], A[m]
                r -= 1
            else:
                m += 1
        return l, r
                
    
    def countSort(self, A):
        # write your code here
        mini = min(A)
        maxm = max(A)
        C = Counter(A)
        i = 0
        for k in range(mini, maxm+1):
            while C[k]: 
                A[i] = k
                i += 1
                C[k] -= 1
        return A
                
    def bucketSort(self, A):
        buckets = [[] for _ in A]
        mini = min(A)
        maxm = max(A)
        rang = maxm - mini
        if not rang: return A
    
        for a in A:
            i = int( (a - mini) * (len(A)-1)/rang )
            buckets[i].append(a)
        
        i = 0
        for b in buckets: # each bucket has the size O(1), so sorting is O(1) too
            b.sort(reverse=True) 
            while b:
                A[i] = b.pop()
                i += 1
        return A
```

# Sort

## Quick Sort by 3 way partition

Lomuto scheme always uses the rightest to pivot.  This scheme degrades to *O*(*n*2) when the array is already in order.
Hoare scheme doesn't always return the pivot index and hard to implement 
Both Lomuto and Hoare doesn't tell you if the mutiple pivot value exist where is the boundary
The three way partition for quick sort is much better for interview. 

**Time Complexity:** Avg O(nlogn) worse O(n^2) you shuffle the array to avoid worse case
**Space:** O(1)

```python
class Solution:
    def sortIntegers2(self, A):
        self.quickSort(A, 0, len(A)-1)
    '''
    My 3 way partition quickSort, better than Lomuto and Hoare
    '''
    def quickSort(self, A, L, R):
        if L >= R: return
        t = A[(L+R)//2]
        l, r = self.partition(A, L, R, t)
        self.quickSort(A,L,l-1)
        self.quickSort(A,r+1,R)

    def partition(self, A, l, r, t):
        m = l
        while m <= r:
            if A[m] < t:
                A[l], A[m] = A[m], A[l]
                m += 1
                l += 1
            elif A[m] > t:
                A[r], A[m] = A[m], A[r]
                r -= 1
            else:
                m += 1

        return l,r
```



## Top kth element solutions compare

1. **Quick Select** It's like QucikSort that only sort one path

   Time: O(n) on avg, O(n^2) worse case,  Space: O(1)

   but you can randomly shuffle elements to be O(n), best sol

   Pro: O(n) is fast, with O(1) memory, the best solution

   Con: A little bit harder to write



2. **Quick Sort** Time: O(n log n) , if Quick Sort worse case O(n^2), Space: O(1)

   Pro: easy

   Con: slower



3. **Count Sort** Time: O(n), Space: O(n)

   if there is a range of the elements O(n)

   Pro: easy and fast

   Con: only for elements in small range



4. **Min Heap**, 
   Time: O(k + (n-k) log k) = O((n-k) log k) ~= O(n log k), 
   Space: O(k)

   Using a size k min heap maintain the top k.  Remove the min

   init k element in heap O(k), for loop the rest of the elements in the array O(n-k) 
   and pop out the min element to put the bigger element log(k) for n-k times.

   Pro: Smaller memory, good for dynamically update/delete/add
   Con: Can't apply to bigger k if you want to re-use the heap



5. **Max Heap**, Time: O(n + k log n), Space: O(n)

   Just put all elements in heap and pop out k times
   
   

   build heap O(n), pop-out k times O(k log n), so O(n + k log n)

   it is just heap sort that stop at k-th popout

   Pro: easy to write, good for dynamically update/delete/add

   Con: bigger memory



6. **BST** (red-black tree, AVTree), search the k-th O(k), add new element O(log k), Space: O(n)

   Notice, avoid write BST in interview

   Pro: good for dynamically update/delete/add and get all ranked top elements, not just k-th one

   Con: slower. No built-in BST in python. It's too hard to write in interview.

**You can ask interviewer these:**

* If it is a steam (dynamically update/delete/add array)?
* Call get k repeatedlly with different k?
* all (sorted/not sorted) top k elements? Partial Quick Sory/Heap
* range of the element?  small range can use count sort.

```python
import random
import heapq

class Solution(object):

    # Min Heap O(k+(n-k)*log(k))
	def findKthLargest(self, nums, k):
        heap = nums[:k]
        heapq.heapify(heap)  # create a min-heap whose size is k
        for num in nums[k:]:
            if num > heap[0]:
               heapq.heapreplace(heap, num)
            # or use:
            # heapq.heappushpop(heap, num)
    return heap[0]

```



## Count Sort

Use it when it's a range of integers. https://www.geeksforgeeks.org/counting-sort/ 
Time: O(n + range) 

```python
def countSort(self, A):
        # write your code here
        mini = min(A)
        maxm = max(A)
        C = Counter(A)
        i = 0
        for k in range(mini, maxm+1):
            while C[k]: 
                A[i] = k
                i += 1
                C[k] -= 1
        return A
```



## Bucket Sort

Use it when input uniformly distributed over a range, like 0.0 to 1.0. (you can multiply 10 or anything to map to the bucket) https://www.geeksforgeeks.org/bucket-sort-2/

```text
1. make Array<Node> Buckets with length n, each of them contain empty linked list  
2. map each val of A[i] to the index of bucket by int( (val - low) * n/range )
3. append to the linked list inside the bucket
4. collect everything and sort each linked list

Beacuse each linked list has the size O(1), sorting it is O(1) too

```

```python
def bucketSort(self, A):
        buckets = [[] for _ in A]
        mini = min(A)
        maxm = max(A)
        rang = maxm - mini
        if not rang: return A
    
        for a in A:
            i = int( (a - mini) * (len(A)-1)/rang )
            buckets[i].append(a)
        
        i = 0
        for b in buckets: # each bucket has the size O(1), so sorting is O(1) too
            b.sort(reverse=True) 
            while b:
                A[i] = b.pop()
                i += 1
        return A
```



k buckets, usuallt k should be n or in the same degree
Time complexity O(n + n^2/k + k) = O(n)

## Radix Sort

https://www.geeksforgeeks.org/radix-sort/

Radix sort using Count Sort on each digit repeatedly, from the smallest to biggest digits.
Time comlexity O(d*(n+b)), d is digits, b is base, so it only works better than count sort O(n + range) when the range is big and the distripution is sparse.
[radix-sort-vs-counting-sort-vs-bucket-sort-whats-the-difference](https://stackoverflow.com/questions/14368392/radix-sort-vs-counting-sort-vs-bucket-sort-whats-the-difference)

## Merge Sort

```python
'''
Merge Sort
'''
class Solution:
    # @param {int[]} A an integer array
    # @return nothing
    def sortIntegers(self, A):
        # Write your code here
        if not A:
            return []
        temp = [None]*len(A)
        self.mergeSort(A, 0, len(A)-1, temp)
        return temp

    def mergeSort(self, A, start, end, temp):
        if start >= end:
            return
        self.mergeSort(A, start, (start + end)//2, temp)
        self.mergeSort(A, (start + end)//2 + 1, end, temp)
        self.mergeLR(A, start, end, temp)

    def mergeLR(self, A, start, end, temp):
        mid_i = (start + end)//2
        i = left_i = start
        right_i = mid_i + 1
        # compare and merge left and right subarray
        while left_i <= mid_i and right_i <= end:
            if A[left_i] <= A[right_i]:
                temp[i] = A[left_i]
                left_i += 1
            else:
                temp[i] = A[right_i]
                right_i += 1
            i += 1
        # copy the rest of subarray
        while left_i <= mid_i:
            temp[i] = A[left_i]
            left_i += 1
            i += 1

        while right_i <= end:
            temp[i] = A[right_i]
            right_i += 1
            i += 1
        # copy temp to A
        for i in range(start, end+1):
            A[i] = temp[i]
```

# Linked List

## Reverse Linked List

  * remove and insert target node to bwtween dummy_head and head
  * `dummy->prev(old head)->target->...`
  * While prev.next:
    1. target = prev.next
    2. Remove target
    3. insert target
    4. DON’T move prev

See 206 Reverse Linked List

## Two Pointers

two pointers can be used in Array, Tree, multiple places.
with Slow and Fast. Find loop in LL or n-th node
See 141 Linked List Cycle, 19 Remove Nth Node From End of List

## Double Linked List

See 641 Design Circular Deque

## Homwork

```
206 Reverse Linked List
141 Linked List Cycle
19 Remove Nth Node From End of List
641 Design Circular Deque
```

## Solutions

### 206 Reverse Linked List

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        '''
        1->2->3->4->5->NULL
        n1.next == n2
        n2.next == n3
        
        d     t n
        D-2-1 3-4-5-Null
        1. head.next = None
        2. n = t.next
        3. t.next = d.next
        4. d.next = t
        '''
        if not head: return
        d = ListNode(None)
        d.next = head
        t = head.next
        head.next = None #1
        while t:
            n = t.next #2
            t.next = d.next 
            d.next = t #4
            t = n
        
        return d.next
```



### 141 Linked List Cycle

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head: return False
        fast, slow = head, head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        
        return False
```

### 19 Remove Nth Node From End of List

```python
# Definition for singly-linked list.
class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        '''
                  f
        D-1-2-3-5-None
              s
        '''
        d = ListNode(None)
        d.next = head
        fast, slow = d, d
        while fast.next:
            if n <= 0:
                slow = slow.next
            fast = fast.next
            n -= 1
        slow.next = slow.next.next
        return d.next
```



### 641 Design Circular Deque

```python
class Node:
    def __init__(self, value=None):
        self.value = value
        self.prev = None
        self.next = None
        
class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.k = k
        self.size = 0
        self.d = Node()
        self.t = Node()
        self.d.next, self.t.prev = self.t, self.d

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.isFull(): return False
        l, n, r = self.d, Node(value), self.d.next
        n.prev, n.next = l, r
        l.next = r.prev = n
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.isFull(): return False
        l, n, r = self.t.prev, Node(value), self.t
        n.prev, n.next = l, r
        l.next = r.prev = n
        self.size += 1
        return True
        

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.isEmpty(): return False
        l, r = self.d, self.d.next.next
        l.next, r.prev = r, l
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.isEmpty(): return False
        l, r = self.t.prev.prev, self.t
        l.next, r.prev = r, l
        self.size -= 1
        return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.isEmpty(): return -1
        return self.d.next.value

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.isEmpty(): return -1
        return self.t.prev.value

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return self.size == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return self.size >= self.k
```



# Tree and Graph

## Binary Searching Tree (BST)

[Binary_search_tree Wiki](https://en.wikipedia.org/wiki/Binary_search_tree)

* Insert and update

  ```c++
  void insert(Node*& root, int key, int value) {
    if (!root)
      root = new Node(key, value);
    else if (key == root->key)
      root->value = value;
    else if (key < root->key)
      insert(root->left, key, value);
    else  // key > root->key
      insert(root->right, key, value);
  }
  ```

* Delete

  * For deleting node has two children: replace it by in-order successor/predecessor, and delete successor/predecessor

  * one children: delete and move the child up

  * no children: just delete it

  ```python
  def find_min(self):   # Gets minimum node in a subtree
      current_node = self
      while current_node.left_child:
          current_node = current_node.left_child
      return current_node

  def replace_node_in_parent(self, new_value=None):
      if self.parent:
          if self == self.parent.left_child:
              self.parent.left_child = new_value
          else:
              self.parent.right_child = new_value
      if new_value:
          new_value.parent = self.parent

  def binary_tree_delete(self, key):
      if key < self.key:
          self.left_child.binary_tree_delete(key)
          return
      if key > self.key:
          self.right_child.binary_tree_delete(key)
          return
      # delete the key here
      if self.left_child and self.right_child: # if both children are present
          successor = self.right_child.find_min()
          self.key = successor.key
          successor.binary_tree_delete(successor.key)
      elif self.left_child:   # if the node has only a *left* child
          self.replace_node_in_parent(self.left_child)
      elif self.right_child:  # if the node has only a *right* child
          self.replace_node_in_parent(self.right_child)
      else:
          self.replace_node_in_parent(None) # this node has no children
  ```

* Search

  ```
  def search_recursively(key, node):
      if node is None or node.key == key:
          return node
      if key < node.key:
          return search_recursively(key, node.left)
      # key > node.key
      return search_recursively(key, node.right)
  ```

## Sort or Valid BST

Pre-order is ascending.e.g 123456 
Flipped-post-order is descending, e.g 654321.

See 98 Validate Binary Search Tree

## Union Find, Disjoint-set (optional)

[Wiki Disjoint-set_data_structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)

Useful when dynamically adding new elements

547 Friend Circles.   p.s it can be done by DFS easily. You don't have to use Union Find

```python
class Node():
    def __init__(self, id):
        self.id = id
        self.parent = self
        self.rank = 0

class Solution(object):
    def __init__(self):
        self.nodes = []
        self.count = 0

    def find(self, n):
        if n != n.parent:
            n.parent = self.find(n.parent) # compress the depth

        return n.parent

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a.id == b.id: return
        if a.rank < b.rank:
            # b's depth is bigger, so attach a to b won't increase depth
            a.parent = b
        elif a.rank > b.rank:
            b.parent = a
        else:
            a.parent = b
            b.rank += 1

        self.count -= 1


    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        for i in range(len(M)):
            self.nodes.append(Node(i))
            self.count += 1

        for i in range(len(M)):
            for j in range(len(M[0])):
                if M[i][j] == 1:
                    self.union(self.nodes[i], self.nodes[j])

        return self.count

```

## Trie (optional)

Trie is a tree to record words. Make each chr into a node, and if the chr is at the end of the word, isWord=True.
208 Implement Trie (Prefix Tree)

```python
class Node:
    def __init__(self):
        self.ch = {}
        self.isWord = False
        
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        trav = self.root
        for ch in word:
            if ch not in trav.ch:
                trav.ch[ch] = Node()
                
            trav = trav.ch[ch]
        
        trav.isWord = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        trav = self.root
        for ch in word:
            if ch not in trav.ch: 
                return False
            
            trav = trav.ch[ch]

        return trav.isWord

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        trav = self.root
        for ch in prefix:
            if ch not in trav.ch: 
                return False
            
            trav = trav.ch[ch]
        
        return True
```



## Interval Tree (optional)

[wiki](https://en.wikipedia.org/wiki/Interval_tree). Find ranges by a **range**. never heard you have to implement it during interview

## Segment Tree (optional)

[Wiki](http://en.wikipedia.org/wiki/Segment_tree). Find ranges by a **number**. never heard you have to implement it 

## SubSet vs Combination vs Permutation

```
------------------------------------------------------------------------
Subsets               | Combination           	|	Permutations
Oder NOT important    | Oder NOT important      |	Oder IS important
Any length            | length = k							| length = len(input)
          To deal repeated elements, skip the previous one
------------------------------------------------------------------------

## Combination and Subset
Given [1,2,3]
expected:
[[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]

DFS, save at pre-order, and skip the children on the left side array of the parent.
for range(i, len)
...
      None
    /  |  \
   1   2   3
  / \  |
 2  3  3
 |
 3

## Permutations
Given [1,2,3]
expected:
[[1,2,3],
 [1,3,2],
 [2,1,3],
 [2,3,1],
 [3,1,2],
 [3,2,1]]

dfs, save at post-order, skip the child visited
for range(0, len): if visited: continue
...
      None
   /   |    \
  1    2     3
 / \   /\    /\
2  3  1  3  1  2
|  |  |  |  |  |
3  2  3  1  2  1

## To not include repeated elements,
both Combination and Permutations skip the repeated childs "under the same parent"
(in the same for loop). Only enter the repeated child once, when first time visit.
...
prev = None
for range(i, len):
	if prev == A[i]: continue
	....

## Repeated Combination and Subset
Given [1,2,2]
expected:
[[],[1],[1,2],[1,2,2],[2]]

          None
     /      |    \
   1        2     skip_2
  / \       |
 2  skip_2 skip_2
 |
 2

## Repeated Permutations
Given [1,2,2]
expected: [[1,2,2],[2,1,2],[2,2,1]]

         None
    /      |   \
   1       2    skip_2
  / \      /\
 2 skip_2 1  2
 |        |  |
 2        2  1


```



## Copy Graph

Step 1. Copy node and make a old_to_new dict

Step 2. Copy neighbors' reference

## Dijkstra's Algorithm (optional)

Dijkstra finds the minimum **cost** that needs to be travelled to get to the goal vertex , whereas BFS is based on minimum "steps" only. It is greedy. Searching in graph.

```python
from collections import defaultdict
from heapq import *

def dijkstra(edges, f, t): #f=from, t=to
    g = defaultdict(list)
    for l,r,c in edges: # parent, child, cost
        g[l].append((c,r)) # build {parent:[(cost, child),..]} dict for travel
		# q is a heap of node based on (cost, parent, path)
    # mins is the minimal accumulated cost from the start f to this node
    q, seen, mins = [(0,f,())], set(), {f: 0}
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")

if __name__ == "__main__":
    edges = [
        ("A", "B", 7),
        ("A", "D", 5),
        ("B", "C", 8),
        ("B", "D", 9),
        ("B", "E", 7),
        ("C", "E", 5),
        ("D", "E", 15),
        ("D", "F", 6),
        ("E", "F", 8),
        ("E", "G", 9),
        ("F", "G", 11)
    ]

    print "=== Dijkstra ==="
    print edges
    print "A -> E:"
    print dijkstra(edges, "A", "E")
    print "F -> G:"
    print dijkstra(edges, "F", "G")

'''
===== output =====
A -> E:
(14, ('E', ('B', ('A', ()))))
F -> G:
(11, ('G', ('F', ())))
'''

```

## Topological sort, kahn's algorithm (Optional)

in-degree is the number of parents, out-degree is the number of children

1. Build indegree dict, init result=[]
2. Find entries (zero indegreee) and put them in a set (or queue, order isn't important)
3. While set not empty, set pop to result, and indegree -1 in dict, if 0 indegree , add to set.

[210 course-schedule-ii](https://leetcode.com/problems/course-schedule-ii/submissions/)

```python
from collections import defaultdict

def findOrder(numCourses, prerequisites):
    """
    :type numCourses: int
    :type prerequisites: List[List[int]]
    :rtype: List[int]
    """
    G = defaultdict(list)
    S = {i for i in range(numCourses)} # either set, queue, stack is ok
    in_degree = [0] * numCourses
    for node, pre in prerequisites:
        G[pre].append(node)
        in_degree[node] += 1
        if node in S:
            S.remove(node)

    r = []
    while S:
        node = S.pop()
        r.append(node)
        for child in G[node]:
            in_degree[child] -= 1
            if not in_degree[child]:
                S.add(child)

    return r if len(r) == numCourses else []
```

## Homework

```
109. Convert Sorted List to Binary Search Tree
98 Validate Binary Search Tree
236 Lowest Common Ancestor of a Binary Tree
102 Binary Tree Level Order Traversal

78 Subset
90 Subsets II
77 Combinations
46 Permutations
47 Permutations II

133. Clone Graph
127. Word Ladder
```

## Solutions 

### 109. Convert Sorted List to Binary Search Tree

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        '''
        [-10,-3,-1,0,5,9,10]
                   ^
              ^        ^ 
           ^     ^   ^    ^
        '''
        A, trav = [], head
        while trav:
            A.append(trav)
            trav = trav.next
        return self.dfs(A)
        
    def dfs(self, A):
        if not A:
            return None
        mid = (len(A)-1)//2
        root = TreeNode(A[mid].val)
        root.left = self.dfs(A[0: mid])
        root.right = self.dfs(A[mid+1:])
        return root
```

### 98 Validate Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.prev = None
        
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        left = self.isValidBST(root.left)
        if self.prev != None and root.val <= self.prev.val:
            return False
        
        self.prev = root
        right = self.isValidBST(root.right)
        return left and right
```

### 236 Lowest Common Ancestor of a Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root: 
            return None
        if root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        #post order
        if left and right:
            return root
        
        return left or right
```

### 102 Binary Tree Level Order Traversal

```python
from queue import Queue
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = Queue()
        q.put(root)
        result = []
        while not q.empty():
            width = q.qsize()
            level = []
            for _ in range(width):
                node = q.get()
                level.append(node.val)
                if node.left: q.put(node.left)
                if node.right: q.put(node.right)
                    
            result.append(level)
            
        return result
```

### 78 Subset

```python
class Solution:
    def subsets(self, A: List[int]) -> List[List[int]]:
        result = []
        self.dfs(A, 0, [], result)
        return result
    '''
         None
       1   2  3
      2 3  3
     3
    '''
    def dfs(self, A, i, path, result):
        result.append(path[:])
        for i in range(i, len(A)):
            path.append(A[i])
            self.dfs(A, i+1, path, result)
            path.pop()
```

### 90 Subsets II

```python
class Solution:
    def subsetsWithDup(self, A: List[int]) -> List[List[int]]:
        result = []
        A.sort() ####
        #[1,2,2]
        self.dfs(A, 0, [], result)
        return result
    '''
         None
       1   2 
      2    2
     2
    '''
    def dfs(self, A, i, path, result):
        result.append(path[:])
        prev = None
        for i in range(i, len(A)):
            if A[i] == prev: continue
            path.append(A[i])
            self.dfs(A, i+1, path, result)
            path.pop()
            prev = A[i]
```

### 77 Combinations

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        A = [i for i in range(1, n+1)]
        result = []
        self.dfs(A, 0, k, [], result)
        return result
    
    def dfs(self, A, i, k, path, result):
        if len(path) == k: result.append(path[:])
        for i in range(i, len(A)):
            path.append(A[i])
            self.dfs(A, i+1, k, path, result)
            path.pop()
```

### 46 Permutations

```python
class Solution:
    def permute(self, A: List[int]) -> List[List[int]]:
        visited = set()
        result = []
        self.dfs(A, [], visited, result)
        return result
    
    def dfs(self, A, path, visited, result):
        if len(path) == len(A):
            result.append(path[:])
            return
        
        for i in range(0, len(A)):
            if i in visited: continue
            visited.add(i)
            path.append(A[i])
            self.dfs(A, path, visited, result)
            visited.remove(i)
            path.pop()
```

### 47 Permutations II

```python
class Solution:
    def permuteUnique(self, A: List[int]) -> List[List[int]]:
        visited = set()
        result = []
        A.sort()
        self.dfs(A, [], visited, result)
        return result
    
    def dfs(self, A, path, visited, result):
        if len(path) == len(A):
            result.append(path[:])
            return
        
        prev = None
        for i in range(0, len(A)):
            if i in visited or A[i] == prev: continue
            visited.add(i)
            path.append(A[i])
            self.dfs(A, path, visited, result)
            visited.remove(i)
            path.pop()
            prev = A[i]
```

### 133 Clone Graph

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        
        old2new = {}
        #1 copy node
        self.cloneNodeDFS(node, old2new)
        
        #2 copy path
        for old, new in old2new.items():
            for n in old.neighbors:
                new.neighbors.append(old2new[n])
        return old2new[node]
    
    def cloneNodeDFS(self, node, old2new):
        if not node or node in old2new:
            return 
        
        newNode = Node(node.val)
        old2new[node] = newNode
        for n in node.neighbors:
            self.cloneNodeDFS(n, old2new)
```

### 127 Word Ladder

```python
from queue import Queue

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        #BFS
        wordList = set(wordList)
        q = Queue()
        q.put(beginWord)
        visited = {beginWord}
        depth = 0
        while not q.empty():
            width = q.qsize()
            depth += 1
            for _ in range(width):
                w = q.get()
                if endWord == w:
                    return depth
                for child in self.getChildren(w, wordList):
                    if child not in visited:
                        q.put(child)
                        visited.add(child)
        return 0
    # Method 1. mutate begin word and check if in wordList set
    # T= O(26*k) better
    # len(word)=k, len(wordList)=n
    #word = 'hit' 
    #ait, bit, cit.... in wordList
    #hat, hbt, hct.....in wordList

    # Method 2. compare each word in wordList to begin word
    #T= O(n*k)
    #for w in wordList:
    #    word[:i] == w[:i] and word[i+1:] == w[i+1:]
    
    def getChildren(self, word, wordList):
        children = []
        for i in range(len(word)):
            for ch in range(ord('a'), ord('z')+1):
                child = word[:i] + chr(ch) + word[i+1:]
                if child != word and child in wordList:
                    children.append(child)
        return children
```



# DP and Greedy

## Dynamic Programming (DP)

**Steps to solve a DP problem**

1) Identify if it is a DP problem? Can the problem be divided into subproblems? Are the answers overlapping? Ans_i = Ans_i-1 + Ans_i-2 ?

2) Decide the table. Does it contain the essential answer for the next answer? Simp enough?

3) Formulate the relationship of the table. dp[i] = dp[i-1] + dp[i-2] 
 Try the case that is as simple as possible first.

4) Do it iteratively (top down) or recursively (bottom up)? 

## Buy and Sell Stock

ref. https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108870/most-consistent-ways-of-dealing-with-the-series-of-stock-problems

  ```
	T[i][k][n]
  T: is max total profit
  k: with at most k transactions  #is the remained transactions (number of transaction)
    Notice k-1 at buy, not -1 at sell
  i: is the i-th day
  n: is number of stocks in hand
 
  boundary cases:
  Assume the stock price at -1 day is Infinity expensive
  T[-1][k][0] = 0, T[-1][k][1] = float('-int')
  T[i][0][0] = 0, T[i][0][1] = float('-int')

  Recurrence relations:
  No stock on hand, Waiting or sold, max profit while not holding, is either 
  the "same" or the "max profit while hoding + price"
  
  T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
  
  One stock on hand, Holding or bought, max profit while hoding, is either the 
  "k-1 max profit while not holding - price" or "holding"
  Notice k+1 when buy
  
  T[i][k][1] = max(T[i-1][k][1], T[i-1][k-1][0] - prices[i])

  Sell, T[i][k][0], is not related to k remained transactions at all, since we can alway sell it or keep it regardless remained transactions

  Buy, T[i][k][1], is related to k

Finally you can simplify the T to two var or array
sold = T[i][k][0], since it's always from i-1, the array of old i isn't needed
hold = T[i][k][1]
For example...
  ```

```python
#714. Best Time to Buy and Sell Stock with Transaction Fee
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        sold = 0
        hold = float('-inf')
        for p in prices:
            sold = max(sold, hold + p)
            #when hold+p>sold
            #hold>sold-p
            #hold>sold-p-fee
            #so don't need to worry hold be affected by updated sold 
            hold = max(hold, sold - p - fee)
        return sold

#188. Best Time to Buy and Sell Stock IV
class Solution:
    def maxProfit(self, K: int, prices: List[int]) -> int:
        if not prices or K == 0: return 0
        sold = 0
        hold = float('-inf')
        
        if K >= len(prices)//2:
            for p in prices:
                sold = max(sold, hold+p)
                hold = max(hold, sold-p)
            return sold
        
        sold = [0] * (K+1)
        hold = [float('-inf')] * (K+1)
        for p in prices:
            for k in range(1, K+1): # Notice: range(1, K+1), if 0 will be wrong
                sold[k] = max(sold[k], hold[k] + p)
                hold[k] = max(hold[k], sold[k-1] - p)
        return  sold[K]
```



## Knapsack problem (optional)

```python
Knapsack problem
'''
bloomberg interview - Allocate NY SF Candidates

There are two offices. One is in NYC, the other is in SF. There are multiple candidates that needs to be allocated to these 2 places to get interview. For each candidate, the cost to NYC or SF is represented as a tuple (e.g. (100, 200), which means allocating this candidate to NYC costs $100; allocating him to SF costs $200). The limitation is that: the allocation of candates to these 2 places (NYC and SF) should be as even as possible. For instance, there are 4 candidates, 2 of them have to be in NYC. If there are 5 candidates, 2 in NYC and 3 in SF or 3 in NYC and 2 in SF.

    Example 1: [(200, 300), (500, 200), (100, 1000), (700, 300)], representing the cost of for allocating 4 candates. The answer should be 800 (200 + 200 + 100 + 300 = 800)

Follow up, if some imployees has preference, and preference is more important than cost?
'''
import unittest
import heapq

# Use Heap for priority, abs diff
def getMinCost(A):
    '''
    @A list [(NY_cost, SF_cost), ...]
    @return minCost
    '''
    heap = [(-abs(NY-SF), NY, SF) for NY, SF in A]
    heapq.heapify(heap)
    half = (len(A)+1) // 2 # +1 for odd case
    minCost = NY_count = SF_count = 0
    while heap:
        _, NY, SF = heapq.heappop(heap)

        if NY <= SF:
            NY_count += 1
            minCost += NY

        else:
            SF_count += 1
            minCost += SF

        if SF_count == half:
            minCost += sum(NY for _, NY, _ in heap)
            break

        if NY_count == half:
            minCost += sum(SF for _, _, SF in heap)
            break

    return minCost

# Use Array + Sort by priority, abs diff
def getMinCost2(A):
    A.sort(key=lambda (NY, SF): abs(NY - SF))
    half = (len(A) + 1) // 2
    minCost = NY_count = SF_count = 0
    while A:
        NY, SF = A.pop()
        if NY < SF:
            NY_count += 1
            minCost += NY

        else:
            SF_count += 1
            minCost += SF

        if SF_count == half:
            minCost += sum(NY for NY, _ in A)
            break

        if NY_count == half:
            minCost += sum(SF for _, SF in A)

    return minCost


class Test(unittest.TestCase):
    def test1(self):
        candidates = [(200, 300), (500, 200), (100, 1000), (700, 300)]
        expectedMinCost = 800
        self.assertEqual(getMinCost(candidates), expectedMinCost)
        self.assertEqual(getMinCost2(candidates), expectedMinCost)

if __name__ == '__main__':
    unittest.main()
```



## Homework

```
409 Longest Palindrome
55 Jump Game
45 Jump Game II
70 Climbing Stairs

121	Best Time to Buy and Sell Stock
122	Best Time to Buy and Sell Stock II
309	Best Time to Buy and Sell Stock with Cooldown    
714	Best Time to Buy and Sell Stock with Transaction Fee

```

## Solutions

### 409 Longest Palindrome

```python
# Greedy
class Solution:
    def longestPalindrome(self, s: str) -> int:
        isOdd = set()
        for c in s:
            if c in isOdd:
                isOdd.remove(c)
            else:
                isOdd.add(c)
        
        mid = 1 if len(isOdd) > 0 else 0    
        return len(s) - len(isOdd) + mid
    
```


### 45 Jump Game II

```python
class Solution:
    # greedy
    def jump(self, A: List[int]) -> int:
        if len(A) <= 1: return 0
        l, r = 0, A[0]
        step = 1
        while r < len(A) - 1:
            step += 1
            nxt = max(i + A[i] for i in range(l, r + 1)) #local max
            l, r = r+1, nxt
        return step
    # DP    
    def jumpDP(self, A: List[int]) -> int:
        dp = [float('inf')] * len(A)
        dp[0]=0
        visited = 0
        for i, maxJump in enumerate(A):
            if i + maxJump <= visited:
                continue 
                
            for j in range(1, maxJump+1):
                if i+j < len(dp):
                    # rule1
                    dp[i+j] = min(dp[i+j], dp[i]+1)
                    visited = i + j
                    
        return dp[-1]
        
        '''
        dp[i] = minimum number of jumps to reach i 
        
        #rule 1
        dp[i+j] = min(dp[i+j], dp[i]+1)
        
        j = 1~maxJump
        i = starting point
        
           i
        A=[2,3,1,1,4]
        dp=[0, inf, inf, inf, inf]
        
        i=0
        j=2
        dp=[0, min(inf,0+1) , min(inf, 0+1), inf, inf]
        dp=[0,1,1, inf, inf]
        
        i=1
        j=3
        dp=[0,1,min(1, 1+1), min(inf, 1+1), min(inf, 1+1)]
        dp=[0,1,1,2,2]
        
        i=2
        j=1
        dp=[0,1,1,2,2]
        skip
        #rule2
        # i1+A[i1] >= i2 + A[i2]
        
        i=3
        j=1
        dp=[0,1,1,2,min(2, 2+1)]
        dp=[0,1,1,2,2]
        
        return 2
        '''
```



### 70 Climbing Stairs

```python
class Solution:
    # Greedy 
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        
        a, b = 1, 2 
        for i in range(2, n):
            a, b = b, a + b
        return b
    
    # iterative DP
    def climbStairsIterative(self, n: int) -> int:
        if n == 1: return 1
        dp = [None] * n
        dp[0], dp[1] = 1, 2 
        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
      
    # Recursive DP
    def climbStairsRecursive(self, n: int) -> int:
        '''
        dp[i] = ways?
        dp[i] = dp[i-1] + dp[i-2]
        iterative?
        recursive?
        '''
        #recursive top bottom
        if n == 1: return 1
        dp = [None] * n
        dp[0], dp[1] = 1, 2 
        return self.getWays(n-1, dp)
    
    def getWays(self, n, dp):
        if not dp[n]:
            dp[n] = self.getWays(n-1, dp) + self.getWays(n-2, dp)
        return dp[n]
```



# OO Design

[Good youtube tutorial for OO Design workflow](https://www.youtube.com/watch?v=fJW65Wo7IHI&list=PLGLfVvz_LVvS5P7khyR4xDp7T9lCk9PgE)

## Tiny URL

    How to index the url?
    Method 1. By order
    1. just put urls in an array, map the index to the 62 digit system, `0,1...8,9,a,b...y,z,A,B...Y,Z` as the code, so the url is shorter. 
     Or or just use 10 digit system 0-9 is ok too.
    Pro: No collision. Unlimited size to extend.
    
    Method 2. Use random code within a fixed range.
    2. use 6 `random.choice(string.digits + string.ascii_letters)`` as code, the chance of collision is existed/62^6. If collision, just do again.
    Pro: len is consistant. Hacker can't guess the index.

```python
import string
import random
from collections import OrderedDict

#Method 1 using 62 digit
class Codec:
    def __init__(self):
        self.urls = []
        self.base = OrderedDict((k, i) for i, k in enumerate(string.digits + string.ascii_letters))
        
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """        
        self.urls.append(longUrl)
        n = len(self.urls)
        code = []
        while n > 0:
            code.append(self.base.keys()[n % len(self.base)])
            n = n // len(self.base)
            
        return 'http://tinyurl.com/' + ''.join(code)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        code = shortUrl.split('/')[-1]
        n = 0
        for ch in code:
            n = n * len(self.base) + self.base[ch]
        
        return self.urls[n-1]
        
#Method 1 using 10 digit
class Codec1:
    def __init__(self):
        self.urls = []
        
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """        
        self.urls.append(longUrl)
        return 'http://tinyurl.com/' + str(len(self.urls)-1)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        return self.urls[int(shortUrl.split('/')[-1])]
        
#Method 2 using random
class Codec2:
    def __init__(self):
        self.code2url = {}
        self.url2code = {}
        
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        letter_set = string.ascii_letters + string.digits
        
        while longUrl not in self.url2code:
            code = ''.join([random.choice(letter_set) for _ in range(6)])
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
        
        return 'http://tinyurl.com/' + self.url2code[longUrl]

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        return self.code2url[shortUrl[-6:]]
```



# System Design

A very good reading material for beginner

https://www.jyt0532.com/2017/03/27/system-design/

## rate limit

[system-design-rate-limiter-and-data-modelling](https://medium.com/@saisandeepmopuri/system-design-rate-limiter-and-data-modelling-9304b0d18250)

- **Leaky Bucket (Queue)**

  ```python
  if queue.qsize() <= cap:
      queue.put(request)
  ```

- **Fixed window counters**

  ```python
  {
   "1:00AM-1:09AM": 7,
   "1:10AM-1:19AM": 8
  }
  ```

  Con: all requests can happend in 1:09-1:10 AM, with in 2 min

- **Sliding window log**

  For `every user`, a queue of timestamps representing the times at which all the historical calls have occurred within the timespan of recent most window is maintained.

  **Pros**

  - Works perfectly

  **Cons**

  - High memory footprint. All the request timestamps needs to be maintained for a window time, thus requires lots of memory to handle multiple users or large window times
  - High time complexity for removing the older timestamps

- **Sliding window counter**
  It is Sliding window log + Fixed window counter (with sub time bucket)

  For example, limit **10 req/30min**
  ..., (0:50-1:00), **[(1:00-1:10), (1:10-1:20), (1:20-1:30)]**, ...
  When we get request during (1:20-1:30), Check the sub-buckets within the previous 30 mins, and old buckets can be deleted.

# Other

## What happens when you type a URL in the browser and press enter?

https://medium.com/@maneesha.wijesinghe1/what-happens-when-you-type-an-url-in-the-browser-and-press-enter-bb0aa2449c1a

### 1. Four caches
- browser cache, 
- OS cache, 
- router cache, 
- ISP (internet service provider)'s DNS server

### 2. Query DNS
Send a requests to DNS servers to query the IP. It first hist the root domain DNS, it redirect to the top domain, second, and then even third domain DNS servers 

Root domain: .
Top domain: .com, .io, .edu...
Second domain: google.com, coincastle.io, bostonuniversity.edu ...
Third domain: map.google.com ...

### 3. TCP and then http:

- Cliend send sync request, Server return sync + acknowledge, client return  acknowledge 
- After TCP connect setup.. Client send HTTP GET or POST
- Server return response, it contain status code, body etc

### 4. render the webpage

Browser go through the html file and may request more resource (send more http requests). 



## process vs thread

One process can have many thread inside

Thread share memory. 
Process don't share memory by default, although you can force them to share memory segments.

## Generate Random

```python
import time
import random

class MyRandom:
    ''' m is divisible by 4 '''
    m = 2**31
    ''' a − 1 should be divisible by all prime factors of m'''
    a = 1103515245
    ''' c and m should be coprime '''
    c = 12345
    def __init__(self):
        self.A = []
        self.removed = set()
        self.seed =  int(time.time()*1000000)%MyRandom.m
    '''
		# Method1 BitShift (Easier)
    Easiest impliment. ^ is XOR, << is binary sift, 21,35,4, magic number
    https://www.javamex.com/tutorials/random_numbers/xorshift.shtml#.XodYntNKhhH
    '''
    def getRandom(self):
        self.seed ^= (self.seed << 21);
        self.seed ^= (self.seed >> 35);
        self.seed ^= (self.seed << 4);
        self.seed = self.seed % MyRandom.m
        return self.seed;
    '''
		# Method2 Linear congruential generator
		https://stackoverflow.com/questions/3062746/special-simple-random-number-generator
    '''
    def _getRandom(self):
        #print(MyRandom.a * self.seed + MyRandom.c)
        for _ in range(100):
            self.seed = (MyRandom.a * self.seed + MyRandom.c) % MyRandom.m;
        return self.seed;
```



## Multithread and Lock

https://www.bogotobogo.com/python/Multithread/python_multithreading_Synchronization_Lock_Objects_Acquire_Release.php

```python
import threading
import time
import logging
import random

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
                    
class Counter(object):
    def __init__(self, start = 0):
        self.lock = threading.Lock()
        self.value = start
    def increment(self):
        logging.debug('Waiting for a lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired a lock')
            self.value = self.value + 1
        finally:
            logging.debug('Released a lock')
            self.lock.release()

def worker(c):
    for i in range(2):
        r = random.random()
        logging.debug('Sleeping %0.02f', r)
        time.sleep(r)
        c.increment()
    logging.debug('Done')

if __name__ == '__main__':
    counter = Counter()
    for i in range(2):
        t = threading.Thread(target=worker, args=(counter,))
        t.start()

    logging.debug('Waiting for worker threads')
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()
    logging.debug('Counter: %d', counter.value)
```

## Manacher’s algorithm (optional)

This is too hard and no good interivewer will ask this, but if you're interested, you can read it. 

[Best explaination](https://articles.leetcode.com/longest-palindromic-substring-part-ii/)

[longest-palindromic-substring](https://leetcode.com/problems/longest-palindromic-substring/)

```python
#Manacher algorithm
#https://articles.leetcode.com/longest-palindromic-substring-part-ii/
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # Build T, ABC -> @#A#B#C#*
        # Edge must be diff chr, to stop expending P at edge
        # Add # to make couting easier
        T = '#'.join('@'+s+'*')
        n = len(T)
        P = [0] * n
        C = R = 0 # C = Center, R = max available right idex of P
        for i in range(1, n-1):
            i_ = 2 * C - i # the mirror of i

            # if R-i < P[i_], we can't access P beyond R
            P[i] = min(R-i, P[i_]) if R > i else 0

            # expending P
            while T[i + P[i] + 1] == T[i - P[i] - 1]:
                P[i] += 1

            # update C and R
            if (i + P[i] > R):
                R = i + P[i]
                C = i

        p, c = max((p, c) for c, p in enumerate(P))
        return s[(c-p)//2 : (c+p)//2]
```



# Speech for resume

Write down your self-intro and speech about your resume here

# Behavioral Questions

Voice record your answer and add notes here

**General BQ:**
What is your most Challenging project? How did you deal with it?
Describe a situation where you exceeded expectations and did more than required?
Have you ever had a conflict with the team? How was it resolved?
Tell me about a time when you demonstrated leadership.

**Negative BQ:** 
What are your weaknesses?
What do you dislike the most about your last job?
Have you ever had a conflict with the team? How was it resolved?
Tell me about a conflict you had with your teammate? Why was there a conflict? What did you do?
What would you do if you disagree with your teammate? Why was there a conflict? What did you do?
Tell me about a mistake you made?
What are the areas where you need to improve your skills?
Are there areas where you need to develop your skills further?
Tell me about a time when you don’t have enough information or resources to finish your job?

**Motivation BQ:**
Why our company? Why Facebook/Amazon/Apple/Google?
Where do you see yourself in five years?

# Resources

**Coding Questions and reading material**

- Dev’s course note

- Cracking the Coding Interview: Good book for beginner, but it's too easy
  [https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/0984782850](https://www.amazon.com/Cracking-Coding-Interview-Programming-Questions/dp/0984782850)

  [search in github](https://www.google.com/search?rlz=1C5CHFA_enUS886US891&sxsrf=ALeKk03D1JZprUmVGhPWTmjFTW101xqtow%3A1587329704041&ei=qLqcXs-EAs2LytMP6a6dqAY&q=cracking+the+coding+interview+github&oq=Cracking+the+Coding+Interview+git&gs_lcp=CgZwc3ktYWIQAxgAMgIIADIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLATIFCAAQywEyBQgAEMsBMgUIABDLAToECAAQRzoECAAQQzoECAAQHjoFCCEQoAFQsQlY1h9g-iZoA3ACeACAAaUIiAHaE5IBCzQuMC4xLjYtMS4xmAEAoAEBqgEHZ3dzLXdpeg&sclient=psy-ab)

- Leetcode
  https://leetcode.com/

- GeeksForGeeks
  https://www.geeksforgeeks.org/

- HackRank (Leetcode usually has better questions and discussion)
  https://www.hackerrank.com/

- 一畝三分地 面經版
  https://www.1point3acres.com/bbs/forum-145-1.html

- 1o24BBS 面經版
  https://1o24bbs.com/c/job/Interview/86
- Wiki 
  https://www.wikipedia.org/

**Internal Referal** 

- 一畝三分地  美国职位内推版: [https://www.1point3acres.com/bbs/tag/%E7%BE%8E%E5%9B%BD%E8%81%8C%E4%BD%8D%E5%86%85%E6%8E%A8](https://www.1point3acres.com/bbs/tag/美国职位内推)

- 1o24 BBS 内推版: https://1o24bbs.com/c/job/referral

- 批踢踢 You can find some referral posts in PTT too, for example.. 
   [北美] Amazon各職位內推 讓台灣人把亞麻填滿滿: https://www.pttweb.cc/bbs/Oversea_Job/M.1528069498.A.520

- Linked-In
   First, search the company and the employees in your network will show up.
  You can even just ask someone you don’t know politely, some of them actually happy to help.
   if you don’t have any networks yet, find some groups related to you. For example…
  Si29’s 科技公司內推+刷題 美國 日本 台灣 工程師討論群: https://www.linkedin.com/groups/13847069/ 
  台灣海外工程師幫: https://www.linkedin.com/groups/13579062/**

**Looking for partners** 

- FB groups: 
  FB is hard to search people by company. However, it’s a good place to find partners or ask questions. For example, Si29’s 科技公司內推+刷題 美國 日本 台灣 工程師討論群”: https://www.facebook.com/groups/601310283993126






