- [python 科学数据库](#python-科学数据库)
  - [Numpy—— 科学计算库](#numpy-科学计算库)
    - [定义](#定义)
    - [计算](#计算)
      - [加](#加)
      - [乘](#乘)
      - [统计量](#统计量)
      - [逻辑运算](#逻辑运算)
      - [array[]](#array)
      - [argmin/argmax](#argminargmax)
    - [属性](#属性)
      - [类型](#类型)
        - [type](#type)
        - [dtype](#dtype)
        - [nbytes](#nbytes)
      - [大小](#大小)
        - [itemsize](#itemsize)
        - [size](#size)
      - [形状](#形状)
        - [shape](#shape)
        - [reshape/newaxis/squeeze](#reshapenewaxissqueeze)
        - [transpose/.T](#transposet)
      - [连接](#连接)
        - [concatenate](#concatenate)
        - [vstack/hstack](#vstackhstack)
      - [维度](#维度)
        - [ndim](#ndim)
        - [flatten/ravel](#flattenravel)
      - [最值](#最值)
      - [压缩](#压缩)
      - [取整](#取整)
    - [多维](#多维)
    - [生成](#生成)
      - [np.arrange()](#nparrange)
      - [linspace](#linspace)
      - [logspace](#logspace)
      - [meshgrid](#meshgrid)
      - [r-/ c-](#r--c-)
      - [zeros/ones](#zerosones)
      - [fill](#fill)
      - [identity](#identity)
    - [随机](#随机)
      - [rand](#rand)
      - [randint](#randint)
      - [normal](#normal)
      - [shuffle](#shuffle)
      - [seed](#seed)
    - [排序](#排序)
      - [sort](#sort)
      - [argsort](#argsort)
      - [searchsorted](#searchsorted)
      - [lexsort](#lexsort)
    - [输出](#输出)
      - [set_printoptions](#set_printoptions)
    - [读写](#读写)
      - [writefile](#writefile)
      - [readlines](#readlines)
      - [loadtxt](#loadtxt)
        - [delimiter](#delimiter)
        - [skiprows](#skiprows)
        - [usecols](#usecols)
      - [savetxt](#savetxt)
      - [save/load](#saveload)
## Numpy—— 科学计算库

安装numpy库 :`pip install numpy`

~~~python
import numpy as np   //引入库
~~~

### 定义

**np.array(x)**  x为list类型。

类型：<class 'numpy.ndarray'>

~~~python
array = np.array([1,2,3,4,5])
print (type(array))
~~~

### 计算

#### 加

1. 数组与单个数值相加

   ~~~python
   array2 = array + 1 # 数组中的每一个值与该数字相加
   # ararry2的值： array([2, 3, 4, 5, 6])
   ~~~

2. 数组与数组相加

   ~~~python
   array2 +array # 数组中对应的值相加
   # 结果：array([ 3,  5,  7,  9, 11])
   ~~~

3. 元素相加

   ~~~python
   array = np.array([[1,2,3],[4,5,6]])
   np.sum(array) #21
   array.sum # 21
   np.sum(array,axis=0) #array([5, 7, 9])
   array.sum(axis=0) #array([5, 7, 9])
   np.sum(array,axis=1) # array([ 6, 15])
   array.sum(axis=1) # array([ 6, 15])
   np.sum(array,axis=-1) # array([ 6, 15])
   ~~~

   

#### 乘

1. 数组与数组相乘

   ~~~python
   array2 * array # 数组中对应的值相乘
   # 结果：array([ 2,  6, 12, 20, 30])
   
   x = np.array([1,1,1])
   y = np.array([[1,2,3],[4,5,6]])
   print (x * y)
   '''
   [[1 2 3]
    [4 5 6]]
   '''
   ~~~

   ~~~python
   x = np.array([5,5])
   y = np.array([2,2])
   np.multiply(x,y) # array([10, 10])
   np.dot(x,y) # 20
   
   x=np.array([[5],
          [5]])
   y=np.array([2,2])
   np.dot(x,y)
   '''
   array([[10, 10],
          [10, 10]])
   '''
   np.dot(y,x) # array([[20]])
   ~~~

   

   

2. 数组中的元素相乘

   ~~~python
   array = np.array([[1,2,3],[4,5,6]])
   array.prod() # 720
   array.prod(axis = 0) # array([ 4, 10, 18])
   array.prod(axis = 1) # array([  6, 120])
   ~~~

#### 统计量

~~~python
array = np.array([[1,2,3],[4,5,6]])
# 均值 
array.mean() # 3.5
array.mean(axis = 0) # array([ 2.5,  3.5,  4.5])
# 标准差
array.std() # 1.707825127659933
array.std(axis = 1) # array([ 0.81649658,  0.81649658])
# 方差
array.var() # 2.9166666666666665

~~~

#### 逻辑运算

~~~python
y = np.array([1,1,1,4])
x = np.array([1,1,1,2])
x == y
# array([ True,  True,  True, False], dtype=bool)
np.logical_and(x,y)
# array([ True,  True,  True,  True], dtype=bool)
np.logical_or(x,y)
#array([ True,  True,  True,  True], dtype=bool)
np.logical_not(x,y)
# array([0, 0, 0, 0])
~~~



###　索引

#### array[]

~~~python
# 索引与切片:跟Python都是一样
array[0] # 1
array[1:3] # array([2, 3])
array[-2:] # array([4, 5])
~~~

#### argmin/argmax

~~~python
# 寻找索引位置
array = np.array([[1,2,3],[4,5,6]])
array.argmin() # 0
array.argmin(axis = 0) # array([0, 0, 0], dtype=int64)
array.argmin(axis=1) # array([0, 0], dtype=int64)
array.argmax() # 5
~~~



### 属性

#### 类型

##### type

~~~python
type(array) # numpy.ndarray
~~~

##### dtype

~~~python
array.dtype # dtype('int32')
~~~

dtype还可以取int64,float32,bool,object ...

对于ndarray结构来说，里面所有的元素必须是同一类型的 如果不是的话，会自动的<u>向下进行转换</u>

~~~python
array = np.array([1,10,3.5,'str'],dtype = np.object)
# array([1, 10, 3.5, 'str'], dtype=object)
array * 2
# array([2, 20, 7.0, 'strstr'], dtype=object)
~~~

##### nbytes

~~~python
array.nbytes  # 20
~~~

#### 大小

##### itemsize

~~~python
array.itemsize # 4
~~~

##### size

~~~python
array.size  # 5
np.size(array) # 5
~~~

#### 形状

##### shape

~~~python
array.shape # array([1, 2, 3, 4, 5])
# (5,)
np.shape(array)
# (5,)
array3=np.array([[1,2,3],[4,5,6]])
array3.shape
# (2, 3)

# 改变shape
array = np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
array.shape # (10,)
array.shape = 2,5
'''
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
'''

#转置
array.shape = 2,5
'''
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
'''
~~~

注意：列表（list) 没有shape

#####  reshape/newaxis/squeeze

~~~python
array.reshape(1,10) # array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# 大小不能改变，若array .shape = 3,4则报错

# 增加新的轴
array = array[np.newaxis,:]
array.shape # (1, 10)

array = np.arange(10)
array = array[:,np.newaxis]
array.shape # (10, 1)
array = array[:,np.newaxis,np.newaxis]
array.shape # (10, 1, 1, 1)
array = array.squeeze()
array.shape #(10,)


~~~

##### transpose/.T

~~~python
'''
'''
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
'''
'''
array.transpose()
'''
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
'''
array.T
'''
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
'''
~~~



#### 连接

##### concatenate

~~~python
# np.concatenate
a = np.array([[123,456,678],[3214,456,134]])
b = np.array([[1235,3124,432],[43,13,134]])
c = np.concatenate((a,b)) # 竖着连接
'''
array([[ 123,  456,  678],
       [3214,  456,  134],
       [1235, 3124,  432],
       [  43,   13,  134]])
'''
c = np.concatenate((a,b),axis = 1) #横着连接
'''
array([[ 123,  456,  678, 1235, 3124,  432],
       [3214,  456,  134,   43,   13,  134]])
'''
~~~

##### vstack/hstack

~~~python
np.vstack((a,b))
'''
array([[ 123,  456,  678],
       [3214,  456,  134],
       [1235, 3124,  432],
       [  43,   13,  134]])
'''

np.hstack((a,b))
'''
array([[ 123,  456,  678, 1235, 3124,  432],
       [3214,  456,  134,   43,   13,  134]])
'''
~~~



#### 维度

##### ndim

~~~python
'''
a=array([1,2,3])
'''
array.ndim # 1
~~~

##### flatten/ravel

~~~python
'''
a= array([[ 123,  456,  678],
       [3214,  456,  134]])
'''
a.flatten()# array([ 123,  456,  678, 3214,  456,  134])
a.ravel() # array([ 123,  456,  678, 3214,  456,  134])
~~~

#### 最值

~~~python
# array = np.array([[1,2,3],[4,5,6]])
array.min() # 1
array.min(axis = 0) # array([1, 2, 3])
array.min(axis = 1) # array([1, 4])
array.max() # 6
~~~

#### 压缩

**clip**  压缩至一定范围

~~~python
array.clip(2,4)
'''
array([[2, 2, 3],
       [4, 4, 4]])
'''
~~~

#### 取整

**round** 四舍五入

~~~python
array = np.array([1.2,3.56,6.41])
#  array([ 1.,  4.,  6.])
array.round(decimals=1)
# array([ 1.2,  3.6,  6.4])
~~~



### 多维

~~~python
array = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
array.shape # (3,3)
array.size # 9
array.ndim # 2
array[1,1] # 5
array[1,1] = 10  //改
'''
array([[ 1,  2,  3],
       [ 4, 10,  6],
       [ 7,  8,  9]])
'''
array[1] # array([ 4, 10,  6])
array[:,1] # array([ 2, 10,  8])
array[0,0:2] #array([1, 2])
~~~

### 生成

#### np.arrange()

~~~python
np.arange(10) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
array = np.arange(0,100,10) # array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
~~~

~~~python
mask = np.array([0,0,0,1,1,1,0,0,1,1],dtype=bool) 
# array([False, False, False,  True,  True,  True, False, False,  True, True])
array[mask] # array([30, 40, 50, 80, 90])
np.where(tang_array > 30) #(array([3, 4], dtype=int64),)

~~~

#### linspace

~~~python
array = np.linspace(0,10,10)
'''
array([  0.        ,   1.11111111,   2.22222222,   3.33333333,
         4.44444444,   5.55555556,   6.66666667,   7.77777778,
         8.88888889,  10.        ])
'''
~~~

#### logspace

~~~python
# 默认以10为底
np.logspace(0,1,5) # array([  1.        ,   1.77827941,   3.16227766,   5.62341325,  10.        ])
~~~

#### meshgrid

~~~python
x = np.linspace(-10,10,5)
y = np.linspace(-10,10,5)
x, y= np.meshgrid(x,y)
'''
x=y=array([[-10.,  -5.,   0.,   5.,  10.],
       [-10.,  -5.,   0.,   5.,  10.],
       [-10.,  -5.,   0.,   5.,  10.],
       [-10.,  -5.,   0.,   5.,  10.],
       [-10.,  -5.,   0.,   5.,  10.]])
'''
~~~

#### r-/ c-

~~~python
np.r_[0:10:1] # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.c_[0:10:1] 
'''
array([[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9]])
'''
~~~



#### zeros/ones

~~~python
np.zeros(3) #array([ 0.,  0.,  0.])
np.ones((3,3)) 
'''
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
'''

array = np.array([1,2,3,4])
np.zeros_like(array)# array([0, 0, 0, 0])
np.ones_like(array) # array([1, 1, 1, 1])
~~~

#### fill

~~~python
a = np.empty(6)
a.fill(1) # array([ 1.,  1.,  1.,  1.,  1.,  1.])
~~~

#### identity





### 随机 

#### rand

~~~python
np.random.rand() # 0.5595234784766201
random_array = np.random.rand(10) #默认值都是从0到1
# array([0.91640149, 0.61464164, 0.81346537, 0.57917897, 0.37086614, 0.54156666, 0.47357823, 0.79819449, 0.76635931, 0.13830215])
np.random.rand(3,2)
'''
array([[ 0.87876027,  0.98090867],
       [ 0.07482644,  0.08780685],
       [ 0.6974858 ,  0.35695858]])
'''
np.random.random_sample() # 0.8279581297618884
~~~

#### randint

~~~python
#返回的是随机的整数，左闭右开
np.random.randint(10,size = (5,4))
'''
array([[8, 0, 3, 7],
       [4, 6, 3, 4],
       [6, 9, 9, 8],
       [9, 1, 4, 0],
       [5, 9, 0, 5]])
'''

np.random.randint(0,10,3)# array([7, 7, 5])
~~~

#### normal

~~~python
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10)
'''
array([ 0.05754667, -0.07006152,  0.06810326, -0.11012173,  0.10064039,
       -0.06935203,  0.14194363,  0.07428931, -0.07412772,  0.12112031])
'''
~~~

#### shuffle

~~~python
array = np.arange(10)
np.random.shuffle(tang_array)# array([6, 2, 5, 7, 4, 3, 1, 0, 8, 9])
~~~

#### seed

~~~python
np.random.seed(100)  # 当种子的值不变时，无论执行多少次生成的随机数都不变
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10)
# array([-0.17,  0.03,  0.12, -0.03,  0.1 ,  0.05,  0.02, -0.11, -0.02,  0.03])
~~~



### 排序

#### sort

~~~python
array = np.array([[1.5,1.3,7.5],
                      [5.6,7.8,1.2]])
np.sort(array)  #默认按行排序
'''
array([[ 1.3,  1.5,  7.5],
       [ 1.2,  5.6,  7.8]])
'''
np.sort(array,axis = 0) #按照列排序
'''
array([[ 1.5,  1.3,  1.2],
       [ 5.6,  7.8,  7.5]])
'''
~~~

#### argsort

~~~python
np.argsort(array) # 排序前的索引
'''
array([[1, 0, 2],
       [2, 0, 1]], dtype=int64)
'''
~~~

####  searchsorted

~~~python
array = np.linspace(0,10,10)
'''
array([  0.        ,   1.11111111,   2.22222222,   3.33333333,
         4.44444444,   5.55555556,   6.66666667,   7.77777778,
         8.88888889,  10.        ])
'''
values = np.array([2.5,6.5,9.5])
np.searchsorted(tang_array,values) # 在排序好的数组中插入的位置
# array([3, 6, 9], dtype=int64)
~~~

#### lexsort

~~~python
array = np.array([[1,0,6],
                       [1,7,0],
                       [2,3,1],
                       [2,4,0]])
index = np.lexsort([-1*array[:,0],array[:,2]]) # 按照第一列降序，第二列升序
# array([0, 1, 3, 2], dtype=int64)
array =array[index]
'''
array([[2, 4, 0],
       [1, 7, 0],
       [2, 3, 1],
       [1, 0, 6]])
'''
~~~

### 输出

#### set_printoptions

~~~python
np.set_printoptions(precision = 2)
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10) 
# array([ 0.01,  0.02,  0.12, -0.01, -0.04,  0.07,  0.14, -0.08, -0.01, -0.03])
~~~



### 读写

####  writefile

~~~python
%%writefile tang.txt
1 2 3 4 5 6
2 3 5 8 7 9
~~~

#### readlines

~~~~python
data = []
with open('tang.txt') as f:
    for line in f.readlines():
        fileds = line.split()
        cur_data = [float(x) for x in fileds]
        data.append(cur_data)
data = np.array(data)
'''
data=array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
'''
~~~~

#### loadtxt

~~~python
data = np.loadtxt('tang.txt')
'''
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
'''
~~~

##### delimiter

~~~python
%%writefile tang2.txt
1,2,3,4,5,6
2,3,5,8,7,9
data = np.loadtxt('tang2.txt',delimiter = ',')
'''
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
'''
~~~

##### skiprows

~~~~python
%%writefile tang2.txt
x,y,z,w,a,b
1,2,3,4,5,6
2,3,5,8,7,9
data = np.loadtxt('tang2.txt',delimiter = ',',skiprows = 1)
'''
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
'''
~~~~

##### usecols

~~~python
%%writefile tang2.txt
x,y,z,w,a,b
1,2,3,4,5,6
2,3,5,8,7,9
data = np.loadtxt('tang2.txt',delimiter = ',',skiprows = 1，usecols=(0,1,4))
'''
array([[ 1.,  2.,  5.],
       [ 2.,  3.,  7.]])
'''
~~~

#### savetxt

~~~python
tang_array = np.array([[1,2,3],[4,5,6]])
np.savetxt('tang4.txt',tang_array)
np.savetxt('tang4.txt',tang_array,fmt='%d')
np.savetxt('tang4.txt',tang_array,fmt='%d',delimiter = ',')
np.savetxt('tang4.txt',tang_array,fmt='%.2f',delimiter = ',')
~~~

#### save/load

读写array结构

~~~python
tang_array = np.array([[1,2,3],[4,5,6]])
np.save('tang_array.npy',tang_array)
tang = np.load('tang_array.npy')
'''
tang= array([[1, 2, 3],
       [4, 5, 6]])
'''
~~~

压缩读取多个array

~~~python
tang_array2 = np.arange(10)
np.savez('tang.npz',a=tang_array,b=tang_array2) #压缩存放
data = np.load('tang.npz')
data.keys() # ['b', 'a']
data['a'] 
'''
array([[1, 2, 3],
       [4, 5, 6]])
'''
data['b']
'''
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
'''
~~~


