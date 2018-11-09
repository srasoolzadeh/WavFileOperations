xs=[3, 1, 2]
print(xs, xs[2])
xs.append('foo')
print(xs)

#List
nums=list(range(5))
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums)
print(nums[:-1])
nums[5:6]=[12, 40000]
print(nums)

sqrs=[]
for i in nums:
    sqrs.append(i**2)
print(sqrs)
news=[i*2 for i in nums]
print(news)
nns=[i*3 for i in nums if i%2==0]
print(nns)

#Dictionary
d={'one': 1, 'two':2, 'three':3}
print(d)
print(d['one'])
d['ors']=4
print(d)
for objs in d:
    print(objs)
    
#Create array by numpy library
import numpy as np
a=np.array([1, 2, 3])
print(a.shape)
print(type(a))
b=np.array([[1, 2, 3],[4,5,6]])
print(b)
print(b.shape)
b[1,2]=67
print(b)
c=np.zeros(5)
print(c, c.shape)
d=np.full((2,4), 30)
print(d, d.shape)
e=np.random.random((4,2))
print(e, e.shape)
a3=np.array([[[1, 2, 3],[4,5,6]], [[10, 20, 30],[40,50,60]]])
print(a3, a3.shape)

#create subarray by Slicing
print("------ Slicing => view -----------")
a1=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a1, a1.shape)
b=a1[:2, 1:3]
print(b, b.shape)
b[0,0]=34
print(a1)

#Row Slicing
r1=a1[1,:]
r2=a1[1:2, :]
print("r1:",r1, r1.shape)
print("r2:", r2, r2.shape)
r1[1]=18
print("a1:", a1)
r2[0,1]=22;
print("a1:", a1)

#column slicing
c1=a1[:, 2]
c2=a1[:,2:3]
print("c1:", c1, c1.shape)
print("c2:", c2, c2.shape)
print("------ integer indexing ----------")
#create rank 1 array directly from another array by integer indexing
p1=a1[[0,1,2],[0,1,0]]
print("p1:", p1, p1.shape)
p1[0]=99
print("a1 after man:", a1)
#create list from array
p2=[a1[0,0],a1[1,1]]
print("List p2:",p2)

#create rank 1 array by integer indexing
p3=np.array([a1[0,0],a1[1,1]])
print("p3:", p3, p3.shape)

#create rank 2 array by integer indexing
p4=np.array([[a1[0,0],a1[1,1]],[a1[0,0],a1[1,1]]])
print("p4:", p4, p4.shape)
#important point: Slicing=>view , Integer indexing=>New array
#Create another array by indices
a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print("a:", a, a.shape)
b=np.array([0,2,0,1]) # indices array
print("b:", b, b.shape)
c=np.array([a[0,2], a[2,0]])
print("c:", c, c.shape, type(c))
# direct integer indexing
d=a[np.arange(4), b] # ****** good example
print("d:", d, d.shape, type(d))
e=a[np.arange(2), np.arange(1)]
print("e:", e, e.shape, type(e))
f=a[[0,1], [0]] # equal to e
print("f:", f, f.shape, type(e))
g=a[[0,1,0,2],[0,0,2,2]]
print("g:", g, g.shape)
#manipulating
a[np.arange(4), b] += 10
print ("a:", a)
g[0]+=1
print("g:", g)
print("a after manipulate g:", a)
print("----------- test manipulation ------------")
s=np.array([[1,2,3],[4,5,6]])
print("S:", s, s.shape)
s1=s[:1, 0:3] # slicing
print("s1:", s1, s1.shape)
s1[0, 1]=90 # slice manipulate effects main array
print("s:", s)
s2=np.array(s[[0,1],[1,2]])      
print("s2:", s2, s2.shape)
s3=s[[0,0,0],[0,1,2]]
print("s3:", s3, s3.shape)
s4=np.array([s[[0,0,0],np.arange(3)],s[[1,1,1],np.arange(3)]])
print("s4:", s4, s4.shape)
s5=np.array([s[np.array(np.zeros(3), dtype=np.int32),np.arange(3)],s[[1,1,1],np.arange(3)]])
print("s5:", s5, s5.shape)

print("------------ Boolean indexing --------------")
a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
boolindex= a>2
print("index a>2:", boolindex)
b=a[a>2]
print("b=a[a>2]:", b)

print("--------- array math -------------")
x=np.array([[1,2],[3,4]], dtype=np.int32)
y=np.array([[5,6],[7,8]], dtype=np.int32)
print("x:",x,"\ny:", y)
sumxy=x+y
print("x+y: ",sumxy)
divxy=x/y
print("x/y:", divxy)
print("sqrt(x):", np.sqrt(x))
v=np.array([9,10])
print("v:", v)
q1=v.dot(x)
print("v.dot(x):", q1)
q2=x.dot(v)
print("x.dot(v):", q2)
q3=np.dot(x,y)
print("dot(x,y):", q3)
sumallx=np.sum(x)
print("sumall x:", sumallx, sumallx.shape)
sumcolx=np.sum(x, axis=0)
print("sum column x:", sumcolx, sumcolx.shape)
sumrowx=np.sum(x, axis=1)
print("sum row x:", sumrowx, sumrowx.shape)
# ---- transpose -----
xt=x.T
print("transpose (x):", xt)
#----- broadcasting ------
v=np.array([30, 40])
for i in range(2):
    x[i, :] += v
print("x ", x)
y+=v
print("y ", y)
#---- reshape -----
r1=x.reshape(4,1)
print("r1 ", r1)
r2=x.reshape(1,4)
print("r2 ", r2)

print("------------ matplotlib --------------------")
import matplotlib.pyplot as plt
x=np.arange(0, 2*np.pi, 0.1)
y=np.sin(x)
plt.plot(x,y)
#plt.show()
y_cos=np.cos(x)
plt.plot(x, y_cos)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Sine , Cosine")
plt.legend(["sine", "cos"])
plt.show
