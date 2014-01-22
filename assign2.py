#import packages as according to instructions
from PIL import Image
import numpy as np
import math
from scipy import signal

#1
def boxfilter(n):
	assert n % 2 == 1, "Dimension must be odd"

	#If sum of all entry in the matrix must add up to one,
	#each matrix can be calculated as following:
	weight = 1.0 / n / n

	#Initialize a n by n matrix of 1's and multiply the weight
	return np.ones((n,n)) * weight

'''
Output:
>>> boxfilter(3)
array([[ 0.11111111,  0.11111111,  0.11111111],
       [ 0.11111111,  0.11111111,  0.11111111],
       [ 0.11111111,  0.11111111,  0.11111111]])
>>> boxfilter(4)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "assign2.py", line 9, in boxfilter
    assert n % 2 == 1, "Dimension must be odd"
AssertionError: Dimension must be odd
>>> boxfilter(5)
array([[ 0.04,  0.04,  0.04,  0.04,  0.04],
       [ 0.04,  0.04,  0.04,  0.04,  0.04],
       [ 0.04,  0.04,  0.04,  0.04,  0.04],
       [ 0.04,  0.04,  0.04,  0.04,  0.04],
       [ 0.04,  0.04,  0.04,  0.04,  0.04]])
'''

#2
def gauss1d(sigma):
	len = round(sigma * 6)
	#If it's even, round it to the next odd by adding one
	if len % 2 == 0:
		len  += 1

	#Set up the array with incrementing by one e.g. [-2, -1, 0, 1, 2]
	arr = np.arange(round(-len/2+1), len/2, 1)
	#Apply the Gaussian function
	arr = np.exp( -1 * np.power(arr, 2) / (2 * np.power(sigma, 2)) )
	#Normalize by dividing by the sum
	return arr/sum(arr)

'''
Output:
>>> gauss1d(0.3)
array([ 0.00383626,  0.99232748,  0.00383626])
>>> gauss1d(0.5)
array([ 0.10650698,  0.78698604,  0.10650698])
>>> gauss1d(1)
array([ 0.00443305,  0.05400558,  0.24203623,  0.39905028,  0.24203623,
        0.05400558,  0.00443305])
>>> gauss1d(2)
array([ 0.0022182 ,  0.00877313,  0.02702316,  0.06482519,  0.12110939,
        0.17621312,  0.19967563,  0.17621312,  0.12110939,  0.06482519,
        0.02702316,  0.00877313,  0.0022182 ])
'''
 
#3
def gauss2d(sigma):
	#Get one axis of the 2D Gaussian filter by using the gauss1d 
	#we already have and converting it to 2D
	a = gauss1d(sigma)
	a = a[np.newaxis]

	#Do the same for the other axis, but make sure to transpose it
	b = gauss1d(sigma)
	b = np.transpose(b[np.newaxis])

	#Convolve the two matrices to create 2D Gaussian filter
	return signal.convolve2d(a, b)
'''
>>> gauss2d(0.5)
array([[ 0.01134374,  0.08381951,  0.01134374],
       [ 0.08381951,  0.61934703,  0.08381951],
       [ 0.01134374,  0.08381951,  0.01134374]])
>>> gauss2d(1)
array([[  1.96519161e-05,   2.39409349e-04,   1.07295826e-03,
          1.76900911e-03,   1.07295826e-03,   2.39409349e-04,
          1.96519161e-05],
       [  2.39409349e-04,   2.91660295e-03,   1.30713076e-02,
          2.15509428e-02,   1.30713076e-02,   2.91660295e-03,
          2.39409349e-04],
       [  1.07295826e-03,   1.30713076e-02,   5.85815363e-02,
          9.65846250e-02,   5.85815363e-02,   1.30713076e-02,
          1.07295826e-03],
       [  1.76900911e-03,   2.15509428e-02,   9.65846250e-02,
          1.59241126e-01,   9.65846250e-02,   2.15509428e-02,
          1.76900911e-03],
       [  1.07295826e-03,   1.30713076e-02,   5.85815363e-02,
          9.65846250e-02,   5.85815363e-02,   1.30713076e-02,
          1.07295826e-03],
       [  2.39409349e-04,   2.91660295e-03,   1.30713076e-02,
          2.15509428e-02,   1.30713076e-02,   2.91660295e-03,
          2.39409349e-04],
       [  1.96519161e-05,   2.39409349e-04,   1.07295826e-03,
          1.76900911e-03,   1.07295826e-03,   2.39409349e-04,
          1.96519161e-05]])
'''

#4
def gaussconvolve2d(image,sigma):
	#Load up the Gaussian filter in a variable named filter
	filter = gauss2d(sigma)
	#Take the dir from input, open the image and convert to greyscale
	im = Image.open(image).convert('L')
	
	#Convert it to a numpy array and apply the Gaussian filter
	im = signal.convolve2d(np.asarray(im),filter,'same')

	#Transform the array back to an image, remember to output as a greyscale as well
	out = Image.fromarray(im).convert('L')

	#Save the image as a .jpd
	out.save('output.jpg', 'JPEG')

'''
Why does Scipy have separate functions 'signal.convolve2d' and 'signal.correlate2d'?

'''

'''
5.
I initially thought that outer multiplication would be the fastest way 
to construct the 2D Gaussian filter, but we were not allowed to do that
for number 3. 2D Gaussian can be expressed as two seperate functions,
one for x and the other for y. It also is the case that the two functions
are identical as 1D Gaussian. So instead of constructing it by using two
seperate filters, adding new axis, transposing it, convolving it, if we 
simply did an outer multiplication on our 1D Gaussian to itself, it would
have been faster, both in computation and in length of code.
'''













