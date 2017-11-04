#!/usr/bin/env
from __future__ import division
import pdb
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pdb
import sympy as sp
import mpmath as mp

###### Problem 1 #######


# calculate distance when N=10,000 and p=1000
def distance_closest(p,N):
	d = (1 - (float(0.5))**(1/N))**(1/p)
	return d

def p1():
	p = 1000
	N = 10000

	solution1 = distance_closest(p, N)
	print('p=%s, N=%s, Solution 1: %s' %(p, N,solution1))

# p_range = np.arange(p)
# N_range = np.arange(N)
# for p1 in p_range:
# 	for N1 in N_range:
# 		solution1 = distance_closest(p1, N1)
# 		print('p=%s, N=%s, Solution 1: %s' %(p1, N1,solution1))
# pdb.set_trace()





####### Problem 3 ############
def f3(x1, x2):
	y = 8*x1 + 12*x2 + x1**2 - 2*x2**2
	return y

def p3():
	x_range = np.arange(-20,20)
	X1, X2 = np.meshgrid(x_range, x_range)
	y = np.ones((len(x_range), len(x_range)))

	i1 = 0
	for x1 in x_range:
		i2 = 0
		for x2 in x_range:
			y[i1, i2] = f3(x1,x2)
			i2 += 1
		i1 += 1

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.scatter(X1, X2, y)
	x1_saddle = -4
	x2_saddle = 3
	y_saddle = f2(x1_saddle, x2_saddle)
	ax.scatter(x1_saddle, x2_saddle, y_saddle, c='r', marker='o', s=200)
	plt.show()

####### Problem 2 ######

def f2(x1,x2):
	y = (x1 + x2)*(x1*x2 + x1*x2**2)
	return y

def p2():
	x_range = np.arange(-20,20)
	X1, X2 = np.meshgrid(x_range, x_range)
	y = np.ones((len(x_range), len(x_range)))

	i1 = 0
	for x1 in x_range:
		i2 = 0
		for x2 in x_range:
			y[i1, i2] = f2(x1,x2)
			i2 += 1
		i1 += 1

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.scatter(X1, X2, y)
	pdb.set_trace()
	# x1_saddle = -4
	# x2_saddle = 3
	# y_saddle = f3(x1_saddle, x2_saddle)
	# ax.scatter(x1_saddle, x2_saddle, y_saddle, c='r', marker='o', s=200)
	plt.show()

######## Problem 2: Symbolic #########
def p2_sym():
	sp.init_printing()
	x1, x2, l = sp.symbols("x1 x2 l")
	f = (x1 + x2) * (x1*x2 + x1*x2**2)
	d_x1 = sp.expand(sp.diff(f, x1))
	d_x2 = sp.expand(sp.diff(f, x2))
	print('f/dx1:')
	print(sp.latex(d_x1))
	print('f/dx2:')
	print(sp.latex(d_x2))
	# d_x1_f = sp.lambdify((x1, x2), d_x1)
	# z1 = mp.findroot(d_x1_f, (0,0))
	# print('Zeros for f/dx1:')
	# print(sp.latex(z1))
	for i1 in np.arange(-10, 10,0.25):
		for i2 in np.arange(-10, 10, 0.25):
			res1 = d_x1.subs(x1, i1).subs(x2, i2)
			res2 = d_x2.subs(x1, i1).subs(x2, i2)
			if res1 == 0 and res2 == 0:
				print('x1: %s, x2: %s' %(i1, i2))

	d2_x1 = sp.expand(sp.diff(d_x1, x1))
	d2_x2 = sp.expand(sp.diff(d_x2, x2))
	d2_x1x2 = sp.expand(sp.diff(d_x1, x2))
	print('f/d2x1:')
	print(sp.latex(d2_x1))
	print('f/d2x2:')
	print(sp.latex(d2_x2))
	print('f/dx1dx2:')
	print(sp.latex(d2_x1x2))

	st_pts = [(0,0), (0,-1), (1,-1), (3/8, -6/8)]
	for x,y in st_pts:
		res1 = d2_x1.subs(x1, x).subs(x2, y)
		res2 = d2_x2.subs(x1, x).subs(x2, y)
		res3 = d2_x1x2.subs(x1, x).subs(x2, y)
		Hes = [[res1, res3], [res3, res2]]
		print('Hessian at %s,%s:' %(x,y))
		print(Hes)
	pdb.set_trace()
	lambda_H = (d2_x1 - l)*(d2_x2 - l) - d2_x1x2**2
	print('Eigenvalue Equation: ')
	print(sp.simplify(sp.expand(lambda_H)))
p2_sym()

