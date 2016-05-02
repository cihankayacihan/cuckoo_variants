import sys
import numpy as np
import scipy.special as sp
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools

def fobj(u):
	x=u[0]
	y=u[1]
	z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	return z

def simple_bounds(s, Lb, Ub):

	for i in range(len(s)):
		if s[i] > Ub[i]:
			s[i] = Ub[i]
		elif s[i] < Lb[i]:
			s[i] = Lb[i] 
	return s

def all_bounds(nest, Lb, Ub):

	for i in range(nest.shape[0]):
		nest[i,:]=simple_bounds(nest[i,:],Lb,Ub)

	return nest

def get_best_nest(nest, newnest, fitness):

	n = nest.shape[0]
	d = nest.shape[1]
	velocity = np.zeros((n,d))
	for i in range(n):
		fnew=fobj(newnest[i,:])
		if fnew < fitness[i]:
			velocity[i,:]=(newnest[i,:]-nest[i,:])*(fitness[i]-fnew)
			fitness[i]=fnew
			nest[i,:]=newnest[i,:]

	fmin = min(fitness)
	K = np.argmin(fitness)
	best = nest[K,:]
	return (fmin, best, nest, fitness, velocity)

def get_cuckoos(nest, best, Lb, Ub):
	n = nest.shape[0]
	beta=3/2;
	sigma=(sp.gamma(1+beta)*np.sin(np.pi*beta/2)/(sp.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
	for j in range(n):
		s = nest[j,:]
		u=np.random.randn(s.shape[0])*sigma
		v=np.random.randn(s.shape[0])
		step=u/abs(v)**(1/beta)
		stepsize=0.01*step*(s-best);
		s=s+stepsize*np.random.randn(s.shape[0])
		nest[j,:]=simple_bounds(s, Lb, Ub)
	return nest

def empty_nests(nest, Lb, Ub, pa):
	n = nest.shape[0]
	K = np.random.random(nest.shape) > pa
	stepsize = np.random.rand()*(nest[np.random.permutation(n),:]-nest[np.random.permutation(n),:])
	new_nest = nest + stepsize * K
	for j in range(new_nest.shape[0]):
		s = new_nest[j,:]
		new_nest[j,:] = simple_bounds(s, Lb, Ub)
	return new_nest

def getCommunication(nest, velocity, co, ca):
	n = nest.shape[0]
	d = nest.shape[1]
	communication = np.zeros((n,d))
	for i in range(n):
		for j in range(n):
			if i != j and np.linalg.norm(nest[i,:]-nest[j,:])>1e-4:
				communication[i,:]=communication[i,:]+np.exp(-co*np.linalg.norm(nest[i,:]-nest[j,:]))*velocity[j,:] \
				+ (np.exp(-ca*np.linalg.norm(nest[i,:]-nest[j,:]))*(nest[i,:]-nest[j,:]))/np.linalg.norm(nest[i,:]-nest[j,:])

	for i in range(communication.shape[0]):
		if np.linalg.norm(communication[i,:]) != 0.:
			communication[i,:] = communication[i,:]/np.linalg.norm(communication[i,:])
	return communication


def cuckoo_search(n=None, nd=None, Lb=None, Ub=None, pa=None):
	if n is None:
		n =25

	if nd is None:
		nd=2

	if Lb is None:
		Lb = np.ones(nd)*0
	if Ub is None:
		Ub = np.ones(nd)*5

	# creation of the list for parameter pairs 
	pa = 0.25
	step = 1
	co = 0.001
	ca = 0.1
	weight = 0.01
	#all_para = list(itertools.product(pa,step))
	#threads = len(all_para)

    # initialization of the nests for each system
	Tol = 1.e-12
	nests = np.zeros((n,nd))
	for i in range(n):
		nests[i,:] = Lb + (Ub-Lb)*np.random.rand(len(Lb))


	fitness = 10**10 * np.ones((n,1))

	for i in range(1):
		best_nest, fmin, nest, fitness = single_cuckoo_search(nests,fitness,Lb,Ub,pa,step,co,ca,weight) 

	return best_nest, fmin, nest, fitness


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step,co,ca,weight):

	n = nest.shape[0]
	d = nest.shape[1]
	fmin, best, nest, fitness, velocity = get_best_nest(nest, nest, fitness)
	new_best = fmin
	old_best = np.inf
	bestnest = best
	

	N_iter=0

	while N_iter < 25:
		new_nest = get_cuckoos(nest, best, Lb, Ub)
		fnew, best, nest, fitness, velocity = get_best_nest(nest, new_nest, fitness)

		N_iter = N_iter + n
		communication = getCommunication(nest, velocity, co, ca)
		velocity = velocity + weight * communication
		print communication
		nest = nest + velocity
		nest = all_bounds(nest, Lb, Ub)

		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness, velocity = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n 
		if fnew < fmin:
			fmin=fnew
			bestnest=best
		print bestnest, fmin

	return bestnest, fmin, nest, fitness

best_nest, fmin, nest, fitness = cuckoo_search()
