# Most of these algorithms are derived from psuedocode in Cormen et. al. 

inf = float('inf')

class Vertex(object):
	""" Adjacent list implementation of a graph. Create a vertex (v) and then its adjacent edges are stored
	in the list v.edges. A graph will then just be a set of vertex objects. Can use this to implement
	directed/undirected and weighted/unweighted graphs easily."""

	def __init__(self, adj=[]):
		self.adjacent = adj
		
	def get_edges(self, adj_vert):
		""" Input an array-like object "adj_vert" containing all the vertices directly connected to our
			current vertex."""
		for u in adj_vert:
			self.adjacent.append(u)
			
	def __repr__(self):
		return str(self)
	
	def __str__(self):
		return str(self)
	
class Priority_Queue_Min(object):
	""" Heap implementation of priority queue of objects with priorities (keys) stored in 
		a dictionary P."""
	
	def __init__(self, A=[],P={}):
		self.heap = A
		self.P = P
		if len(self.heap) > 0:
			self.build_min_heap()
	
	def min_heapify(self,i):
		P = self.P
		l = 2*i + 1
		r = 2*i + 2
		if l < len(self.heap) and P[self.heap[l]] < P[self.heap[i]]:
			smallest = l
		else:
			smallest = i
		if r < len(self.heap) and P[self.heap[r]] < P[self.heap[smallest]]:
			smallest = r
		if smallest != i:
			self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
			self.min_heapify(smallest)
			
	def build_min_heap(self):
		k = (len(self.heap) // 2) - 1
		for i in range(k,-1,-1):
			self.min_heapify(i)
	
	def extract_min(self):
		if len(self.heap) < 1:
			print('heap-underflow')
			return
		m = self.heap[0]
		self.heap[0] = self.heap[-1]
		self.heap = self.heap[:-1]
		self.min_heapify(1)
		return m
	
	def decrease_key(self,x,key):
		i = self.heap.get_index(x)
		if key > self.P[self.heap[i]]:
			print('Can only decrease key value')
			return
		self.P[self.heap[i]] = key
		while i > 0 and self.P[self.heap[(i-1)//2]] > self.P[self.heap[i]]:
			self.heap[i], self.heap[(i-1)//2] = self.heap[(i-2)//2], self.heap[i]
			i = (i-1) // 2
			
	def insert(self,x,key_x):
		self.heap.append(x)
		self.P[x] = inf
		self.decrease_key(len(self.heap),key_x)
		
# Single Shortest Paths

def init_single_source(G,s):
	P = {}
	pi = {}
	for v in G:
		P[v] = inf
		pi[v] = None
	P[s] = 0
	return P, pi
	
def relax(u,v,P,pi,w):
	""" Sub routine that will update the distances. Here w is the dictionary encoding the edge weights,
	    i.e. w[(u,v)] = weight of edge from u to v."""
	P, pi = P, pi
	if P[v] > P[u] + w[(u,v)]:
		P[v] = P[u] + w[(u,v)]
		pi[v] = u
	return P, pi

def Bellman_Ford(G,w,s):
	""" Bellman-Ford algorithm for finding a single shortest path from a source, s, to every vertex
		in the connected component containing s. This algorithm is less efficient than Dijkstra's
		but can handle the case of negative edge weights. It also has a built in infinite, negative-weight
		cycle detector."""
	P, pi = init_single_source(G,s)
	for i in range(len(G)-1):
		for v in G:
			for u in v.adjacent:			# These two lines (100-101) are equivalent to looping over the edges of the graph.
				P, pi = relax(v,u,P,pi,w)
	for v in G:							# Infinite negative cycle detection.
		for u in v.adjacent:
			if P[u] > P[v] + w[(v,u)]:
				print('Negative weight cycle detected')
				return False
	return P, pi
				
def Dijkstra(G,w,s):
	""" Dijkstra's is faster than Bellman-Ford, but is only valid when the weights are positive."""
	P, pi = init_single_source(G,s)
	S = []
	Q = Priority_Queue_min(G,P)
	while len(Q.heap) > 0:
		u = Q.extract_min()
		S.append(u)
		for v in u.adjacent:
			sum = Q.P[u] + w[(u,v)]
			if Q.P[v] > sum:
				Q.decrease_key(v,sum)
				pi[v] = u
	P = Q.P
	return P, pi

def find_shortest_path(G,w,s,t):
	""" This function will print a shortest path from s to t, as well as its weight. Assumes no
		negative cycles."""
	m = min(list(w.values()))
	if m >= 0:
		P, pi = Dijkstra(G,w,s)
	else:
		P, pi = Bellman_Ford(G,w,s)
	if P[t] < inf:
		path = []
		p = t
		while P[p] != 0:
			path = [str(p)] + path
			p = pi[p]
		print("-->".join(path))
		print('Weight = ', P[t])
	else:
		print('They are not connected')
		

# Bredth First Search

def BFS(G,s):
	color = {}
	P = {}
	pi = {}
	for v in G:
		color[v] = 'white'
		pi[v] = None
		P[v] = inf
	color[s] = 'grey'
	P[s] = 0
	Q = [s]
	while Q:
		u = Q.pop(0)
		for v in u.adjacent:
			if color[v] == 'white':
				color[v] = 'grey'
				P[v] = P[u] + 1
				pi[v] = u
				Q.append(v)
		color[u] = 'black'
	return P, pi

# Minimum Spanning Tree

def Prim(G,w,r):
	""" Prim's algorithm starting with arbitrary root vertex r. Here we assume G is connected and undirected.
		This function returns the dictionary 'pi' which keeps track of the parent of a vertex in the MST if 
		the vertex is in the MST and keeps it NIL otherwise. Can easily use this info to print out the tree."""
	P, pi = {}, {}
	for u in G:
		P[u] = inf
		pi[u] = None
	P[r] = 0
	Q = Priority_Queue_Min(G,P)
	while Q.heap:
		u = Q.extract_min()
		for v in u.adjacent:
			if v in Q.heap and w[(u,v)] < Q.P[v]:
				pi[v] = u
				Q.decrease_key(v,w[(u,v)])
	return pi
