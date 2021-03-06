from numpy import *

def compute_error(b, m, points):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m*x + b))**2
	totalError /= float(len(points))
	return totalError



def step_gradient(b_current, m_current, points, learning_rate):
	#gradient_descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N)*(y - ((m_current * x) + b_current))
		m_gradient += -(2/N)*x*(y - ((m_current * x) + b_current))
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return new_b,new_m




def gradient_descent_runner(points, initial_b, initial_m, num_iterations,learning_rate):
	b = initial_b
	m = initial_m
	for i in range(num_iterations):
		b,m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def run():
	points = genfromtxt("data.csv", delimiter = ',')
	learning_rate = 0.0001
	# y = mx + b
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, num_iterations,learning_rate)
	print(b)
	print(m)


if __name__ == '__main__':
	run()
