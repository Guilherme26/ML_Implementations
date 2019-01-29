"""
	This module is able to solve the problem of finding 
	the Simple Linear Regression on a distribution. The 
	model attempts to find the parameters disposition 
	that causes the minimum error considering the whole data
"""
import copy

class LinearRegression():
	theta = [0,0]
	alpha = 0.0005
	x = []
	y = []
	_noChange = 0

	def __init__(self, alpha=0.0005, theta=[0,0]):
		self.alpha = alpha
		self.theta = theta
		self.x, self.y = [], []
		self._noChange = 0

	def fit(self, x, y):
		self.x, self.y = x, y
		while True:
			# The algorithm converged
			if self._noChange > 3:
				break

			self._updateWeight()

	def predict(self, xi):
		return self.theta[0] + self.theta[1]*xi

	"""
		This function calculates the gradient descent until convergence, 
		that is, not representative changes in theta parameters
	"""
	def _updateWeight(self):
		# Makes a shallow copy of the object
		theta_tmp = copy.copy(self.theta)
		err_sum = self._calc_error()
		
		theta_tmp[0] = theta_tmp[0] - (self.alpha / len(self.x) * err_sum[0])
		theta_tmp[1] = theta_tmp[1] - (self.alpha / len(self.x) * err_sum[1])

		if (round(theta_tmp[0],4) == round(self.theta[0],4) and round(theta_tmp[1],4) == round(self.theta[1],4)):
			self._noChange += 1
		else:
			self._noChange = 0

		self.theta = theta_tmp

	"""
		This is the hypotesis assumed to this problem, 
		and it considers that data has linear distribution
	"""
	def _hypothesis(self, xi):
		return self.theta[0] + self.theta[1]*xi

	def _calc_error(self):
		err_sum = [0,0]
		
		for xi, yi in zip(self.x, self.y):
			err_sum[0] = err_sum[0] + (self._hypothesis(xi) - yi)
			err_sum[1] = err_sum[1] + (self._hypothesis(xi) - yi) * xi

		return err_sum
