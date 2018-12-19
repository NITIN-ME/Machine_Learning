import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn

x = np.array([[1, 2, 3], [4, 5, 6]]);
print("x:{}".format(x))
print("\n-------------------------------------------\n")

eye = np.eye(4)
print("Numpy array: \n{}".format(eye))
print("\n-------------------------------------------\n")

sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n{}".format(sparse_matrix))
print("\n-------------------------------------------\n")

data = np.ones(4)
row_indices = np.arange(4)
col_indicies = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indicies)))
print("COO representation:\n{}".format(eye_coo))
print("\n-------------------------------------------\n")

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = "x")
#plt.show()

data = {'Name': ["John", "Anna", "Peter", "Linda"],
		'Location' : ["New York", "Paris", "Berlin", "London"],
		'Age' : [24, 13, 53, 33]
	}
data_pandas = pd.DataFrame(data)
display(data_pandas)
print("\n-------------------------------------------\n")
display(data_pandas[data_pandas.Age > 30])

print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("NumPy version: {}".format(np.__version__))
print("SciPy version: {}".format(sp.__version__))
print("IPython version: {}".format(IPython.__version__))
print("scikit-learn version: {}".format(sklearn.__version__))
