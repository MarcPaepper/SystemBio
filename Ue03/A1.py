import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load CSV
data = np.genfromtxt('Ue03/phenoData.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
arr = np.array([list(row) for row in data])

# Remove rows containing 'NA'
mask = np.any(arr == 'NA', axis=1)
clean_arr = arr[~mask]

# Extract columns safely
def extract_numeric_column(arr, col_name):
    idx = data.dtype.names.index(col_name)
    col_data = []
    for val in clean_arr[:, idx]:
        try:
            col_data.append(float(val))
        except ValueError:
            continue
    return np.array(col_data)

age = extract_numeric_column(clean_arr, 'age')
height = extract_numeric_column(clean_arr, 'height')

def compute_stats(col_data):
    minimum = np.min(col_data)
    maximum = np.max(col_data)
    median = np.median(col_data)
    mean = np.mean(col_data)
    mode_val = stats.mode(col_data, keepdims=True).mode[0]
    variance = np.var(col_data, ddof=1)
    std_dev = np.std(col_data, ddof=1)
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    return {
        'min': minimum, 'max': maximum, 'range': maximum - minimum, 
        'median': median, 'mean': mean, 'mode': mode_val,
        'variance': variance, 'std_dev': std_dev,
        'q1': q1, 'q3': q3, 'iqr': iqr
    }

age_stats = compute_stats(age)
height_stats = compute_stats(height)

print("Age stats:", age_stats)
print("Height stats:", height_stats)

plt.figure(figsize=(12,5))

# Age histogram
plt.subplot(1,2,1)
plt.hist(age, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Height histogram
plt.subplot(1,2,2)
plt.hist(height, bins=10, color='salmon', edgecolor='black')
plt.title('Histogram of Height')
plt.xlabel('Height')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
