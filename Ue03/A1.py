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
weight = extract_numeric_column(clean_arr, 'weight')
tin_median = extract_numeric_column(clean_arr, 'TINmedian')

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

# Define custom bins for age (10-14.999..., 15-19.999..., etc.)
age_min = 10 * (np.floor(np.min(age) / 10))
age_max = 10 * (np.ceil(np.max(age) / 10))
age_bins = np.arange(age_min, age_max + 5, 5)  # 10-14.999..., 15-19.999..., etc.

# Define custom bins for height (10-19.999..., 20-29.999..., etc.)
height_min = 10 * (np.floor(np.min(height) / 10))
height_max = 10 * (np.ceil(np.max(height) / 10))
height_bins = np.arange(height_min, height_max + 10, 10)  # 10-19.999..., 20-29.999..., etc.

# Age histogram
plt.subplot(1,2,1)
plt.hist(age, bins=age_bins, color='skyblue', edgecolor='black', rwidth=0.9)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(age_bins)

# Height histogram
plt.subplot(1,2,2)
plt.hist(height, bins=height_bins, color='salmon', edgecolor='black', rwidth=0.9)
plt.title('Histogram of Height')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.xticks(height_bins)

plt.tight_layout()
# plt.show()
plt.close()




# --- Boxplot Section ---
# Outlier cutoff factors for each variable (edit these as needed)
outlier_cutoffs = {
    'Age': np.inf,      # Extend whiskers to min/max (no outliers)
    'Weight': 1.5,      # Standard 1.5 IQR for outlier detection
    'Height': 1.5,      # Standard 1.5 IQR for outlier detection
    'TIN.median': 1.5   # Standard 1.5 IQR for outlier detection
}

# Improved boxplot style
plt.figure(figsize=(10, 6))
labels = ['Age', 'Weight', 'Height', 'TIN.median']
data_list = [age, weight, height, tin_median]


# Style: all boxes same color, black median, no mean/outliers, solid whiskers, y starts at 0
# Define boxplot style properties
boxprops = dict(linestyle='-', linewidth=2, color='navy')  # Box border style
medianprops = dict(linestyle='-', linewidth=2.5, color='black')  # Median line style
capprops = dict(linestyle='-', linewidth=2, color='navy')  # Caps at ends of whiskers
whiskerprops = dict(linestyle='-', linewidth=2, color='navy')  # Whisker line style

# Create the boxplot
box = plt.boxplot(
    data_list,                # List of arrays to plot
    labels=labels,            # X-axis labels
    patch_artist=True,        # Enable box fill color
    showmeans=False,          # Do not show mean marker
    showfliers=False,         # Do not show outliers
    boxprops=boxprops,        # Box style
    medianprops=medianprops,  # Median line style
    capprops=capprops,        # Cap style
    whiskerprops=whiskerprops # Whisker style
)

# Set all boxes to the same color
box_color = '#5DADE2'  # Chosen fill color for all boxes
for patch in box['boxes']:
    patch.set_facecolor(box_color)  # Apply fill color to each box

plt.title('Boxplots of Age, Weight, Height, TIN.median', fontsize=14, fontweight='bold')
plt.ylabel('Value', fontsize=12)
plt.grid(axis='y', linewidth=1, alpha=0.7)  # Add horizontal grid lines for readability
plt.xticks(fontsize=11)  # Set font size for x-axis labels
plt.yticks(fontsize=11)  # Set font size for y-axis labels
plt.ylim(bottom=0)       # Force y-axis to start at 0

# Explanations for boxplot elements
explanation = '''\nBoxplot explanations:\n- The box shows the interquartile range (IQR), from Q1 (25th percentile) to Q3 (75th percentile).\n- The line inside the box is the median (Q2, 50th percentile).\n- The whiskers extend to the most extreme data points not considered outliers (typically 1.5*IQR from the box).\n- Outliers are shown as individual points beyond the whiskers.\n'''
print(explanation)

# Print boxplot statistics for each variable
for i, (label, data_entry) in enumerate(zip(labels, data_list)):
    q1 = np.percentile(data_entry, 25)
    q3 = np.percentile(data_entry, 75)
    iqr = q3 - q1
    median = np.median(data_entry)
    # Get the cutoff for this variable
    cutoff = outlier_cutoffs[label]
    lower_whisker = np.min(data_entry[data_entry >= q1 - cutoff * iqr])
    upper_whisker = np.max(data_entry[data_entry <= q3 + cutoff * iqr])
    outliers = data_entry[(data_entry < lower_whisker) | (data_entry > upper_whisker)]
    print(f"{label} (outlier cutoff: {cutoff}*IQR):")
    print(f"  Lower whisker: {lower_whisker}")
    print(f"  Upper whisker: {upper_whisker}")
    print(f"  Median: {median}")
    print(f"  Q(0.25): {q1}")
    print(f"  Q(0.75): {q3}")
    print(f"  Number of Outliers: {len(outliers)}\n")

# plt.show()
plt.close()

# Create a stacked bar chart for the data of VTFV, Diabetes and Hypertension, with “VTFV”, “Diabetes” and “Hypertension” on the x-axis and absolute frequency of “Yes” and “No” as stacked bar on the y-axis. 	




# --- Stacked Bar Chart for VTVF, Diabetes, Hypertension ---
categories = ['VTVF', 'Diabetes', 'Hypertension']
yes_counts = []
no_counts = []

# Extract categorical columns (don't convert to float)
for cat in categories:
    idx = data.dtype.names.index(cat)
    col_data = clean_arr[:, idx]
    
    # Count 'Yes' and 'No' values (note the quotes around the values!)
    yes_count = np.sum(col_data == '"Yes"')  # Changed from 'Yes' to '"Yes"'
    no_count = np.sum(col_data == '"No"')    # Changed from 'No' to '"No"'
    
    yes_counts.append(yes_count)
    no_counts.append(no_count)
    
    print(f"{cat}: Yes={yes_count}, No={no_count}")

x = np.arange(len(categories))
plt.figure(figsize=(7,5))
plt.bar(x, no_counts, label='No', color='#5DADE2', width=0.5)
plt.bar(x, yes_counts, bottom=no_counts, label='Yes', color='#F1948A', width=0.5)
plt.xticks(x, categories)
plt.xlim(-1, len(categories) + 0)
plt.ylabel('Absolute Frequency')
plt.title('Absolute Frequency of Yes/No for VTVF, Diabetes, Hypertension')
plt.legend()
plt.tight_layout()
# plt.show()
plt.close()

# --- Frequency Tables ---
print("\n=== Frequency Tables ===\n")

# Gender
gender_col = clean_arr[:, data.dtype.names.index('gender')]
print("Gender:")
for value in np.unique(gender_col):
    count = np.sum(gender_col == value)
    print(f"  {value}: {count}")

# Race
race_col = clean_arr[:, data.dtype.names.index('race')]
print("\nRace:")
for value in np.unique(race_col):
    count = np.sum(race_col == value)
    print(f"  {value}: {count}")

# Etiology
etiology_col = clean_arr[:, data.dtype.names.index('etiology')]
print("\nEtiology:")
for value in np.unique(etiology_col):
    count = np.sum(etiology_col == value)
    print(f"  {value}: {count}")



# --- Pie Charts for Gender, Race, and Etiology ---

# Extract data
gender_col = clean_arr[:, data.dtype.names.index('gender')]
race_col = clean_arr[:, data.dtype.names.index('race')]
etiology_col = clean_arr[:, data.dtype.names.index('etiology')]

# Count frequencies
gender_unique, gender_counts = np.unique(gender_col, return_counts=True)
race_unique, race_counts = np.unique(race_col, return_counts=True)
etiology_unique, etiology_counts = np.unique(etiology_col, return_counts=True)

# Define color palettes
gender_colors = ['#5DADE2', '#F1948A']
race_colors = ['#5DADE2', '#F39C12']
etiology_colors = ['#5DADE2', '#F39C12', '#52BE80']

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gender pie chart
axes[0].pie(gender_counts, labels=gender_unique, autopct='%1.1f%%', startangle=90,
            colors=gender_colors, textprops={'fontsize': 11})
axes[0].set_title('Gender Distribution', fontsize=13, fontweight='bold', pad=20)

# Race pie chart
axes[1].pie(race_counts, labels=race_unique, autopct='%1.1f%%', startangle=90,
            colors=race_colors, textprops={'fontsize': 11})
axes[1].set_title('Race Distribution', fontsize=13, fontweight='bold', pad=20)

# Etiology pie chart
axes[2].pie(etiology_counts, labels=etiology_unique, autopct='%1.1f%%', startangle=90,
            colors=etiology_colors, textprops={'fontsize': 11})
axes[2].set_title('Etiology Distribution', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()