import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib
from adjustText import adjust_text
matplotlib.use('TkAgg')


# Function to automatically adjust annotations to avoid overlap
def adjust_annotations(ax, x_data, y_data, labels):

    annotations = []
    special_labels = {
        'P38': (5, -5),
        'P9': (5, -5)}
    for i, label in enumerate(labels):
        if label in participants_to_highlight:
            if label not in special_labels:
                annotations.append(
                    ax.text(x_data[i], y_data[i], label, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5), zorder=2))

    adjust_text(annotations, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
    for special_label, offset in special_labels.items():
        if special_label in labels.values:
            special_indices = labels[labels == special_label].index
            for idx in special_indices:
                x_special = x_data.iloc[idx]
                y_special = y_data.iloc[idx]
                ax.annotate(special_label, (x_special, y_special),
                            textcoords="offset points", xytext=offset,  # Custom offset
                            ha='left', va='top', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5),
                            arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

def adjust_annotations_ukraine(ax, x_data, y_data, labels):
    annotations = []
    special_labels = {
        'P38': (5, -5)}
    for i, label in enumerate(labels):
        if label in participants_to_highlight:
            if label not in special_labels:
                annotations.append(
                    ax.text(x_data[i], y_data[i], label, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5), zorder=2)
                )
    adjust_text(annotations, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
    for special_label, offset in special_labels.items():
        if special_label in labels.values:
            special_indices = labels[labels == special_label].index
            for idx in special_indices:
                x_special = x_data.iloc[idx]
                y_special = y_data.iloc[idx]
                ax.annotate(special_label, (x_special, y_special),
                            textcoords="offset points", xytext=offset,  # Custom offset
                            ha='left', va='top', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5),
                            arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))


# Load data from Excel files
graph1_data = pd.read_excel('')
data_data = pd.read_excel('')
participants_to_label = ['P1', 'P2', 'P3', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P17', 'P18', 'P19', 'P20', 'P22', 'P23']
filtered_graph1_data = graph1_data[graph1_data['kod'].isin(participants_to_label)]

# Reverse intensity variable
data_data['intensity_reversed'] = 1 - data_data['intensity']
graph1_data['skepticism'] = (graph1_data['skepticism'] - 2)

# Set colors for sociality variable
palette_full = {0: 'red', 1: 'gray', 2: 'blue', 3: 'black', 4: 'green'}
graph1_data['Sociality'] = graph1_data['Sociality'].replace({1: 0, 2: 1, 3: 2, 4: 3})
data_data['sociality'] = data_data['sociality'].replace({4: 3})
sociality_labels = {0: 'low sociality', 1: 'medium sociality', 2: 'developed sociality', 3: 'undetermined'}
graph1_data['sociality_label'] = graph1_data['Sociality'].map(sociality_labels)
palette_full = {
    'low sociality': 'red',
    'medium sociality': 'gray',
    'developed sociality': 'blue',
    'undetermined': 'black'
}
legend_order = ['low sociality', 'medium sociality', 'developed sociality', 'undetermined']

# === GRAPH 1 ( Study 1)  ===
print(graph1_data['Sociality'].unique())
x1 = graph1_data['skepticism']
y1 = graph1_data['complexity']
sociality1 = graph1_data['sociality_label']
legend_order_1 = ['low sociality', 'medium sociality', 'developed sociality']

plt.figure(figsize=(12, 6))
ax = sns.scatterplot(x=x1, y=y1, hue=sociality1, palette=palette_full, style=sociality1, s=100, hue_order=legend_order_1, style_order=legend_order_1, zorder=1)

# Fit and plot quadratic regression line for Graph 1
z1 = np.polyfit(x1, y1, 2)
p1 = np.poly1d(z1)
x_smooth = np.linspace(np.min(x1), np.max(x1), 300)
y_smooth = p1(x_smooth)

plt.plot(x_smooth, y_smooth, color='black') #smoother

for i in range(len(filtered_graph1_data)):
    participant_code = filtered_graph1_data.iloc[i]['kod']
    x_pos = filtered_graph1_data.iloc[i]['skepticism']
    y_pos = filtered_graph1_data.iloc[i]['complexity']

    # Custom offset for P8 to position it precisely between P3 and P13
    if participant_code == 'P8':
        plt.annotate(participant_code, (x_pos + 0.5, y_pos),
                     textcoords="offset points", xytext=(0, -7),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                               facecolor="white", alpha=0.5))
    if participant_code == 'P13':
        plt.annotate(participant_code, (x_pos, y_pos - 0.005),
                     textcoords="offset points", xytext=(0, -10),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                               facecolor="white", alpha=0.5))
    if participant_code == 'P19':
        plt.annotate(participant_code, (x_pos, y_pos - 0.005),
                     textcoords="offset points", xytext=(0, 15),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                               facecolor="white", alpha=0.5))
    if participant_code not in ['P8', 'P13', 'P19']:
        plt.annotate(participant_code, (x_pos, y_pos), textcoords="offset points", xytext=(10, 10),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                               facecolor="white", alpha=0.5))

# Remove title
plt.xlabel('Conspiracy mentality questionnaire')
plt.gcf().text(0.5, 0.03, '(CMQ)', ha='center', va='top', fontsize=10)
plt.ylabel('Complexity')
plt.legend(title='Sociality')
plt.grid(True)
plt.savefig('study1.jpg', format='jpg', dpi=400)
plt.show()


##### Start of Graphs for Study 2
participants_to_highlight = ['P1', 'P5', 'P6', 'P9', 'P10', 'P11', 'P12', 'P15', 'P16', 'P17', 'P18', 'P20', 'P22', 'P23', 'P26', 'P31', 'P32', 'P51', 'P52']
sociality_labels = {0: 'low sociality', 1: 'medium sociality', 2: 'developed sociality', 3: 'undetermined'}
data_data['sociality_label'] = data_data['sociality'].map(sociality_labels)

palette_full = {'low sociality': 'red', 'medium sociality': 'gray', 'developed sociality': 'blue', 'undetermined': 'black'}

# === GRAPH CMQ (Conspiracy mentality questinaries)
data_data['sociality'] = data_data['sociality'].replace({4: 3})
x2 = data_data['CMQ']
y2 = data_data['intensity_reversed']
sociality2 = data_data['sociality_label']

plt.figure(figsize=(14, 6))
ax = sns.scatterplot(x=x2, y=y2, hue=sociality2, palette=palette_full, style=sociality2, s=100, hue_order=legend_order, style_order=legend_order, zorder=1)

adjust_annotations(ax, x2, y2, data_data['kod'])

z2 = np.polyfit(x2, y2, 2)
p2 = np.poly1d(z2)
plt.plot(np.sort(x2), p2(np.sort(x2)), color='black')

plt.xlabel('Conspiracy mentality questionnaire (CMQ)')
#plt.gcf().text(0.5, 0.03, '(CMQ)', ha='left', va='top', fontsize=10)
plt.ylabel('Complexity')
#plt.legend(title='Sociality')
plt.legend(
    title='Sociality',
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    borderaxespad=0
)

plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('study2_CMQ.jpg', format='jpg', dpi=400)
plt.show()

# === GRAPH CONSPIRACY GENERAL ===
x3 = data_data['conspiracy_general']
y3 = data_data['intensity_reversed']
sociality3 = data_data['sociality_label']

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x=x3, y=y3, hue=sociality3, palette=palette_full, style=sociality3, s=100, hue_order=legend_order, style_order=legend_order, zorder=1)

adjust_annotations(ax, x3, y3, data_data['kod'])


# Fit and plot quadratic regression line for Graph 3
z3 = np.polyfit(x3, y3, 2)
p3 = np.poly1d(z3)
plt.plot(np.sort(x3), p3(np.sort(x3)), color='black')

# Remove title
plt.xlabel('General conspiracy beliefs')
plt.gcf().text(0.5, 0.03, '(GCB)', ha='center', va='top', fontsize=10)
plt.ylabel('Complexity')
plt.legend(title='Sociality')
plt.grid(True)
plt.savefig('study2_GCB.jpg', format='jpg', dpi=400)
plt.show()

# === Model Statistics for Graph 1 ===
y1_pred = p1(x1)
r_squared_1 = r2_score(y1, y1_pred)
n = len(y1)
k = 2
rss = np.sum((y1 - y1_pred) ** 2)
tss = np.sum((y1 - np.mean(y1)) ** 2)
f_statistic = (tss - rss) / (rss / (n - k - 1))
p_value = stats.f.sf(f_statistic, k, n - k - 1)

print("First graph COVID study")
print(f"The overall model was not significant, F({k-1}, {n-k-1}) = {f_statistic:.4f}, p = {p_value:.4f}, "
      f"with an R² value of {r_squared_1:.4f}, indicating that {r_squared_1 * 100:.2f}% of the variance in complexity was explained by the model.")
print(f"Complexity = {z1[0]:.4f} * (Conspiracy mentality)^2 + {z1[1]:.4f} * (Conspiracy mentality) + {z1[2]:.4f}")

# === Model Statistics for Graph 2 ===
y2_pred = p2(x2)
r_squared_2 = r2_score(y2, y2_pred)
n2 = len(y2)
k2 = 2
rss2 = np.sum((y2 - y2_pred) ** 2)
tss2 = np.sum((y2 - np.mean(y2)) ** 2)
f_statistic2 = (tss2 - rss2) / (rss2 / (n2 - k2 - 1))
p_value2 = stats.f.sf(f_statistic2, k2, n2 - k2 - 1)

print("2nd graph CMQ")
print(f"The overall model was not significant, F({k2-1}, {n2-k2-1}) = {f_statistic2:.4f}, p = {p_value2:.4f}, "
      f"with an R² value of {r_squared_2:.4f}, indicating that {r_squared_2 * 100:.2f}% of the variance in intensity was explained by the model.")
print(f"Complexity = {z2[0]:.4f} * (CMQ)^2 + {z2[1]:.4f} * (CMQ) + {z2[2]:.4f}")

# === Model Statistics for Graph 3 ===
y3_pred = p3(x3)

r_squared_3 = r2_score(y3, y3_pred)

# Calculate F-statistic and p-value for Graph 3
n3 = len(y3)
k3 = 2
rss3 = np.sum((y3 - y3_pred) ** 2)
tss3 = np.sum((y3 - np.mean(y3)) ** 2)
f_statistic3 = (tss3 - rss3) / (rss3 / (n3 - k3 - 1))
p_value3 = stats.f.sf(f_statistic3, k3, n3 - k3 - 1)

print("Graph 3 General conspiracy beliefs")
print(f"The overall model was not significant, F({k3-1}, {n3-k3-1}) = {f_statistic3:.4f}, p = {p_value3:.4f}, "
      f"with an R² value of {r_squared_3:.4f}, indicating that {r_squared_3 * 100:.2f}% of the variance in intensity was explained by the model.")
print(f"Complexity = {z3[0]:.4f} * (Conspiracy general)^2 + {z3[1]:.4f} * (Conspiracy general) + {z3[2]:.4f}")


# === GRAPH UKRAINE CONSPIRACY ===
x5 = data_data['conspiracy_UA']
y5 = data_data['intensity_reversed']
sociality4 = data_data['sociality_label']

plt.figure(figsize=(14, 6))
ax = sns.scatterplot(x=x5, y=y5, hue=sociality4, palette=palette_full, style=sociality4, s=100, hue_order=legend_order, style_order=legend_order, zorder=1)
adjust_annotations_ukraine(ax, x5, y5, data_data['kod'])

z5 = np.polyfit(x5, y5, 2)
p5 = np.poly1d(z5)
plt.plot(np.sort(x5), p3(np.sort(x5)), color='black')
plt.xlabel('Ukraine conspiracy beliefs')
plt.gcf().text(0.5, 0.03, '(UCB)', ha='center', va='top', fontsize=10)
plt.ylabel('Complexity')
plt.legend(
    title='Sociality',
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    borderaxespad=0)
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('study2_UCB.jpg', format='jpg', dpi=400)
plt.show()

z_UA = np.polyfit(x5, y5, 2)
p_UA = np.poly1d(z5)

y_UA_pred = p_UA(x5)
r_squared_UA = r2_score(y5, y_UA_pred)

n_UA = len(y5)
k_UA = 2
rss_UA = np.sum((y5 - y_UA_pred) ** 2)
tss_UA = np.sum((y5 - np.mean(y5)) ** 2)
f_statistic_UA = (tss_UA - rss_UA) / (rss_UA / (n_UA - k_UA - 1))
p_value_UA = stats.f.sf(f_statistic_UA, k_UA, n_UA - k_UA - 1)

print("4th Graph Ukraine-war conspiracy beliefs")

print(f"The overall model was not significant, F({k_UA-1}, {n_UA-k_UA-1}) = {f_statistic_UA:.4f}, p = {p_value_UA:.4f}, "
      f"with an R² value of {r_squared_UA:.4f}, indicating that {r_squared_UA * 100:.2f}% of the variance in complexity was explained by the model.")
print(f"Complexity = {z_UA[0]:.4f} * (Conspiracy Ukraine)^2 + {z_UA[1]:.4f} * (Conspiracy Ukraine) + {z_UA[2]:.4f}")
