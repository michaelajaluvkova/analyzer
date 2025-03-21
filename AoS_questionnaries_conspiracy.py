import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib
from adjustText import adjust_text
matplotlib.use('TkAgg')

def adjust_annotations(ax, x_data, y_data, labels, ukraine=None):
    annotations = []
    if ukraine == False:
        special_labels = {'P38': (5, -5), 'P9': (5, -5)}
        for i, label in enumerate(labels):
            if label in participants_to_highlight and label not in special_labels:
                annotations.append(
                    ax.text(x_data.iloc[i], y_data.iloc[i], label, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5), zorder=2))
        adjust_text(annotations, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
        for special_label, offset in special_labels.items():
            if special_label in labels.values:
                special_indices = labels[labels == special_label].index
                for idx in special_indices:
                    ax.annotate(special_label, (x_data.iloc[idx], y_data.iloc[idx]),
                                textcoords="offset points", xytext=offset,
                                ha='left', va='top', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                          facecolor="white", alpha=0.5),
                                arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
    else:
        special_labels = {'P38': (5, -5)}
        for i, label in enumerate(labels):
            if label in participants_to_highlight and label not in special_labels:
                annotations.append(
                    ax.text(x_data[i], y_data[i], label, fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                      facecolor="white", alpha=0.5), zorder=2))
        adjust_text(annotations, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))
        for special_label, offset in special_labels.items():
            if special_label in labels.values:
                special_indices = labels[labels == special_label].index
                for idx in special_indices:
                    ax.annotate(special_label, (x_data.iloc[idx], y_data.iloc[idx]),
                                textcoords="offset points", xytext=offset,
                                ha='left', va='top', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                                          facecolor="white", alpha=0.5),
                                arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5))

def plot_quadratic_regression(x, y, labels, hue=None, hue_order=None, palette=None, xlabel='', ylabel='', title='', filename=None,
                              ukraine=False, legend_title='Sociality', annotate=True, figsize=(14, 6)):
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(x=x, y=y, hue=hue, palette=palette,
                         style=hue, s=100, hue_order=hue_order,
                         style_order=hue_order, zorder=1)
    if annotate:
        adjust_annotations(ax, x, y, labels, ukraine=ukraine)
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    y_pred = p(x_sorted)
    plt.plot(x_sorted, y_pred, color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    if hue is not None:
        plt.legend(title=legend_title, bbox_to_anchor=(1.21, 1))
        plt.subplots_adjust(right=0.8)
    if filename:
        plt.savefig(filename, format='jpg', dpi=400)
    plt.show()
    y_fit = p(x)
    r_squared = r2_score(y, y_fit)
    n = len(y)
    k = 2
    rss = np.sum((y - y_fit) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    f_stat = (tss - rss) / (rss / (n - k - 1))
    p_val = stats.f.sf(f_stat, k, n - k - 1)
    print(f"{title}")
    print(f"The overall model was{' not' if p_val > 0.05 else ''} significant, F({k-1}, {n-k-1}) = {f_stat:.4f}, p = {p_val:.4f}, "
          f"with an R² value of {r_squared:.4f}, indicating that {r_squared * 100:.2f}% of the variance in {ylabel.lower()} was explained by the model.")
    print(f"{ylabel} = {z[0]:.4f} * ({xlabel})^2 + {z[1]:.4f} * ({xlabel}) + {z[2]:.4f}")
    return {"r_squared": r_squared, "f_stat": f_stat, "p_value": p_val, "coefficients": z}

# === DATA PREP ===
graph1_data = pd.read_csv('covid_data_study_1.csv', decimal=',')
data_data = pd.read_excel('data_2024.xlsx')
participants_to_highlight = ['P1', 'P2', 'P3', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P17', 'P18', 'P19', 'P20', 'P22', 'P30']
filtered_graph1_data = graph1_data[graph1_data['kod'].isin(participants_to_highlight)]
data_data['intensity_reversed'] = 1 - data_data['intensity'] #reversing variable

# setting palletes
palette_full = {0: 'red', 1: 'gray', 2: 'blue', 3: 'black', 4: 'green'}
graph1_data['Sociality'] = graph1_data['Sociality'].replace({1: 0, 2: 1, 3: 2, 4: 3})
data_data['sociality'] = data_data['sociality'].replace({4: 3})
sociality_labels = {0: 'low sociality', 1: 'medium sociality', 2: 'developed sociality', 3: 'undetermined'}
graph1_data['sociality_label'] = graph1_data['Sociality'].map(sociality_labels)
palette_full = {'low sociality': 'red', 'medium sociality': 'gray', 'developed sociality': 'blue', 'undetermined': 'black'}
legend_order = ['low sociality', 'medium sociality', 'developed sociality', 'undetermined']

# === GRAPH 1 ( Study 1)  ===
#print(graph1_data['Sociality'].unique())
x1 = pd.to_numeric(graph1_data['skepticism_novy'], errors='coerce')
y1 = pd.to_numeric(graph1_data['complexity'], errors='coerce')


sociality1 = graph1_data['sociality_label']
legend_order_1 = ['low sociality', 'medium sociality', 'developed sociality']

plt.figure(figsize=(12, 6))
ax = sns.scatterplot(x=x1, y=y1, hue=sociality1, palette=palette_full, style=sociality1, s=100, hue_order=legend_order_1, style_order=legend_order_1, zorder=1)

# quadratic regression
z1 = np.polyfit(x1, y1, 2)
p1 = np.poly1d(z1)
x_smooth = np.linspace(np.min(x1), np.max(x1), 300)  # smoothening
y_smooth = p1(x_smooth)

plt.plot(x_smooth, y_smooth, color='black') #se zahlazenim

data_data['kod'] = data_data['kod'].reset_index(drop=True)

for i in range(len(filtered_graph1_data)):
    participant_code = filtered_graph1_data.iloc[i]['kod']
    x_pos = pd.to_numeric(filtered_graph1_data.iloc[i]['skepticism_novy'])
    y_pos = pd.to_numeric(filtered_graph1_data.iloc[i]['complexity'])

    # Chunk of overlapping participants - customization offsets
    if participant_code == 'P8':
        plt.annotate(participant_code, (x_pos, y_pos - 0.015),
                     textcoords="offset points", xytext=(0, -15),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.2", edgecolor="red",
                               facecolor="white", alpha=0.5))

    if participant_code == 'P13':
        plt.annotate(participant_code, (x_pos, y_pos - 0.005),
                     textcoords="offset points", xytext=(0, -10),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="red",
                               facecolor="white", alpha=0.5))

    if participant_code == 'P19':
        plt.annotate(participant_code, (x_pos, y_pos),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                               facecolor="white", alpha=0.5))

    if participant_code == 'P3':
        plt.annotate(participant_code, (x_pos, y_pos), textcoords="offset points", xytext=(0, 0),
                     ha='center', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                               facecolor="white", alpha=0.5, zorder=2))

    if participant_code not in ['P3', 'P8', 'P13', 'P19']:
        plt.annotate(participant_code, (x_pos, y_pos), textcoords="offset points", xytext=(0, 0),
                     ha='left', va='top', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.15", edgecolor="red",
                               facecolor="white", alpha=0.5, zorder=2))

plt.xlabel('Conspiracy mentality questionnaire')
plt.gcf().text(0.5, 0.03, '(CMQ)', ha='center', va='top', fontsize=10)
plt.ylabel('Complexity')
plt.legend(title='Sociality')
plt.grid(True)
plt.savefig('study1.jpg', format='jpg', dpi=400)
plt.show()


# === GRAPHS STUDY 2 ===
plot_quadratic_regression(
    x=data_data['CMQ'],
    y=data_data['intensity_reversed'],
    labels=data_data['kod'],
    hue=data_data['sociality_label'], hue_order=legend_order, palette=palette_full,
    xlabel='Conspiracy mentality questionnaire (CMQ)', ylabel='Complexity', title='',
    filename='study2_CMQ.jpg')

plot_quadratic_regression(
    x=data_data['conspiracy_general'],
    y=data_data['intensity_reversed'],
    labels=data_data['kod'],
    hue=data_data['sociality_label'], hue_order=legend_order, palette=palette_full,
    xlabel='General conspiracy beliefs (GCB)', ylabel='Complexity', title='',
    filename='study2_GCB.jpg')

plot_quadratic_regression(
    x=data_data['conspiracy_UA'],
    y=data_data['intensity_reversed'],
    labels=data_data['kod'],
    hue=data_data['sociality_label'], hue_order=legend_order, palette=palette_full,
    xlabel='Ukraine conspiracy beliefs (UCB)', ylabel='Complexity', title='',
    filename='study2_UCB.jpg', ukraine=True)

