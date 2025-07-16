### Import necessary libraries
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import pathlib
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import itertools
from scipy.stats import brunnermunzel
from skimage.feature import local_binary_pattern
from scipy.fft import fft2
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# --- Functions ------------

### figure 1 show images and 
def show_images(input_image, param_columns, save=False):
    """
    figure 1 show images and parameters    
    """
    for i in range(len(param_columns)):
        plt.figure(figsize=(8, 6))
        plt.imshow(input_image[:, :, i], cmap='viridis', aspect='auto')
        plt.colorbar(label=param_columns[i])
        plt.title(param_columns[i])
        plt.xticks([0, 51, 102, 153, 204], 
				  ["0", "0.2", "0.4", "0.6", "0.8"])
        plt.yticks([16, 67, 118, 169, 220],
				  ["0.8", "0.6", "0.4", "0.2", "0"])
        if save==True:
            plt.savefig(f'figure_{i+1}_{param_columns[i]}.png', bbox_inches='tight')
        elif save==False:
	        plt.show()

def preprocess(input_images):
    result_images = np.zeros_like(input_images)
    for i in range(input_images.shape[2]):
        gaussian_image = gaussian(input_images[:, :, i], sigma=2)
    # min-max normalization
        min_val = np.min(gaussian_image)
        max_val = np.max(gaussian_image)
    # perform min-max normalization
        normalized_data = (gaussian_image - min_val) / (max_val - min_val)
        result_images[:, :, i] = normalized_data
    return result_images


def kmeans_clustering(input_images, number = 3, random_state=0, n_init=100,name='name', title='title', save=False, inertia_centroids=False):
    kmeans = KMeans(n_clusters=number, random_state=random_state, n_init=n_init, max_iter=300, tol=1e-04)
    if input_images.ndim == 3:
        (x, y, z) = input_images.shape
        result = kmeans.fit_predict(input_images.reshape(x*y, z))
    elif input_images.ndim == 2:
        result = kmeans.fit_predict(input_images)
        (x, y) = input_images.shape
    else:
        print('please check input images shape')
    result_reshape = result.reshape(x, y)
    if save == True:
        plt.figure()
        plt.imshow(result_reshape, cmap='jet')
        plt.axis('off')
        plt.title(title, fontsize=18)
        plt.savefig(name, bbox_inches='tight', transparent=True)
        plt.clf()
        plt.close()
    if inertia_centroids == True:
        centroids_ = kmeans.cluster_centers_
        inertia_ = kmeans.inertia_
        return result_reshape, inertia_, centroids_
    else:
        return result_reshape

def evaluate_clustering(ground_truth, unsupervised_labels):
    """
    Function to evaluate the clustering results by comparing the ground truth labels with the unsupervised labels.

    Parameters:
    ground_truth (np.array): the ground truth labels (1D array)
    unsupervised_labels (np.array): the labels obtained from unsupervised learning (1D array)

    Returns:
    float: the accuracy of the clustering results
    """
    # create a confusion matrix
    conf_matrix = confusion_matrix(ground_truth, unsupervised_labels, normalize='true')

    # apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # mapping the labels
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # apply the mapping to the unsupervised labels
    mapped_labels = np.array([label_mapping[label] for label in unsupervised_labels])

    return conf_matrix, mapped_labels


def calculate_accuracy(annotation_labels, cluster_labels, n_clusters=3):
    """
    Function to calculate the percentage of agreement between annotation results and K-means clustering results and plot the results.

    :param annotation_labels: The ground truth labels (1D array)
    :param data: The data used for clustering (2D array)
    :param n_clusters: The number of clusters in K-means clustering
    :return: accuracy_dict: A dictionary containing the accuracy for each cluster
    """
    # Calculate the accuracy for each cluster
    accuracy_dict = {}
    for i in range(n_clusters):
        mask = (annotation_labels== i)
        
       
        
        cluster_accuracy = accuracy_score(annotation_labels[mask], cluster_labels[mask])
        accuracy_dict[f'Cluster {i}'] = cluster_accuracy
        
    
    return accuracy_dict
    
def glid_search(input_data, annotation_result, path='path', n_cluster=3, n_init=100, title='title', name='name', properties=['Adhesion energy', 'Adhesion force', 'Deformation', 'Energy dissipation', 'Modulus', 'Stiffness approach', 'Stiffness retract', 'Probe current'],
                save=True,save_labels=False):
    '''
    I want to make threshold for labeling.
    data (numpy.ndarray): Image data.
    title (str): Title of the plot.
    name (str): Name of the plot.
    properties (list): List of the AFM properties.
    save (bool): If True, save the plot.
    save_pd (bool): If True, save the pandas data frame as excel file.
    
    Generate all possible combinations of properties and run glid_search on each.
    Perform clustering on the results.
    '''
    Phase = ['Matrix', 'Shell', 'Core']
    current_directory = os.getcwd()
    num_properties = len(properties)
    indices = list(range(num_properties))
    directory_path = os.path.join(current_directory, 'output')
    os.makedirs(directory_path, exist_ok=True)
    os.chdir(directory_path)
    records = []
    count = 1
    for r in tqdm(range(1, num_properties + 1)):
        for combination in itertools.combinations(indices, r):
            selected_properties = [properties[i] for i in combination]

            record = {"Number": count,"the number of parameters": r}
            for j, prop in enumerate(properties):
                record[prop] = 1 if j in combination else 0
            
            # Extract the corresponding data for the selected properties
            selected_data = input_data[:, :, list(combination)]
            kmeans_result = kmeans_clustering(selected_data, number=3, n_init=n_init, random_state=0, save=False,  title=title+'_'+str(selected_properties), name=name+'_'+str(selected_properties)+'_'+str(r))   
            k_shape = kmeans_result.reshape(-1)
            new_labels = evaluate_clustering(annotation_result.reshape(-1), k_shape)[1]
            plt.figure()
            plt.imshow(new_labels.reshape(kmeans_result.shape), cmap='jet')
            plt.axis('off')
            if title == save:
                plt.title('result ' +title+'_'+str(selected_properties), fontsize=18, fontname='Arial')
            if save == True:
                plt.savefig('result_'+name+'_'+str(selected_properties)+'_'+str(r)+'.png', bbox_inches='tight')
            plt.clf()
            plt.close()
            accuracy = calculate_accuracy(annotation_result.reshape(-1), new_labels, n_clusters=3)
            for k in range(3):
                
                record[Phase[k]] = accuracy.get(f"Cluster {k}")
            record["Balanced"] = np.mean(list(accuracy.values()))
            records.append(record)
            count += 1
            if save_labels == True:
                np.save(f"labels_{name}_{str(selected_properties)}.npy", new_labels.reshape(kmeans_result.shape))
    Df = pd.DataFrame(records)
    if save:
        Df.to_excel(f'glid_search_summary_{name}.xlsx', index=False)
    return Df
def box_plot(input_data, title, x_label= '',rotation=0, save_fig=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=input_data, color='skyblue')
    #plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Balanced accuracy')
    plt.xticks(rotation=rotation)
    plt.ylim(0, 1)  # 縦軸の範囲を設定
    plt.yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
      # 縦軸の刻み目を設定
    plt.tight_layout()
    if save_fig:
       plt.savefig(f'box_plot_{title}.png', bbox_inches='tight',transparent=True, pad_inches=0.3)
    plt.clf()
    plt.close()
    return 

def BrunerMunzel_test(data, features, target='Balanced accuracy', method='greater', witch_m = 'modality'):
    t_list = []
    p_list = []
    if witch_m == 'modality':
        for i in range(len(features)):
            target_data = data[data[features[i]] == 1][target]
            non_target_data = data[data[features[i]] == 0][target]
            t, p = brunnermunzel(target_data, non_target_data, alternative=method, distribution='t')
            t_list.append(t)
            p_list.append(p)
    elif witch_m == 'Nmode':
        for i in range(1, 8):
            target_data = data[data['Nmode'] == i][target]
            non_target_data = data[data['Nmode'] == 1][target]
            t, p = brunnermunzel(target_data, non_target_data, alternative=method, distribution='t')
            t_list.append(t)
            p_list.append(p)
    return t_list, p_list

def calculate_silhouette(img_list, n_clusters=3):
    """
    img_list: (list of np.ndarray) 2D画像たち (すべて同じ形)
    n_clusters: (int) k-meansで分けるクラスター数
    """
    stacked = np.stack(img_list, axis=-1)
    x, y, z = stacked.shape
    data = stacked.reshape(-1, z)  # (pixel数, z次元)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=100)
    labels = kmeans.fit_predict(data)
    
    score = silhouette_score(data, labels)
    return score

def fft_similarity(img_list, img_target):
    if  isinstance (img_list, np.ndarray):
        f1 = np.abs(fft2(img_list))
        f2 = np.abs(fft2(img_target))
        f1_flat = f1.ravel()
        f2_flat = f2.ravel()
        return cosine(f1_flat, f2_flat)
    elif isinstance(img_list, list):
        fft_list = [np.abs(fft2(img)).flatten() for img in img_list]
        fft_target = np.abs(fft2(img_target)).flatten()
        combined = np.concatenate(fft_list)
        target_dup = np.tile(fft_target, len(img_list))
        return cosine(combined, target_dup)

# --- Main ------------ 
### input data
file_path = (r'xxx')
os.chdir(file_path)

input_image = pd.read_csv('multi-parameter_data_with_groundtruth.csv')
# check the x and y coordinates and parameter
x_unique = np.sort(input_image['x'].unique())
y_unique = np.sort(input_image['y'].unique())
param_columns = input_image.columns[2:-1]  # exclude 'Ground truth' column

# check the size
x_size = len(x_unique)
y_size = len(y_unique)
z_size = len(param_columns)

# treatment of the image array
image_array = np.zeros((y_size, x_size, z_size))
x_map = {val: idx for idx, val in enumerate(x_unique)}
y_map = {val: idx for idx, val in enumerate(y_unique)}

# set values in the array
for _, row in input_image.iterrows():
    xi = x_map[row['x']]
    yi = y_map[row['y']]
    image_array[yi, xi, :] = row[param_columns].values

ground_truth = input_image['Ground truth'].values.reshape(y_size, x_size)
preprocessed_image_array = preprocess(image_array)
# grid search
grid_result = glid_search(preprocessed_image_array, ground_truth, path=r"C:\Users\user\Box\Oshida_1stPaper\ACS Joural of chemical information and methods\output", n_init=100, title='K-means clustering', name='kmeans_clustering', properties=param_columns, save=True, save_labels=True)

# figure 2 kmeans clustering
os.makedirs(file_path + r'\output\figure2', exist_ok=True)
os.chdir(file_path + r'\output\figure2')
for i in range(len(image_array[0, 0, :])):
    input_image_fig2 = np.stack([preprocessed_image_array[:, :, i], preprocessed_image_array[:,:,i]], axis=2)
    k_result = kmeans_clustering(input_image_fig2, number=3, n_init=100, random_state=0, save=True, title='', name=f'kmeans_clustering_{param_columns[i]}')
 
# figure 3 show the boxplot for the number of parameters
os.makedirs(file_path + r'\output\figure3', exist_ok=True)
os.chdir(file_path + r'\output\figure3')
input_data = []
for j in range(0, len(param_columns)): 
        
    m_data = grid_result[grid_result['the number of parameters'] == j]
    input_data.append(m_data['Balanced'].values)
input_pd = pd.DataFrame(input_data).T
input_pd.columns = [k for k in range(1, 9)]
box_plot(input_pd, title='figure3', save_fig=True)

# figure 4 show the boxplot for the effect of each parameter
os.makedirs(file_path + r'\output\figure4', exist_ok=True)
os.chdir(file_path + r'\output\figure4')

plot_data = []
for parameter in param_columns:
    with_parameter = grid_result[grid_result[parameter] == 1]["Balanced"]
    without_parameter = grid_result[grid_result[parameter] == 0]["Balanced"]
    plot_data.append(pd.DataFrame({
        "Balanced accuracy": np.concatenate([with_parameter, without_parameter]),
        "Parameter": [parameter] * (len(with_parameter) + len(without_parameter)),
        "Group": ["With"] * len(with_parameter) + ["Without"] * len(without_parameter)
    }))
plot_df = pd.concat(plot_data, axis=0)
plt.figure(figsize=(10, 6))
sns.boxplot(x="Parameter", y="Balanced accuracy", hue="Group", data=plot_df, palette=["#1f77b4", "#ff7f0e"])
plt.xticks(rotation=90, ha="right")
plt.ylabel("Balanced accuracy")
plt.xlabel("")
plt.ylim(0, 1.05)
plt.legend(title="", loc="lower left")
plt.savefig("boxplot.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
# calculate the Brunner-Munzel test for each parameter
p_result, t_result = BrunerMunzel_test(grid_result, param_columns, target='Balanced', method='greater', witch_m='modality')
statistic_df = pd.DataFrame({
	"Parameter": param_columns,
	"T-statistic": t_result,
	"P-value": p_result
})
statistic_df.to_excel("brunner_munzel_results.xlsx", index=False)

os.makedirs(file_path + r'\output\figure6', exist_ok=True)
os.chdir(file_path + r'\output\figure6')
# calculate the silhouette score and cosine distance
    # figure 6a 
silhouette_score_fig6a = []
for i in range(len(preprocessed_image_array[0, 0, :])):
	silhouette_score_a = calculate_silhouette([preprocessed_image_array[:, :, i],preprocessed_image_array[:, :, i]], n_clusters=3)
	silhouette_score_fig6a.append(silhouette_score_a)
df_silhouette_score_fig6a = pd.DataFrame({
	"Parameter": param_columns,
	"Silhouette Score": silhouette_score_fig6a
})
df_silhouette_score_fig6a.to_excel("silhouette_scores_fig6a.xlsx", index=False)
	# figure 6b
silhouette_score_fig6b = []
cosine_distance_fig6b = []
image_array_6b = preprocessed_image_array[:,:,[0,1,3,4,5,6,7]]
exclude_params = ["Deformation"]
# param_columns から除外
filtered_params = [p for p in param_columns if p not in exclude_params]

for i in range(len(image_array_6b[0, 0, :])):
    input_data_6b = np.stack([image_array_6b[:, :, i], preprocessed_image_array[:, :, 2]], axis=2)
    silhouette_score_b = calculate_silhouette(input_data_6b, n_clusters=3)
    cosine_distance_b = fft_similarity(image_array_6b[:, :, i], preprocessed_image_array[:, :, 2])
    silhouette_score_fig6b.append(silhouette_score_b)
    cosine_distance_fig6b.append(cosine_distance_b)
df_silhouette_score_fig6b = pd.DataFrame({
	"Parameter": filtered_params,
	"Silhouette Score": silhouette_score_fig6b,
	"Cosine Distance": cosine_distance_fig6b
})
df_silhouette_score_fig6b.to_excel("silhouette_scores_fig6b.xlsx", index=False)
	# figure 6c
silhouette_score_fig6c = []
cosine_distance_fig6c = []
image_array_6c = preprocessed_image_array[:,:,[0,1,3,4,5,6]]
exclude_params = ["Deformation", "Probe current"]
combined_indices = [2,7]
filtered_params = [p for p in param_columns if p not in exclude_params]
input_imgs_fig6c = [preprocessed_image_array[:,:,i] for i in combined_indices]
for i in range(len(image_array_6c[0, 0, :])):
    input_data_6c = np.stack([image_array_6c[:, :, i], preprocessed_image_array[:, :, 2],preprocessed_image_array[:, :, 7]], axis=2)
    silhouette_score_c = calculate_silhouette(input_data_6c, n_clusters=3)
    cosine_distance_c = fft_similarity(input_imgs_fig6c, image_array_6c[:, :, i])
    silhouette_score_fig6c.append(silhouette_score_c)
    cosine_distance_fig6c.append(cosine_distance_c)
df_silhouette_score_fig6c = pd.DataFrame({
	"Parameter": filtered_params,
	"Silhouette Score": silhouette_score_fig6c,
	"Cosine Distance": cosine_distance_fig6c
})
df_silhouette_score_fig6c.to_excel("silhouette_scores_fig6c.xlsx", index=False)
# figure 6d
silhouette_score_fig6d = []
cosine_distance_6d = []
combined_indices = [2,4,7]
image_array_6d = preprocessed_image_array[:,:,[0,1,3,5,6]]
exclude_params = ["Deformation","Modulus", "Probe current"]
filtered_params = [p for p in param_columns if p not in exclude_params]
input_imgs_fig6d = [preprocessed_image_array[:,:,i] for i in combined_indices]
for i in range(len(image_array_6d[0, 0, :])):
    input_data_6d = np.stack([image_array_6d[:, :, i], preprocessed_image_array[:, :, 2],preprocessed_image_array[:, :, 4],preprocessed_image_array[:, :, 7]], axis=2)
    silhouette_score_d = calculate_silhouette(input_data_6d, n_clusters=3)
    cosine_distance_d = fft_similarity(input_imgs_fig6d, image_array_6d[:, :, i])
    silhouette_score_fig6d.append(silhouette_score_d)
    cosine_distance_6d.append(cosine_distance_d)
df_silhouette_score_fig6d = pd.DataFrame({
	"Parameter": filtered_params,
	"Silhouette Score": silhouette_score_fig6d,
	"Cosine Distance": cosine_distance_6d
})
df_silhouette_score_fig6d.to_excel("silhouette_scores_fig6d.xlsx", index=False)
