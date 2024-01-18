#%%
from dataloader import load_data
from models import *
import torch.optim as optim
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
from hook_register import *
from utils import find_expressed_genes
# data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv"
# batch_size = 8
# max_epoch = 250
# feature_transform = False
# eval_interval = 1
# atention_pooling_flag = False
# outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression/saved_models"
# gene_space_dim = 3
#%%
data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/All_countings/training_data_17_tumors_31_classes.csv"
batch_size = 4
max_epoch = 50
feature_transform = False
eval_interval = 2
atention_pooling_flag = True
outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression/17_tumors_31_classes_saved_models"
gene_space_dim = 3
# LOSS_SELECT = 'CE' # 'CE' or 'NLL'
# WEIGHT_LOSS_FLAG = True
# MULTI_GPU_FLAG = True
# pre_trained = True
lr=0.001

gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader = load_data(file_path=data_dir, batch_size=batch_size)

class_num = len(number_to_label.keys())
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# best_val_acc = 0
# best_val_idx = 0
# confusion_matrix_all_best = []
# for model_idx in range(50):
model = PointNetCls(gene_idx_dim = 2, 
                gene_space_num = gene_space_dim, 
                class_num=class_num, 
                feature_transform=feature_transform, 
                atention_pooling_flag = atention_pooling_flag)
if device == torch.device("cpu"):
    model_state_dict = torch.load(outf+f"/cls_model_geneSpaceD_3_transfeat_False_attenpool_True_pretrain_best.pth", map_location=torch.device('cpu'))
else:
    model_state_dict = torch.load(outf+f"/cls_model_geneSpaceD_3_transfeat_False_attenpool_True_pretrain_best.pth")

# model.load_state_dict(model_state_dict)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    name = k[7:]  # remove `module.` prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(device)
#%%

total_correct = 0
total_testset = 0
confusion_matrix_all = np.zeros((class_num, class_num))

for i,data in enumerate(test_loader, 0):
    features1_count, features2_gene_idx, labels = data
    features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
    features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
    with torch.no_grad():
        model = model.eval()
        pred, _, _ = model(features1_count,features2_gene_idx)
        pred_choice = pred.data.max(1)[1]
        correct = torch.sum(pred_choice == labels)
        total_correct += correct.item()
        total_testset += features1_count.shape[0]

        pred_labels_np = pred_choice.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for idx_batch in range(labels_np.shape[0]):
            label_i = labels_np[idx_batch]
            pred_i = pred_labels_np[idx_batch]
            confusion_matrix_all[label_i,pred_i] += 1
    # print("accuracy {}".format(correct.item()/float(batch_size)))
print("final accuracy {}".format(total_correct / float(total_testset)))
print(confusion_matrix_all)

# # Move the model to CPU and delete it
# model.to('cpu')
# del model
# torch.cuda.empty_cache()  # Clear CUDA cache
result_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression/results'
np.save(result_dir+f"/confusion_matrix.npy", confusion_matrix_all)



# %%
# hook register
activation={}
layer_name_list = ['gstn', ['feat','atention_pooling']]
Hook_register(model, layer_name_list, activation)
for i,data in enumerate(test_loader, 0):
    features1_count, features2_gene_idx, labels = data
    features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
    features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
    model = model.eval()
    pred, _, _ = model(features1_count,features2_gene_idx)
    break
print(activation)
#%% deal with the gene token space first.
gene_token_space = activation['gstn'].cpu().numpy()[0,:,:]
gene_token_space = gene_token_space.T
np.save(result_dir+f"/gene_token_space.npy", gene_token_space)

# 1. get rid of the zero gene tokens
file_path = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/All_countings/training_data_17_tumors_31_classes.csv"
expressed_genes = find_expressed_genes(file_path)
gene_token_space = gene_token_space[expressed_genes,:]
gene_list = []
for gene_idx in range(len(expressed_genes)):
    if expressed_genes[gene_idx]:
        gene_list.append(gene_number_name_mapping[gene_idx])

#%%
# 2. cluster the gene token space (unsupervised Leiden clustering).
from scipy.spatial.distance import pdist, squareform
distance_matrix = squareform(pdist(gene_token_space))

import networkx as nx

threshold_distance = 0.2  # Set a threshold for edge creation
G = nx.Graph()
# First, add all points as nodes
for i in range(len(gene_token_space)):
    G.add_node(i)

for i in range(len(gene_token_space)):
    for j in range(i + 1, len(gene_token_space)):
        if distance_matrix[i, j] < threshold_distance:
            G.add_edge(i, j)

import leidenalg
import igraph as ig

# # Convert networkx graph to igraph
# igraph_G = ig.Graph.from_networkx(G)

# # Apply Leiden algorithm
# partition = leidenalg.find_partition(igraph_G, leidenalg.ModularityVertexPartition)

# # Get cluster labels
# clusters = partition.membership

def apply_leiden_to_subgraph(graph, nodes, partition_type=leidenalg.ModularityVertexPartition, max_size=2000):
    subgraph = graph.subgraph(nodes)
    partition = leidenalg.find_partition(subgraph, partition_type)
    return [[nodes[node] for node in cluster] for cluster in partition]

# Function to ensure all clusters are within the max size limit
def ensure_max_cluster_size(graph, initial_partition, max_cluster_size=2000):
    final_clusters = []
    for cluster in initial_partition:
        if len(cluster) > max_cluster_size:
            # If the cluster is too large, reapply Leiden to this cluster
            sub_clusters = apply_leiden_to_subgraph(graph, cluster)
            final_clusters.extend(sub_clusters)
        else:
            final_clusters.append(cluster)
    return final_clusters

# Convert networkx graph to igraph
igraph_G = ig.Graph.from_networkx(G)

# Apply Leiden algorithm
initial_partition = leidenalg.find_partition(igraph_G, leidenalg.ModularityVertexPartition)
initial_clusters = [cluster for cluster in initial_partition]

# Ensure max cluster size
final_clusters = ensure_max_cluster_size(igraph_G, initial_clusters, max_cluster_size=2000)
clusters = [0]*len(gene_token_space)
for cluster_idx in range(len(final_clusters)):
    for gene_idx in final_clusters[cluster_idx]:
        clusters[gene_idx] = cluster_idx
# %% plot the gene token space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



tsne = TSNE(n_components=2,init='pca',random_state=0)

gene_token_space_2d = tsne.fit_transform(gene_token_space)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
scatter = ax.scatter(gene_token_space_2d[:,0],
                      gene_token_space_2d[:,1],
                        c=clusters,
                        cmap='Spectral',
                          marker='o',
                          s=0.5,)
# # Choose a different colormap, adjust marker size and transparency
# scatter = ax.scatter(gene_token_space[:, 0],  # X coordinates
#                      gene_token_space[:, 1],  # Y coordinates
#                      gene_token_space[:, 2],  # Z coordinates
#                      c=clusters,              # Color by cluster label
#                      cmap='Spectral',         # Different colormap
#                      marker='o',              # Marker style
#                      alpha=0.6,               # Transparency
# )                   
# Create a colorbar and legend
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster Labels')

# Improve labeling and add title
ax.set_xlabel('TSNE X')
ax.set_ylabel('TSNE Y')
# ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot with Improved Colors')

# based on "gene_list, clusters" build gene cluster dictionary
from collections import defaultdict

# Assuming gene_list and clusters are defined
# Example: gene_list = ['gene_name0', 'gene_name1', ...]
#          clusters = ['cluster_label0', 'cluster_label1', ...]
#%%
# Group genes by cluster
clustered_genes = defaultdict(list)
for gene, cluster in zip(gene_list, clusters):
    clustered_genes[cluster].append(gene)

# Write to text files
for cluster, genes in clustered_genes.items():
    if len(genes) > 10:  # Check if the cluster has more than 10 genes
        with open(f"./results/clusters/{cluster}.txt", "w") as file:
            for gene in genes:
                file.write(gene + "\n")
# %%
import tqdm
gene_score = activation['feat.atention_pooling'].cpu().numpy()[0,:,:]
gene_score_sum = np.zeros((gene_score.shape[0],class_num))
gene_score_class_count = np.zeros(class_num)
confusion_matrix_all_test_here = np.zeros((class_num, class_num))

for i,data in tqdm.tqdm(enumerate(train_loader, 0)):
    features1_count, features2_gene_idx, labels = data
    features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
    features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
    model = model.eval()
    pred, _, _ = model(features1_count,features2_gene_idx)
    pred_choice = pred.data.max(1)[1]
    correct = (pred_choice == labels)

    gene_score = activation['feat.atention_pooling'].cpu().numpy()
    pred_labels_np = pred_choice.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for idx_batch in range(labels_np.shape[0]):
        label_i = labels_np[idx_batch]
        pred_i = pred_labels_np[idx_batch]
        # if correct[idx_batch]:
        gene_score_sum[:,label_i] += gene_score[idx_batch,:,0]
        gene_score_class_count[label_i] += 1
        confusion_matrix_all_test_here[label_i,pred_i] += 1

np.save(result_dir+f"/gene_score_sum.npy", gene_score_sum)
np.save(result_dir+f"/gene_score_class_count.npy", gene_score_class_count)
print(confusion_matrix_all_test_here)
np.save(result_dir+f"/confusion_matrix_all_test_here.npy", confusion_matrix_all_test_here)
# %%
gene_score_sum = np.load(result_dir+"/gene_score_sum.npy")
gene_score_class_count = np.load(result_dir+"/gene_score_class_count.npy")

gene_score_mean = gene_score_sum / gene_score_class_count

# Find the indices of the top 20 largest numbers in each column
top_20_indices = np.argsort(-gene_score_mean, axis=0)[:20]
important_genes_all_class = []
with open(f"./results/important_genes.txt", "w") as file:
    for i in range(class_num):
        print(f"Top 20 genes for class {i}:")
        file.write(f"Top 20 genes for class {i}:\n")
        impartant_gene = []
        for j in range(20):
            print(gene_number_name_mapping[top_20_indices[j, i]], end=" ")
            file.write(gene_number_name_mapping[top_20_indices[j, i]]+" ")
            impartant_gene.append(gene_number_name_mapping[top_20_indices[j, i]])
        file.write(f"\n")
        important_genes_all_class.append(impartant_gene)
        print()

# %%
gene_token_space = activation['gstn'].cpu().numpy()[0,:,:]
gene_token_space = gene_token_space.T

distance_matrix = squareform(pdist(gene_token_space))
# Assuming distance_matrix is already defined and gene_token_space is a numpy array
# For example, distance_matrix = squareform(pdist(gene_token_space))

gene_index = 12054  # The index of your reference gene

# Get the distances from gene 12054 to all other genes
distances_to_gene = distance_matrix[gene_index]

# Set the distance to itself as infinity to ignore it
distances_to_gene[gene_index] = np.inf

# Get the indices of the top 5 smallest distances
top_20_closest_indices = np.argsort(distances_to_gene)[:20]

print("Top 5 genes closest to gene 12054:", top_20_closest_indices)
top_20_closet_genes = []
for idx in top_20_closest_indices:
    top_20_closet_genes.append(gene_number_name_mapping[idx])
    print(gene_number_name_mapping[idx], end=" ")
# %%
