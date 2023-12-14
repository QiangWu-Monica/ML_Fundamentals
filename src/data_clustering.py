import numpy as np
import math
from  time import time
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

f = open('seeds_dataset.txt','r')
N = len(f.readlines())
f.seek(0)
data_all = np.array([float(i) for i in np.array(f.read().split())]).reshape(N,-1)
data_train, labels = data_all[:,:-1], data_all[:,-1]
D = data_train.shape[1]
f.close()

def purity(clusters,K):
    purity = []
    for k in range(K):
        m, d = [j for j,x in enumerate(clusters) if x==k], []
        for i in range(1,4):
            d.append(len(set(m)&set([j for j,x in enumerate(labels) if x==i])))
        purity.append(max(d))
    return sum(purity)/N

def rand_index(clusters,K):
    ind = 0
    for i in range(N-1):
        for j in range(i+1,N):
            ind += (clusters[i] == clusters[j]) == (labels[i] == labels[j])
    ind += (clusters[N-1] == clusters[N-2]) == (labels[N-1] == labels[N-2])
    return 2*ind/(N*(N-1))

def normal_mutual_infor(clusters,K):
    X = np.array([len([j for j,x in enumerate(labels) if x==i])/N for i in [1,2,3]]) # classes
    Y = np.array([len([j for j,x in enumerate(clusters) if x==i])/N for i in range(K)]) # clusters
    den = 0
    for i in X:
        if i != 0:
            den -= i*np.log2(i)
    for i in Y:
        if i != 0:
            den -= i*np.log2(i)
    condi = 0
    for i in range(K):
        points = [j for j,x in enumerate(clusters) if x==i]
        count = [len([j for j,x in enumerate(labels[points]) if x==i])/N for i in [1,2,3]]
        partial = 0
        for i in count:
            if i != 0:
                partial -= i*np.log2(i)
        condi += (len(points)/N)*partial
    return 2*(-X@np.log2(X)-condi)/den


# ------------------Perform K-means Algorithm----------------------------- #
def K_means(centers,K):
    t0, iter_num, times = time(), 1, []
    clusters = [] # Initialize the K centers by randomly picking some points
    for j in range(N): # Assign each data point to a cluster
        distances = [np.linalg.norm(i) for i in centers-data_train[j]]
        clusters.append(np.argmin(distances))
    clusters_new, new_centers = [], centers
    for i in range(K): # Use the mean values to redefine center points
        if [j for j,x in enumerate(clusters) if x==i] != []:
            new_centers[i] = data_train[[j for j,x in enumerate(clusters) if x==i]].mean(0)
    for j in range(N): # Assign each data point to this new cluster
        distances = [np.linalg.norm(i) for i in new_centers-data_train[j]]
        clusters_new.append(np.argmin(distances))
    times.append(time()-t0)
    while clusters != clusters_new: # Iterate this process
        clusters, centers = clusters_new, new_centers
        clusters_new = [] # Reconstruct the center points and clusters
        for i in range(K): # Use the mean values to redefine center points
            if [j for j,x in enumerate(clusters) if x==i] != []:
                new_centers[i] = data_train[[j for j,x in enumerate(clusters) if x==i]].mean(0)
        for j in range(N): # Assign each data point to this new cluster
            distances = [i@i for i in new_centers-data_train[j]]
            clusters_new.append(np.argmin(distances))
        iter_num += 1
        times.append(time()-t0)
    t1 = time()
    print("K-means. No. of Clusters:",K,'Time:',t1-t0, 'Iterations:',iter_num)
    print('K-means. Purity:',purity(clusters_new,K),'Rand Index:',rand_index(clusters_new,K))
    print(normal_mutual_infor(clusters_new,K))
    print()
    visualization(clusters_new,K,'K-means')
    cpu_time(times,iter_num,'K-means')
    return new_centers


# -------Perform Accelerated K-means with Triangle Inequality------------ #
def Accelerated_Kmeans(centers,K):
    times = []
    t0, iter_num = time(), 1
    clusters = [] # Initialize the K centers by randomly picking some points
    dists = squareform(pdist(centers))
    remaining_point = []
    for i in range(N):
        d, l = 0, np.linalg.norm(data_train[i]-centers[0])
        for j in range(1,K):
            if 2*l <= dists[j][j-1]:
                continue
            else:
                p = np.linalg.norm(data_train[i]-centers[j])
                if l > p:
                    l,d = p,j
        clusters.append(d)
    new_centers = centers
    for i in range(K): # Use the mean values to redefine center points
        if [j for j,x in enumerate(clusters) if x==i] != []:
            new_centers[i] = data_train[[j for j,x in enumerate(clusters) if x==i]].mean(0)
    new_dists = squareform(pdist(new_centers))
    min_dists = np.min(new_dists+100*np.eye(K),1)
    end = True
    for i in range(N):
        d = clusters[i]
        l = np.linalg.norm(data_train[i]-new_centers[d])
        if 2*l <= min_dists[d]:
            continue
        else:
            for j in range(K):
                if j == d:
                    continue
                else:
                    if 2*l <= new_dists[j][j-1]:
                        continue
                    else:
                        p = np.linalg.norm(data_train[i]-centers[j])
                        if l > p:
                            l,d = p,j
        if d != clusters[i]:
            end = False
            clusters[i] = d
    times.append(time()-t0)
    while end == False:
        end = True
        for i in range(K): # Use the mean values to redefine center points
            if [j for j,x in enumerate(clusters) if x==i] != []:
                new_centers[i] = data_train[[j for j,x in enumerate(clusters) if x==i]].mean(0)
        new_dists = squareform(pdist(new_centers))
        min_dists = np.min(new_dists+100*np.eye(K),1)
        for i in range(N):
            d = clusters[i]
            l = np.linalg.norm(data_train[i]-new_centers[d])
            if 2*l <= min_dists[d]:
                continue
            else:
                for j in range(K):
                    if j == d:
                        continue
                    elif 2*l <= new_dists[j][j-1]:
                        continue
                    else:
                        p = np.linalg.norm(data_train[i]-centers[j])
                        if l > p:
                            l,d = p,j
            if d != clusters[i]:
                end = False
                clusters[i] = d
        iter_num += 1
        times.append(time()-t0)
    t1 = time()
    print("Accelerated K-means. No. of Clusters:",K,'Time:',t1-t0,'Iterations:',iter_num)
    print('Accelerated K-means. Purity:',purity(clusters,K),'Rand Index:',rand_index(clusters,K))
    print(normal_mutual_infor(clusters,K))
    print()
    visualization(clusters,K,'Accelerated K-means')
    cpu_time(times,iter_num,'Accelerated K-means')
    return new_centers


# ---------------------Perform Soft K-means------------------------------ #
def Soft_Kmeans(centers,K):
    t0, iter_num, beta, tol, times = time(), 1, 0.5, 1e-2, []
    clusters, degree = [], [] # Initialize the K centers by randomly picking some points
    for j in range(N): # Assign each data point to a cluster
        point_degree = np.exp(np.array([-np.linalg.norm(i) for i in centers-data_train[j]])*beta)
        degree.append(point_degree/sum(point_degree))
    centers_new = (np.array(degree).T@data_train)/np.sum(np.array(degree),0).reshape(-1,1)
    times.append(time()-t0)
    while max(np.linalg.norm(centers_new-centers,axis=1)) > tol:
        centers, degree = centers_new, []
        for j in range(N): # Assign each data point to a cluster
            point_degree = np.exp(np.array([-np.linalg.norm(i) for i in centers-data_train[j]])*beta)
            degree.append(point_degree/sum(point_degree))
        centers_new = (np.array(degree).T@data_train)/np.sum(np.array(degree),0).reshape(-1,1)
        iter_num += 1
        times.append(time()-t0)
    t1 = time()
    clusters_new = []

    for j in range(N): # Assign each data point to the converged centers
        distances = [np.linalg.norm(i) for i in centers_new-data_train[j]]
        clusters_new.append(np.argmin(distances))
    print("Soft K-means. No. of Clusters:",K,'Time:',t1-t0,'Iterations:',iter_num)
    print('Soft K-means. Purity:',purity(clusters_new,K),'Rand Index:',rand_index(clusters_new,K))
    print(normal_mutual_infor(clusters_new,K))
    print()
    visualization(clusters_new,K,'Soft K-means')
    cpu_time(times,iter_num,'Soft K-means')
    return centers_new


# ------------------------Perform EM Algorithm-------------------------- #
def GMM_EM(means,K):
    t0, iter_num, tol, times = time(), 1, 1e-2, []
    covariances, coefficients = np.array([0.1*np.eye(D)]*K), np.ones(K)/K
    res, inverse = [], np.linalg.inv(covariances)

    # E step
    for i in range(N):
        diff = data_train[i]-means
        for j in range(K):
            res.append(np.exp(-diff[j]@inverse[j]@diff[j].T/2)/(np.linalg.det(2*math.pi*covariances[j])))
    res = coefficients*np.array(res).reshape(N,K)
    log_likelihood = np.sum(np.log(np.sum(res,1)))
    res = res.T/np.sum(res,1)

    # M step
    N_k = np.sum(res,1)
    new_means, new_coefficients, new_covariances = ((res@data_train).T/N_k).T, N_k/N, []
    for i in range(K):
        diffs = []
        for j in range(N):
            diffs.append(np.outer((data_train[j]-new_means[i]),(data_train[j]-new_means[i])))
        new_covariances.append(np.tensordot(res[i],np.array(diffs),axes=1)/N_k[i])
    new_covariances = np.array(new_covariances)
    new_res, inverse = [], np.linalg.inv(new_covariances)
    for i in range(N):
        diff = data_train[i]-new_means
        for j in range(K):
            new_res.append(np.exp(-diff[j]@inverse[j]@diff[j].T/2)/(np.linalg.det(2*math.pi*new_covariances[j])))
    new_res = new_coefficients*np.array(new_res).reshape(N,K)
    new_log_likelihood = np.sum(np.log(np.sum(new_res,1)))
    times.append(time()-t0)
    while abs(log_likelihood-new_log_likelihood) > tol:
        iter_num += 1
        
        # E step
        means, covariances, coefficients, res, log_likelihood = new_means, new_covariances, new_coefficients, new_res.T/np.sum(new_res,1), new_log_likelihood
        N_k = np.sum(res,1)
        new_means, new_coefficients, new_covariances = ((res@data_train).T/N_k).T, N_k/N, []

        # M step
        for i in range(K):
            diffs = []
            for j in range(N):
                diffs.append(np.outer((data_train[j]-new_means[i]),(data_train[j]-new_means[i])))
            new_covariances.append(np.tensordot(res[i],np.array(diffs),axes=1)/N_k[i])
        new_covariances = np.array(new_covariances)
        new_res, inverse = [], np.linalg.inv(new_covariances)
        for i in range(N):
            diff = data_train[i]-new_means
            for j in range(K):
                new_res.append(np.exp(-diff[j]@inverse[j]@diff[j].T/2)/(np.linalg.det(2*math.pi*new_covariances[j])))
        new_res = new_coefficients*np.array(new_res).reshape(N,K)
        new_log_likelihood = np.sum(np.log(np.sum(new_res,1)))
        times.append(time()-t0)
    t1, clusters_new = time(), []
    for j in range(N): # Assign each data point to the converged centers
        distances = [np.linalg.norm(i) for i in new_means-data_train[j]]
        clusters_new.append(np.argmin(distances))
    print("GMM-EM. No. of Clusters:",K,'Time:',t1-t0,'Iterations:',iter_num)
    print('GMM-EM. Purity:',purity(clusters_new,K),'Rand Index:',rand_index(clusters_new,K))
    print(normal_mutual_infor(clusters_new,K))
    print()
    visualization(clusters_new,K,'GMM-EM')
    cpu_time(times,iter_num,'GMM-EM')
    return new_means

def visualization(clusters,K,method):
    plt.figure()
    data_std = StandardScaler().fit_transform(data_train)
    tsne = TSNE(n_components=2, learning_rate=100)
    tsne.fit_transform(data_std)
    data = np.array(tsne.embedding_)
    pca = sk.decomposition.PCA(n_components=2)
    data_pca = pca.fit_transform(data_std)
    plt.scatter(data_pca[:,0], data_pca[:,1], c=clusters)
    plt.xlabel('dim X')
    plt.ylabel('dim Y')
    plt.title('PCA projection of '+method+'\n'+f'No. of centers K={K}')
    plt.savefig(f'figs/{method}_{K}.png')

def cpu_time(time,iter,method):
    plt.figure()
    plt.plot(range(iter),time,label=method)
    plt.xlabel('Iteration Number')
    plt.ylabel('CPU Time')
    plt.legend()
    plt.title(f'CPU Time with Respect to Iterations, K={K}')
    plt.savefig(f'figs/cpu_time_{K}.png')
    
# ---------------------Performing the algorithms------------------------- #
Ks = [3,6]
for K in Ks:
    initial = data_train[[int(i) for i in np.linspace(0,N-1,K)]]
    K_means(initial,K)
    Accelerated_Kmeans(initial,K)
    Soft_Kmeans(initial,K)
    GMM_EM(initial,K)

    # sensitivity analysis
    Kmean, acc_Kmean, soft_Kmean, EM = [],[],[],[]
    choice = np.random.choice(N,K)
    initial = data_train[choice]
    evaluate, results, test = [K_means(initial,K), Accelerated_Kmeans(initial,K), Soft_Kmeans(initial,K), GMM_EM(initial,K)],[], 10
    for i in range(test):
        new_initial = data_train[np.random.choice(N,K)]
        new_evaluate = [K_means(new_initial,K), Accelerated_Kmeans(new_initial,K), Soft_Kmeans(new_initial,K), GMM_EM(new_initial,K)]
        dist_input = np.linalg.norm(np.sum(np.array([i-j for i in initial for j in new_initial]),1))
        for k in range(4):
            dist_output = np.linalg.norm(np.sum(np.array([i-j for i in evaluate[k] for j in new_evaluate[k]]),1))
            results.append(dist_output/dist_input)
    plt.figure()
    plt.scatter(range(test),results[::4],c='r',label='K Means')
    plt.scatter(range(test),results[1::4],c='y',label='Accelerated K Means')
    plt.scatter(range(test),results[2::4],c='b',label='Soft K Means')
    plt.scatter(range(test),results[3::4],c='g',label='GMM-EM')
    plt.xlabel('Trial Tests')
    plt.ylabel('Sensitivity Index')
    plt.legend()
    plt.savefig(f'sensitivity_{K}.png')
