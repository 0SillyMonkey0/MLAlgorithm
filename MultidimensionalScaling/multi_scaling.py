import numpy as np

def pearson(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    sum1 = np.sum(v1)
    sum2 = np.sum(v2)

    sum1Sq = np.sum(v1 ** 2)
    sum2Sq = np.sum(v2 ** 2)
    pSum = np.sum((v1 * v2) ** 2)

    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0: return 0

    return 1.0 - num / den
    

def scale_down(data, distance=pearson, rate = 0.01):
    num_epoch = 100
    n, dim = data.shape
    real_dist = np.array([[distance(data[i], data[j]) for i in range(n)] for j in range(n)])
    loc = np.random.rand(n, 2)
    for epoch in range(num_epoch):
        fake_dist = np.array([[distance(loc[i], loc[j]) for i in range(n)] for j in range(n)])
        grad = np.zeros_like(loc)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                error = (fake_dist[i, j] - real_dist[i, j]) / (real_dist[i ,j] + 1)
                grad[i] += (loc[i] - loc[j]) * error / fake_dist[i, j]
        for i in range(n):
            loc[i] -= rate * grad[i]
    return loc
    
blognames, words, data = readfile('./data/blogdata.txt')
clust = scale_down(data)