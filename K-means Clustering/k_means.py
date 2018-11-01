import numpy as np

def readfile(file_path):
    lines = [line for line in open(file_path)]
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = np.array([])
    for line in lines[1:]:
        line = line.strip().split('\t')
        rownames.append(line[0])
        data = np.append(data, np.array(line[1:], dtype = float))
    data = data.reshape((len(lines) - 1, -1))
    return rownames, colnames, data

def pearson(v1, v2):
    sum1 = np.sum(v1)
    sum2 = np.sum(v2)

    sum1Sq = np.sum(v1 ** 2)
    sum2Sq = np.sum(v2 ** 2)
    pSum = np.sum((v1 * v2) ** 2)

    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if den == 0: return 0

    return 1.0 - num / den

def k_means_cluster(rows, distance = pearson, k = 10, num_iter = 100):
    row_num, dimensions = rows.shape
    print(dimensions)
    min_val = np.min(rows, axis = 0)
    max_val = np.max(rows, axis = 0)
    init_center = [[] for i in range(k)]
    for i in range(k):
        rand_init_center = np.array([])
        for d in range(dimensions):
            rand_init_center = np.append(rand_init_center, np.random.randint(max_val[d]))
        init_center[i] = rand_init_center
        
    for i in range(num_iter):
        clusters = [[] for t in range(k)]
        best_matches = [[] for t in range(k)]
        for r in range(row_num):
            min_dist = distance(rows[r], init_center[0])
            center_id = 0
            for j in range(k):
                dist = distance(rows[r], init_center[j])
                if dist < min_dist: 
                    min_dist = dist
                    center_id = j
                    
            clusters[center_id].append(rows[r])
            best_matches[center_id].append(r)

        for j in range(k):
            data_mean = np.mean(clusters[j], axis = 0)
            init_center[j] = data_mean
    return best_matches
        
                
    
    
blognames, words, data = readfile('./data/blogdata.txt')
clust = k_means_cluster(data, k = 10, num_iter = 100)