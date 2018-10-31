from math import  sqrt
import numpy as np
from PIL import Image, ImageDraw

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

class bicluster:
    def __init__(self, vec, left = None, right = None, distance = 0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance

def hcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1
    clust = [bicluster(rows[i], id = i) for i in range(len(rows))]
    
    while len(clust) > 1:
        lowestpair = (0, 1)
        min_distance = float('inf')
        length = len(clust)
        for i in range(length):
            for j in range(i + 1, length):
                distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                if distances[(clust[i].id, clust[j].id)] < min_distance:
                    min_distance = distances[(clust[i].id, clust[j].id)]
                    lowestpair = (i, j)

        merge_vec = (clust[lowestpair[0]].vec + clust[lowestpair[1]].vec) / 2 
        newcluster = bicluster(merge_vec, left = clust[lowestpair[0]], \
                               right = clust[lowestpair[1]], distance=min_distance,\
                              id=currentclustid )
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        
        clust.append(newcluster)
    return clust[0]

def getheight(clust):
    if clust.left == None and clust.right == None:
        return 1
    else:
        return getheight(clust.left) + getheight(clust.right)

def getdepth(clust):
    if clust.left == None and clust.right == None:
        return 0
    else:
        return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

def drawdendrogram(clust, labels, jpeg = 'cluster.jpg'):
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)
    scaling = float(w - 150) /depth
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h/2, 10, h/2), fill=(255, 0, 0))
    drawnode(draw, clust, 10, (h/2), scaling, labels)
    img.save(jpeg, 'JPEG')

def drawnode(draw, clust, x, y, scaling, labels):
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        l1 = clust.distance * scaling

        # draw the vertical line, the branch with bigger height get more space, namely
        # the line of it would be shorter, vice versa
        # And the height should be divided by 2 bacuase we draw the tree from the middle
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill = (255, 0, 0))

        # draw the horizontal line
        draw.line((x, top + h1 / 2, x + l1, top + h1 / 2), fill = (255, 0, 0))
        draw.line((x, bottom - h2 / 2, x + l1, bottom - h2 / 2), fill = (255, 0 , 0))

        drawnode(draw, clust.left, x + l1, top + h1 / 2, scaling, labels)
        drawnode(draw, clust.right, x + l1, bottom - h2 / 2, scaling, labels)
    else:
        draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))