import numpy as np
from PIL import Image, ImageDraw

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]

class node:
    def __init__(self, col = 1, value = None, results = None, tb = None, fb = None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb
        
def divide_set(data, col, val):
    split_func = None
    if isinstance(val, int) or isinstance(val, float):
        split_func = lambda row: row[col] >= val
    else:
        split_func = lambda row: row[col] == val
    set1 = [row for row in data if split_func(row)]
    set2 = [row for row in data if not split_func(row)]
    return set1, set2

def unique_counts(data):
    dim = len(data[0])
    y_count  = {}
    for row in data:
       y_count.setdefault(row[dim - 1], 0)
       y_count[row[dim - 1]] += 1
    return y_count

def entorpy(data):
    n = len(data)
    if n == 0:
        return 0
    
    y_count  = unique_counts(data)
    p = [-np.log(y_count[key] / n) * (y_count[key] / n) for key in y_count]
    en = sum(p)
    return en

def build_tree(data, scoref = entorpy, prune_value = 0):
    if len(data) == 0:
      return node()
    current_score = entorpy(data)
    best_gain = 0
    best_set = ([], [])
    best_criteria = None
    col_num = len(data[0])
    for col in range(col_num - 1):
        feature_dict = {}
        for row in data:
            feature_dict[row[col]] = 1
        for val in feature_dict.keys():
            set1, set2 = divide_set(data, col, val)    
            #print('col=', col, 'val=', val, '  ', len(set1), len(set2))
            p = len(set1) / len(data)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, val)
                best_set = (set1, set2)
    print(len(best_set[0]), len(best_set[1]))
    
    if best_gain > prune_value:
        tbranch = build_tree(best_set[0])
        fbranch = build_tree(best_set[1])
        return node(col=best_criteria[0], value=best_criteria[1],
                            tb=tbranch, fb=fbranch)
    else:                        
        return node(results = unique_counts(data))
        
def classify(data, tree):
    if tree.results is not None:
        return tree.results
    if isinstance(data[tree.col], int) or  isinstance(data[tree.col], int):
        if data[tree.col] >= tree.value:
            return classify(data, tree.tb)
        else:
            return classify(data, tree.fb)
            
    else:
        if data[tree.col] == tree.value:
            return classify(data, tree.tb)
        else:
            return classify(data, tree.fb)
            
def miss_data_classify(data, tree):
    if tree.results is not None:
        return tree.results
        
    v = data[tree.col]
    if v is None:
        tr = md_classify(data, tree.tb) 
        fr = md_classify(data, tree.fb)
        t_count = sum(tr.values())
        f_count = sum(fr.values())
        t_weight = t_count / (t_count + f_count)
        f_weight = f_count / (t_count + f_count)
        results = {}
        for k, v in tr.items():
            results[k] = t_weight * v
        for k, v in fr.items():
            results[k] = f_weight * v
        return results
    else:
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch  = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return md_classify(data, branch)
        
# prune after the tree has been built
def prune(tree, mingain):
    if tree.tb.results is None:
        prune(tree.tb, mingain)
    if tree.fb.results is None:
        prune(tree.fb, mingain)
    
    if tree.tb.results is not None and tree.fb.results is not None:
        fb, tb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)

        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)
            
def get_width(tree):
    if tree.tb is None and tree.fb is None:
        return 1
    return get_width(tree.tb) + get_width(tree.fb) + 1
    
def get_depth(tree):
    if tree.tb is None and tree.fb is None:
        return 1
    return max(get_depth(tree.tb), get_depth(tree.fb)) + 1
    
def draw_tree(tree, jpeg='tree.jpg'):
    w = get_width(tree) * 100
    h = get_depth(tree) * 100 + 20

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')

def draw_node(draw, tree, x, y):
    if tree.results is None:
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))
        left_node = (x - get_width(tree.fb) * 100 / 2, y + 100)
        right_node = (x + get_width(tree.tb) * 100 / 2, y + 100)
        draw.line((x, y, left_node[0], left_node[1]), fill = (255, 0, 0))
        draw.line((x, y, right_node[0], right_node[1]), fill = (255, 0, 0))
        
        draw_node(draw, tree.tb, left_node[0], left_node[1])
        draw_node(draw, tree.fb, right_node[0], right_node[1])
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))
        
        
data = my_data
root = build_tree(data)
draw_tree(root)

print(classify(['(direct)', 'USA', 'yes', 5], root))
print(miss_data_classify(['(direct)', 'USA', None, 5], root))