import re


class NaiveBayes():
    def __init__(self):
        self.cate_num = {}
        self.feature_cate_count = {}
    
    def getwords(doc):
        splitter = re.compile(r'\W+')
        words = [s.lower() for s in splitter.split(doc)
                 if 2 < len(s) < 20]
        return dict([(w, 1) for w in words])
    
    def train(self, doc, cate):
        map = getwords(doc)
        self.cate_num.setdefault(cate, 0)
        self.cate_num[cate] += 1
        for key in map:
            self.feature_cate_count.setdefault(key, {})
            self.feature_cate_count[key].setdefault(cate, 0)
            self.feature_cate_count[key][cate] += 1
            
            
    def get_feature_count(self, word, cate):
        if word in self.feature_cate_count and cate in self.feature_cate_count[word]:
            return self.feature_cate_count[word][cate]
        else: return 0
        
    def get_conditional_prob(self, word, cate):
        if self.cate_num[cate] == 0: return 0
        weight = 1
        prior_prob = 0.5
        conditional_prob = self.get_feature_count(word, cate) / self.cate_num[cate]
        totals = sum([self.get_feature_count(word, cate) for cate in self.cate_num])
        weighted_prob = ((weight * prior_prob) + (totals * conditional_prob)) / (weight + totals)
        return weighted_prob
        
        
    def get_marginal_prob(self, cate):
        cate_total_num = 0
        for c in self.cate_num:
            cate_total_num += self.cate_num[c]
        return self.cate_num[cate] / cate_total_num
        
    def predict(self, doc):
        doc = getwords(doc)
        max_score = -1
        cate = None
        ret = None
        for cate in self.cate_num:
            current_score = self.get_marginal_prob(cate)
            for word in doc:
                current_score *= self.get_conditional_prob(word, cate)
            if current_score > max_score:
                max_score = current_score
                ret = cate
        return ret
        
classifier = NaiveBayes()    
classifier.train('Nobody owns the water.', 'good')
classifier.train('the quick rabbit jumps fences', 'good')
classifier.train('buy pharmaceuticals now', 'bad')
classifier.train('make quick money at the online casino', 'bad')
classifier.train('the quick brown fox jumps', 'good')

c1assifier.predict('quick rabbit')
classifier.predict('quick money')