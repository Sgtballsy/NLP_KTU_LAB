import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string 
def cosine_sim(t1,t2): 
    t1Tok = word_tokenize(t1) 
    t2Tok = word_tokenize(t2) 
    sw = stopwords.words('english') 
    t1Set = set(t1Tok) - set(sw) - set(string.punctuation) 
    t2Set = set(t2Tok) - set(sw) - set(string.punctuation) 
    v = t1Set.union(t2Set) 
    l1 = [] 
    for i in v: 
        if i in t1Set: 
            l1.append(1) 
        else: 
            l1.append(0) 
    l2 = [] 
    for i in v: 
        if i in t2Set: 
            l2.append(1) 
        else: 
            l2.append(0) 
    c = 0 
    for i in range(len(v)): 
        c += l1[i] * l2[i] 
    cos = c / float((sum(l1) * sum(l2)) ** 0.5) 
    return cos 
n = int(input("Enter the number of sentences: ")) 
allsentences = [] 
for i in range(n): 
    allsentences.append(input(f"Enter sentence {i+1}: ")) 
coslist = {} 
for i in allsentences: 
    for j in allsentences: 
        if i != j: 
            p = cosine_sim(i, j) 
            if p not in coslist: 
                coslist[p] = [i,j] 
print("Max Cosine Similarity: ", max(coslist)) 
print("Sentences: ", coslist[max(coslist)])