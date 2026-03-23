# 📦 CBOW & GloVe

> שיטות לייצוג מילים כווקטורים — Word Embeddings.  
> המטרה: **מילים דומות = ווקטורים קרובים במרחב.**

---

## Word2Vec — שתי גישות

### CBOW — Continuous Bag of Words
מנבא מילה מרכזית מתוך מילות הסביבה:

```
קלט:  ["הכלב", "רץ", "בגן"]
פלט:  "מהר"
```

### Skip-gram
הפוך מ-CBOW — מנבא מילות סביבה ממילה מרכזית:

```
קלט:  "מהר"
פלט:  ["הכלב", "רץ", "בגן"]
```

| | CBOW | Skip-gram |
|-|------|-----------|
| מהירות | ⚡⚡⚡ | ⚡⚡ |
| מילים נדירות | פחות טוב | טוב יותר |
| dataset קטן | עובד | פחות טוב |

---

## GloVe — Global Vectors

במקום חלון מקומי — GloVe בונה **מטריצת co-occurrence** על כל הקורפוס:

$$J = \sum_{i,j} f(X_{ij})(w_i^T\tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

### המאפיין המפורסם
```
king − man + woman ≈ queen
Paris − France + Italy ≈ Rome
```

מרחקים בין ווקטורים משקפים קשרים סמנטיים!

---

## קוד

```python
import gensim.downloader as api

# טעינת GloVe מאומן
model = api.load('glove-wiki-gigaword-100')

# מילים דומות
model.most_similar('king')
# [('queen', 0.75), ('prince', 0.70), ...]

# אנלוגיה
model.most_similar(
    positive=['king', 'woman'],
    negative=['man']
)
# → queen

# קוסינוס בין שתי מילים
model.similarity('cat', 'dog')   # 0.82
model.similarity('cat', 'table') # 0.21
```

---

> 💡 **GloVe לעומת Word2Vec:** GloVe לרוב מביא תוצאות טובות יותר כי הוא מנצל מידע גלובלי מכל הקורפוס.
