

```python
import pandas as pd
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from textblob import TextBlob
from string import punctuation
import re
import cgi
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from pandas import Series
import networkx as nx
import jgraph
```


```python
data_dir_path = '/Users/hemasundar/PycharmProjects/text-analytics/data/'
stop_words_file = os.path.join(data_dir_path,'stopwords.txt')
reviews_file = os.path.join(data_dir_path,'dead_pool_reviews.txt')
```


```python
def remove_stop_words(tokens_list):
    stop_words = list(stopwords.words('english'))
    additional_stop_words = map(lambda x : x.strip(),open(stop_words_file).readlines())
    additional_stop_words = list(additional_stop_words)
    stop_words.extend(additional_stop_words)
    stop_words.extend(['film','movie','movies','dont','give','make','makes','dont','didnt','doesnt','time','word','made','watch','deadpool','films','time'])
    stop_words_removed = [word for word in tokens_list if word not in stop_words]
    return stop_words_removed
```


```python
def tokenizer(text):
    words = word_tokenize(text)
    return words
```


```python
def clean_text(text):
    text = text.lower() # Convert to lower case
    remove_punctuation = text.translate(None,punctuation) #Remove punctuation
    ascii_encode = remove_punctuation.decode("utf8").encode('ascii', 'ignore')
    remove_html = cgi.escape(ascii_encode) # Remove html
    remove_line_break = remove_html.replace('\n',' ') # Remove new lines
    strip_spaces = remove_line_break.strip() # Strip the spaces
    remove_numbers = re.sub('[^a-zA-Z ]','',strip_spaces)
    return remove_numbers
```


```python
cols = ['ratings','reviews']
df = pd.read_csv(reviews_file,sep='\t',lineterminator='\n',names=cols)
df['clean_reviews'] = df['reviews'].apply(clean_text)
df['final_word_list'] = df['clean_reviews'].apply(tokenizer).apply(remove_stop_words).apply(' '.join)
```


```python
terms = ' '.join(df['final_word_list'])
```


```python
wordcloud = WordCloud(
    background_color='black',
    max_words=5000,
    width=800,
    height=500
).generate(terms)

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('./cloud2.png', dpi=500)
plt.show()
```


![png](output_7_0.png)



```python
clean_reviews = df['clean_reviews']
```


```python
print(df['final_word_list'][:10])
```

    0    eighth xmen franchise merc mouth seeks revenge...
    1    firstly state completely hilarious reading rev...
    2    glance typical superhero due marvel hilarious ...
    3    admit huge fan reading comics trades excited m...
    4    lot marvel moviesbut honesti marvel rated pret...
    5    ive waited long cinema love ive watched days a...
    6    opening credits funny unusual marvel swears in...
    7    merc mouth xmen origins wolverine laid bottom ...
    8    hugely entertaining motion picture admit comic...
    9    easily offended people worst marvel marvel gen...
    Name: final_word_list, dtype: object



```python
vectorizer = CountVectorizer(max_df = 0.8,min_df = 0.2,analyzer = 'word',ngram_range=(1,2))
word_vec = vectorizer.fit_transform(df['final_word_list'])
vocab = vectorizer.vocabulary_
count_dict = {w: Series(word_vec.getcol(vocab[w]).data) for w in vocab}
tdm_df = pd.DataFrame(count_dict).fillna(0)
```


```python
# Xc = (X.T * X)
# g = sp.diags(1./Xc.diagonal())
# Xc_norm = g * Xc # normalized co-occurence matrix
```


```python
print(tdm_df)
```

        action  bad  character  characters  comedy  comic  funny  good  great  \
    0      2.0  1.0        1.0         3.0     1.0    1.0    1.0   1.0    1.0   
    1      1.0  1.0        1.0         1.0     1.0    1.0    1.0   2.0    1.0   
    2      1.0  2.0        3.0         1.0     3.0    1.0    1.0   1.0    2.0   
    3      1.0  1.0        1.0         1.0     1.0    2.0    4.0   3.0    2.0   
    4      2.0  1.0        4.0         1.0     1.0    1.0    1.0   1.0    1.0   
    5      1.0  1.0        1.0         1.0     1.0    1.0    1.0   1.0    1.0   
    6      1.0  1.0        1.0         1.0     1.0    2.0    2.0   1.0    1.0   
    7      3.0  4.0        1.0         1.0     1.0    1.0    1.0   1.0    4.0   
    8      1.0  2.0        2.0         1.0     2.0    2.0    1.0   1.0    2.0   
    9      1.0  1.0        3.0         1.0     1.0    1.0    1.0   1.0    1.0   
    10     1.0  2.0        1.0         1.0     2.0    1.0    2.0   1.0    1.0   
    11     1.0  1.0        1.0         1.0     1.0    1.0    1.0   1.0    3.0   
    12     1.0  1.0        1.0         1.0     2.0    3.0    3.0   1.0    1.0   
    13     1.0  1.0        2.0         1.0     2.0    1.0    1.0   2.0    3.0   
    14     1.0  2.0        5.0         1.0     1.0    3.0    2.0   1.0    2.0   
    15     2.0  2.0        2.0         1.0     2.0    1.0    1.0   2.0    1.0   
    16     1.0  1.0        1.0         1.0     1.0    1.0    1.0   1.0    1.0   
    17     1.0  1.0        1.0         1.0     1.0    2.0    1.0   2.0    1.0   
    18     1.0  1.0        1.0         1.0     1.0    1.0    1.0   1.0    1.0   
    19     1.0  1.0        1.0         1.0     1.0    1.0    1.0   1.0    1.0   
    20     1.0  1.0        1.0         1.0     1.0    1.0    2.0   1.0    2.0   
    21     1.0  5.0        1.0         1.0     1.0    1.0    1.0   1.0    1.0   
    22     1.0  1.0        1.0         1.0     1.0    1.0    3.0   3.0    1.0   
    23     1.0  2.0        1.0         1.0     1.0    2.0    1.0   1.0    2.0   
    24     1.0  1.0        2.0         1.0     1.0    6.0    1.0   1.0    2.0   
    25     2.0  1.0        1.0         1.0     0.0    1.0    1.0   2.0    2.0   
    26     1.0  3.0        2.0         1.0     0.0    1.0    1.0   3.0    1.0   
    27     2.0  1.0        1.0         0.0     0.0    3.0    1.0   1.0    1.0   
    28     2.0  1.0        6.0         0.0     0.0    1.0    1.0   1.0    0.0   
    29     1.0  1.0        2.0         0.0     0.0    4.0    1.0   2.0    0.0   
    30     2.0  1.0        1.0         0.0     0.0    1.0    1.0   3.0    0.0   
    31     2.0  2.0        1.0         0.0     0.0    1.0    1.0   2.0    0.0   
    32     1.0  0.0        0.0         0.0     0.0    2.0    2.0   1.0    0.0   
    33     2.0  0.0        0.0         0.0     0.0    0.0    1.0   1.0    0.0   
    34     1.0  0.0        0.0         0.0     0.0    0.0    1.0   2.0    0.0   
    35     2.0  0.0        0.0         0.0     0.0    0.0    1.0   1.0    0.0   
    36     1.0  0.0        0.0         0.0     0.0    0.0    3.0   1.0    0.0   
    37     1.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    38     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    39     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    40     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    41     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    42     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    43     0.0  0.0        0.0         0.0     0.0    0.0    0.0   1.0    0.0   
    44     0.0  0.0        0.0         0.0     0.0    0.0    0.0   0.0    0.0   
    45     0.0  0.0        0.0         0.0     0.0    0.0    0.0   0.0    0.0   
    46     0.0  0.0        0.0         0.0     0.0    0.0    0.0   0.0    0.0   
    47     0.0  0.0        0.0         0.0     0.0    0.0    0.0   0.0    0.0   
    48     0.0  0.0        0.0         0.0     0.0    0.0    0.0   0.0    0.0   
    
        humor  ...   people  plot  reynolds  ryan  ryan reynolds  scenes  story  \
    0     1.0  ...      2.0   1.0         1     1            1.0     1.0    3.0   
    1     1.0  ...      4.0   1.0         1     1            1.0     1.0    1.0   
    2     1.0  ...      1.0   2.0         1     2            1.0     1.0    2.0   
    3     2.0  ...      1.0   1.0         2     2            2.0     1.0    1.0   
    4     1.0  ...      2.0   1.0         2     2            2.0     1.0    4.0   
    5     1.0  ...      2.0   1.0         1     1            1.0     1.0    1.0   
    6     2.0  ...      2.0   1.0         1     1            1.0     1.0    1.0   
    7     2.0  ...      4.0   5.0         2     2            1.0     1.0    2.0   
    8     3.0  ...      2.0   1.0         1     1            1.0     1.0    1.0   
    9     1.0  ...      2.0   1.0         2     1            1.0     1.0    1.0   
    10    1.0  ...      1.0   2.0         1     1            1.0     1.0    1.0   
    11    1.0  ...      1.0   2.0         1     1            1.0     2.0    1.0   
    12    1.0  ...      2.0   1.0         1     1            2.0     1.0    1.0   
    13    1.0  ...      1.0   1.0         1     2            1.0     1.0    1.0   
    14    2.0  ...      1.0   1.0         1     1            2.0     1.0    2.0   
    15    1.0  ...      2.0   1.0         2     2            1.0     1.0    1.0   
    16    1.0  ...      2.0   3.0         1     1            2.0     1.0    1.0   
    17    1.0  ...      1.0   1.0         2     2            1.0     1.0    2.0   
    18    2.0  ...      1.0   1.0         1     1            1.0     1.0    2.0   
    19    2.0  ...      1.0   2.0         2     1            1.0     1.0    1.0   
    20    2.0  ...      1.0   1.0         1     1            2.0     1.0    1.0   
    21    2.0  ...      1.0   1.0         1     1            1.0     1.0    1.0   
    22    2.0  ...      1.0   2.0         1     2            2.0     1.0    1.0   
    23    1.0  ...      1.0   1.0         1     1            1.0     1.0    2.0   
    24    0.0  ...      2.0   1.0         2     2            1.0     0.0    1.0   
    25    0.0  ...      2.0   0.0         1     1            1.0     0.0    1.0   
    26    0.0  ...      1.0   0.0         2     1            1.0     0.0    1.0   
    27    0.0  ...      1.0   0.0         1     1            1.0     0.0    1.0   
    28    0.0  ...      2.0   0.0         1     1            1.0     0.0    1.0   
    29    0.0  ...      1.0   0.0         1     1            1.0     0.0    1.0   
    30    0.0  ...      1.0   0.0         1     1            1.0     0.0    0.0   
    31    0.0  ...      1.0   0.0         1     2            1.0     0.0    0.0   
    32    0.0  ...      2.0   0.0         2     1            2.0     0.0    0.0   
    33    0.0  ...      1.0   0.0         1     1            1.0     0.0    0.0   
    34    0.0  ...      2.0   0.0         1     1            1.0     0.0    0.0   
    35    0.0  ...      3.0   0.0         1     2            2.0     0.0    0.0   
    36    0.0  ...      1.0   0.0         4     1            1.0     0.0    0.0   
    37    0.0  ...      1.0   0.0         1     1            1.0     0.0    0.0   
    38    0.0  ...      0.0   0.0         1     1            3.0     0.0    0.0   
    39    0.0  ...      0.0   0.0         1     1            1.0     0.0    0.0   
    40    0.0  ...      0.0   0.0         2     2            1.0     0.0    0.0   
    41    0.0  ...      0.0   0.0         1     1            1.0     0.0    0.0   
    42    0.0  ...      0.0   0.0         1     1            1.0     0.0    0.0   
    43    0.0  ...      0.0   0.0         1     3            0.0     0.0    0.0   
    44    0.0  ...      0.0   0.0         7     1            0.0     0.0    0.0   
    45    0.0  ...      0.0   0.0         1     1            0.0     0.0    0.0   
    46    0.0  ...      0.0   0.0         2     1            0.0     0.0    0.0   
    47    0.0  ...      0.0   0.0         1     1            0.0     0.0    0.0   
    48    0.0  ...      0.0   0.0         1     1            0.0     0.0    0.0   
    
        superhero  violence  xmen  
    0         3.0       1.0   2.0  
    1         1.0       1.0   2.0  
    2         2.0       3.0   1.0  
    3         1.0       1.0   1.0  
    4         3.0       1.0   1.0  
    5         1.0       1.0   3.0  
    6         1.0       1.0   2.0  
    7         1.0       1.0   1.0  
    8         3.0       1.0   3.0  
    9         1.0       1.0   1.0  
    10        1.0       1.0   1.0  
    11        1.0       1.0   1.0  
    12        1.0       2.0   4.0  
    13        1.0       1.0   1.0  
    14        1.0       1.0   1.0  
    15        1.0       2.0   1.0  
    16        1.0       1.0   2.0  
    17        4.0       2.0   1.0  
    18        1.0       1.0   1.0  
    19        1.0       1.0   1.0  
    20        1.0       1.0   1.0  
    21        1.0       2.0   5.0  
    22        1.0       1.0   1.0  
    23        1.0       1.0   1.0  
    24        2.0       3.0   1.0  
    25        1.0       1.0   1.0  
    26        1.0       1.0   0.0  
    27        3.0       2.0   0.0  
    28        4.0       1.0   0.0  
    29        1.0       5.0   0.0  
    30        1.0       1.0   0.0  
    31        0.0       0.0   0.0  
    32        0.0       0.0   0.0  
    33        0.0       0.0   0.0  
    34        0.0       0.0   0.0  
    35        0.0       0.0   0.0  
    36        0.0       0.0   0.0  
    37        0.0       0.0   0.0  
    38        0.0       0.0   0.0  
    39        0.0       0.0   0.0  
    40        0.0       0.0   0.0  
    41        0.0       0.0   0.0  
    42        0.0       0.0   0.0  
    43        0.0       0.0   0.0  
    44        0.0       0.0   0.0  
    45        0.0       0.0   0.0  
    46        0.0       0.0   0.0  
    47        0.0       0.0   0.0  
    48        0.0       0.0   0.0  
    
    [49 rows x 23 columns]



```python
word_vec_cooc = (word_vec.T * word_vec) # this is co-occurrence matrix in sparse csr format
word_vec_cooc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
word_vec_dense = word_vec_cooc.todense()
print(word_vec_dense) # print out matrix in dense format
```

    [[  0  20  31  14  27  42  13  36  38  20  24  22  44  26  14  59  45  40
       10  28  26  17  23]
     [ 20   0  13   7  14  21  22  28  14  10  43  10  12  19  28  16  18  11
        8  16  10  17  12]
     [ 31  13   0  24  12  50  32  37  25  23  32  35  37  29  15  82  54  52
       10  26  44  29  34]
     [ 14   7  24   0   8  19  14  17  14  14  19   5  25  20   8  19  14  12
        5  22  21  10  10]
     [ 27  14  12   8   0  17  15  26   4   3  35  24  15  15   6  29  27  23
        4   9  11   9   8]
     [ 42  21  50  19  17   0  19  30  33  24  34  28  36  37   7  69  42  40
       11  19  29  23  24]
     [ 13  22  32  14  15  19   0  23  20   8  57  19  34  34  16  24  18  14
        6  16   7  14  23]
     [ 36  28  37  17  26  30  23   0  25  16  31  19  34  30  16  40  39  35
       13  28  22  22  24]
     [ 38  14  25  14   4  33  20  25   0  21  25  15  31  24   9  48  31  30
        2  16  30  10  23]
     [ 20  10  23  14   3  24   8  16  21   0  15  20  30  30  16  32  23  21
        8  11  23  28  10]
     [ 24  43  32  19  35  34  57  31  25  15   0  23  20  43  47  48  35  31
       11  22  31  15  20]
     [ 22  10  35   5  24  28  19  19  15  20  23   0  28  17  12  46  33  31
        9  12  26  23  18]
     [ 44  12  37  25  15  36  34  34  31  30  20  28   0  56  19  39  41  35
       10  30  25  26  48]
     [ 26  19  29  20  15  37  34  30  24  30  43  17  56   0  24  30  27  23
       14  20  16  29  26]
     [ 14  28  15   8   6   7  16  16   9  16  47  12  19  24   0  18  14  10
        7   6  12  25   6]
     [ 59  16  82  19  29  69  24  40  48  32  48  46  39  30  18   0 101  97
       14  22  53  31  41]
     [ 45  18  54  14  27  42  18  39  31  23  35  33  41  27  14 101   0  81
       14  19  35  26  39]
     [ 40  11  52  12  23  40  14  35  30  21  31  31  35  23  10  97  81   0
       12  17  34  21  32]
     [ 10   8  10   5   4  11   6  13   2   8  11   9  10  14   7  14  14  12
        0  14  11  23   8]
     [ 28  16  26  22   9  19  16  28  16  11  22  12  30  20   6  22  19  17
       14   0  12  15  22]
     [ 26  10  44  21  11  29   7  22  30  23  31  26  25  16  12  53  35  34
       11  12   0  23  17]
     [ 17  17  29  10   9  23  14  22  10  28  15  23  26  29  25  31  26  21
       23  15  23   0  17]
     [ 23  12  34  10   8  24  23  24  23  10  20  18  48  26   6  41  39  32
        8  22  17  17   0]]



```python
adj_df = pd.DataFrame(word_vec_dense,columns = node_list)
```


```python
node_list = map(lambda x : str(x),vocab.keys())
```


```python
adj_df['source'] = node_list
```


```python
adj_df['target'] = node_list[::-1]
```


```python
print(adj_df)
```

        xmen  marvel  people  love  action  funny  great  reynolds  comedy  \
    0      0      20      31    14      27     42     13        36      38   
    1     20       0      13     7      14     21     22        28      14   
    2     31      13       0    24      12     50     32        37      25   
    3     14       7      24     0       8     19     14        17      14   
    4     27      14      12     8       0     17     15        26       4   
    5     42      21      50    19      17      0     19        30      33   
    6     13      22      32    14      15     19      0        23      20   
    7     36      28      37    17      26     30     23         0      25   
    8     38      14      25    14       4     33     20        25       0   
    9     20      10      23    14       3     24      8        16      21   
    10    24      43      32    19      35     34     57        31      25   
    11    22      10      35     5      24     28     19        19      15   
    12    44      12      37    25      15     36     34        34      31   
    13    26      19      29    20      15     37     34        30      24   
    14    14      28      15     8       6      7     16        16       9   
    15    59      16      82    19      29     69     24        40      48   
    16    45      18      54    14      27     42     18        39      31   
    17    40      11      52    12      23     40     14        35      30   
    18    10       8      10     5       4     11      6        13       2   
    19    28      16      26    22       9     19     16        28      16   
    20    26      10      44    21      11     29      7        22      30   
    21    17      17      29    10       9     23     14        22      10   
    22    23      12      34    10       8     24     23        24      23   
    
        superhero      ...        humor  ryan  bad  ryan reynolds  violence  good  \
    0          20      ...           14    59   45             40        10    28   
    1          10      ...           28    16   18             11         8    16   
    2          23      ...           15    82   54             52        10    26   
    3          14      ...            8    19   14             12         5    22   
    4           3      ...            6    29   27             23         4     9   
    5          24      ...            7    69   42             40        11    19   
    6           8      ...           16    24   18             14         6    16   
    7          16      ...           16    40   39             35        13    28   
    8          21      ...            9    48   31             30         2    16   
    9           0      ...           16    32   23             21         8    11   
    10         15      ...           47    48   35             31        11    22   
    11         20      ...           12    46   33             31         9    12   
    12         30      ...           19    39   41             35        10    30   
    13         30      ...           24    30   27             23        14    20   
    14         16      ...            0    18   14             10         7     6   
    15         32      ...           18     0  101             97        14    22   
    16         23      ...           14   101    0             81        14    19   
    17         21      ...           10    97   81              0        12    17   
    18          8      ...            7    14   14             12         0    14   
    19         11      ...            6    22   19             17        14     0   
    20         23      ...           12    53   35             34        11    12   
    21         28      ...           25    31   26             21        23    15   
    22         10      ...            6    41   39             32         8    22   
    
        characters  jokes  story         source  
    0           26     17     23           xmen  
    1           10     17     12         marvel  
    2           44     29     34         people  
    3           21     10     10           love  
    4           11      9      8         action  
    5           29     23     24          funny  
    6            7     14     23          great  
    7           22     22     24       reynolds  
    8           30     10     23         comedy  
    9           23     28     10      superhero  
    10          31     15     20           plot  
    11          26     23     18          comic  
    12          25     26     48      character  
    13          16     29     26         scenes  
    14          12     25      6          humor  
    15          53     31     41           ryan  
    16          35     26     39            bad  
    17          34     21     32  ryan reynolds  
    18          11     23      8       violence  
    19          12     15     22           good  
    20           0     23     17     characters  
    21          23      0     17          jokes  
    22          17     17      0          story  
    
    [23 rows x 24 columns]



```python

```


```python
#A = np.matrix(word_vec_dense)

Gr = nx.from_pandas_dataframe(adj_df,'source','source',node_list)

```


```python
print()
```


```python
print(node_list)
```

    ['xmen', 'marvel', 'people', 'love', 'action', 'funny', 'great', 'reynolds', 'comedy', 'superhero', 'plot', 'comic', 'character', 'scenes', 'humor', 'ryan', 'bad', 'ryan reynolds', 'violence', 'good', 'characters', 'jokes', 'story']



```python
node_list = ['marvel', 'people', 'love', 'action', 'funny', 'great', 'reynolds', 'comedy', 'superhero', 'plot', 'comic', 'character', 'scenes', 'humor', 'ryan', 'bad', 'ryan reynolds', 'violence', 'good', 'characters', 'jokes', 'story']
```


```python
pos=nx.spring_layout(Gr)
```


```python
nx.draw_networkx(Gr,node_size=500, nodelist = node_list,alpha = 0.8)
plt.show()
```


![png](output_25_0.png)


#### 


```python

```


```python

```


```python

```
