The point of this notebook is to build a movie recommender systems based on movie genres and ratings 

Dataset from MovieLens: https://grouplens.org/datasets/movielens/latest/


```python
# load libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

## Data reading and EDA


```python
def summary(df):
    print (f"shape of data: {df.shape}")
    sum = pd.DataFrame(df.dtypes, columns=['data type'])
    sum['#missing'] = df.isnull().sum().values
    sum['%missing'] = df.isnull().sum().values / len(df)
    sum['unique'] = df.nunique().values

    # add statistics
    desc = pd.DataFrame(df.describe(include='all').transpose())
    sum['mean'] = desc['mean'].values
    sum['std'] = desc['std'].values
    sum['min'] = desc['min'].values
    sum['25%'] = desc['25%'].values
    sum['50%'] = desc['50%'].values
    sum['75%'] = desc['75%'].values
    sum['max'] = desc['max'].values

    return sum
```


```python
links = pd.read_csv("links.csv")
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
```


```python
summary(links).style.background_gradient(cmap='YlOrBr')
```

    shape of data: (9742, 3)


<table id="T_0c34f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0c34f_level0_col0" class="col_heading level0 col0" >data type</th>
      <th id="T_0c34f_level0_col1" class="col_heading level0 col1" >#missing</th>
      <th id="T_0c34f_level0_col2" class="col_heading level0 col2" >%missing</th>
      <th id="T_0c34f_level0_col3" class="col_heading level0 col3" >unique</th>
      <th id="T_0c34f_level0_col4" class="col_heading level0 col4" >mean</th>
      <th id="T_0c34f_level0_col5" class="col_heading level0 col5" >std</th>
      <th id="T_0c34f_level0_col6" class="col_heading level0 col6" >min</th>
      <th id="T_0c34f_level0_col7" class="col_heading level0 col7" >25%</th>
      <th id="T_0c34f_level0_col8" class="col_heading level0 col8" >50%</th>
      <th id="T_0c34f_level0_col9" class="col_heading level0 col9" >75%</th>
      <th id="T_0c34f_level0_col10" class="col_heading level0 col10" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0c34f_level0_row0" class="row_heading level0 row0" >movieId</th>
      <td id="T_0c34f_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_0c34f_row0_col1" class="data row0 col1" >0</td>
      <td id="T_0c34f_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_0c34f_row0_col3" class="data row0 col3" >9742</td>
      <td id="T_0c34f_row0_col4" class="data row0 col4" >42200.353623</td>
      <td id="T_0c34f_row0_col5" class="data row0 col5" >52160.494854</td>
      <td id="T_0c34f_row0_col6" class="data row0 col6" >1.000000</td>
      <td id="T_0c34f_row0_col7" class="data row0 col7" >3248.250000</td>
      <td id="T_0c34f_row0_col8" class="data row0 col8" >7300.000000</td>
      <td id="T_0c34f_row0_col9" class="data row0 col9" >76232.000000</td>
      <td id="T_0c34f_row0_col10" class="data row0 col10" >193609.000000</td>
    </tr>
    <tr>
      <th id="T_0c34f_level0_row1" class="row_heading level0 row1" >imdbId</th>
      <td id="T_0c34f_row1_col0" class="data row1 col0" >int64</td>
      <td id="T_0c34f_row1_col1" class="data row1 col1" >0</td>
      <td id="T_0c34f_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_0c34f_row1_col3" class="data row1 col3" >9742</td>
      <td id="T_0c34f_row1_col4" class="data row1 col4" >677183.898173</td>
      <td id="T_0c34f_row1_col5" class="data row1 col5" >1107227.576760</td>
      <td id="T_0c34f_row1_col6" class="data row1 col6" >417.000000</td>
      <td id="T_0c34f_row1_col7" class="data row1 col7" >95180.750000</td>
      <td id="T_0c34f_row1_col8" class="data row1 col8" >167260.500000</td>
      <td id="T_0c34f_row1_col9" class="data row1 col9" >805568.500000</td>
      <td id="T_0c34f_row1_col10" class="data row1 col10" >8391976.000000</td>
    </tr>
    <tr>
      <th id="T_0c34f_level0_row2" class="row_heading level0 row2" >tmdbId</th>
      <td id="T_0c34f_row2_col0" class="data row2 col0" >float64</td>
      <td id="T_0c34f_row2_col1" class="data row2 col1" >8</td>
      <td id="T_0c34f_row2_col2" class="data row2 col2" >0.000821</td>
      <td id="T_0c34f_row2_col3" class="data row2 col3" >9733</td>
      <td id="T_0c34f_row2_col4" class="data row2 col4" >55162.123793</td>
      <td id="T_0c34f_row2_col5" class="data row2 col5" >93653.481487</td>
      <td id="T_0c34f_row2_col6" class="data row2 col6" >2.000000</td>
      <td id="T_0c34f_row2_col7" class="data row2 col7" >9665.500000</td>
      <td id="T_0c34f_row2_col8" class="data row2 col8" >16529.000000</td>
      <td id="T_0c34f_row2_col9" class="data row2 col9" >44205.750000</td>
      <td id="T_0c34f_row2_col10" class="data row2 col10" >525662.000000</td>
    </tr>
  </tbody>
</table>




there are some missing values for `tmdbld` variable 


```python
summary(movies).style.background_gradient(cmap='YlOrBr')
```

    shape of data: (9742, 3)




<table id="T_882ad">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_882ad_level0_col0" class="col_heading level0 col0" >data type</th>
      <th id="T_882ad_level0_col1" class="col_heading level0 col1" >#missing</th>
      <th id="T_882ad_level0_col2" class="col_heading level0 col2" >%missing</th>
      <th id="T_882ad_level0_col3" class="col_heading level0 col3" >unique</th>
      <th id="T_882ad_level0_col4" class="col_heading level0 col4" >mean</th>
      <th id="T_882ad_level0_col5" class="col_heading level0 col5" >std</th>
      <th id="T_882ad_level0_col6" class="col_heading level0 col6" >min</th>
      <th id="T_882ad_level0_col7" class="col_heading level0 col7" >25%</th>
      <th id="T_882ad_level0_col8" class="col_heading level0 col8" >50%</th>
      <th id="T_882ad_level0_col9" class="col_heading level0 col9" >75%</th>
      <th id="T_882ad_level0_col10" class="col_heading level0 col10" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_882ad_level0_row0" class="row_heading level0 row0" >movieId</th>
      <td id="T_882ad_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_882ad_row0_col1" class="data row0 col1" >0</td>
      <td id="T_882ad_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_882ad_row0_col3" class="data row0 col3" >9742</td>
      <td id="T_882ad_row0_col4" class="data row0 col4" >42200.353623</td>
      <td id="T_882ad_row0_col5" class="data row0 col5" >52160.494854</td>
      <td id="T_882ad_row0_col6" class="data row0 col6" >1.000000</td>
      <td id="T_882ad_row0_col7" class="data row0 col7" >3248.250000</td>
      <td id="T_882ad_row0_col8" class="data row0 col8" >7300.000000</td>
      <td id="T_882ad_row0_col9" class="data row0 col9" >76232.000000</td>
      <td id="T_882ad_row0_col10" class="data row0 col10" >193609.000000</td>
    </tr>
    <tr>
      <th id="T_882ad_level0_row1" class="row_heading level0 row1" >title</th>
      <td id="T_882ad_row1_col0" class="data row1 col0" >object</td>
      <td id="T_882ad_row1_col1" class="data row1 col1" >0</td>
      <td id="T_882ad_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_882ad_row1_col3" class="data row1 col3" >9737</td>
      <td id="T_882ad_row1_col4" class="data row1 col4" >nan</td>
      <td id="T_882ad_row1_col5" class="data row1 col5" >nan</td>
      <td id="T_882ad_row1_col6" class="data row1 col6" >nan</td>
      <td id="T_882ad_row1_col7" class="data row1 col7" >nan</td>
      <td id="T_882ad_row1_col8" class="data row1 col8" >nan</td>
      <td id="T_882ad_row1_col9" class="data row1 col9" >nan</td>
      <td id="T_882ad_row1_col10" class="data row1 col10" >nan</td>
    </tr>
    <tr>
      <th id="T_882ad_level0_row2" class="row_heading level0 row2" >genres</th>
      <td id="T_882ad_row2_col0" class="data row2 col0" >object</td>
      <td id="T_882ad_row2_col1" class="data row2 col1" >0</td>
      <td id="T_882ad_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_882ad_row2_col3" class="data row2 col3" >951</td>
      <td id="T_882ad_row2_col4" class="data row2 col4" >nan</td>
      <td id="T_882ad_row2_col5" class="data row2 col5" >nan</td>
      <td id="T_882ad_row2_col6" class="data row2 col6" >nan</td>
      <td id="T_882ad_row2_col7" class="data row2 col7" >nan</td>
      <td id="T_882ad_row2_col8" class="data row2 col8" >nan</td>
      <td id="T_882ad_row2_col9" class="data row2 col9" >nan</td>
      <td id="T_882ad_row2_col10" class="data row2 col10" >nan</td>
    </tr>
  </tbody>
</table>





```python
summary(ratings).style.background_gradient(cmap='YlOrBr')
```

    shape of data: (100836, 4)





<table id="T_4c92c">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4c92c_level0_col0" class="col_heading level0 col0" >data type</th>
      <th id="T_4c92c_level0_col1" class="col_heading level0 col1" >#missing</th>
      <th id="T_4c92c_level0_col2" class="col_heading level0 col2" >%missing</th>
      <th id="T_4c92c_level0_col3" class="col_heading level0 col3" >unique</th>
      <th id="T_4c92c_level0_col4" class="col_heading level0 col4" >mean</th>
      <th id="T_4c92c_level0_col5" class="col_heading level0 col5" >std</th>
      <th id="T_4c92c_level0_col6" class="col_heading level0 col6" >min</th>
      <th id="T_4c92c_level0_col7" class="col_heading level0 col7" >25%</th>
      <th id="T_4c92c_level0_col8" class="col_heading level0 col8" >50%</th>
      <th id="T_4c92c_level0_col9" class="col_heading level0 col9" >75%</th>
      <th id="T_4c92c_level0_col10" class="col_heading level0 col10" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4c92c_level0_row0" class="row_heading level0 row0" >userId</th>
      <td id="T_4c92c_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_4c92c_row0_col1" class="data row0 col1" >0</td>
      <td id="T_4c92c_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_4c92c_row0_col3" class="data row0 col3" >610</td>
      <td id="T_4c92c_row0_col4" class="data row0 col4" >326.127564</td>
      <td id="T_4c92c_row0_col5" class="data row0 col5" >182.618491</td>
      <td id="T_4c92c_row0_col6" class="data row0 col6" >1.000000</td>
      <td id="T_4c92c_row0_col7" class="data row0 col7" >177.000000</td>
      <td id="T_4c92c_row0_col8" class="data row0 col8" >325.000000</td>
      <td id="T_4c92c_row0_col9" class="data row0 col9" >477.000000</td>
      <td id="T_4c92c_row0_col10" class="data row0 col10" >610.000000</td>
    </tr>
    <tr>
      <th id="T_4c92c_level0_row1" class="row_heading level0 row1" >movieId</th>
      <td id="T_4c92c_row1_col0" class="data row1 col0" >int64</td>
      <td id="T_4c92c_row1_col1" class="data row1 col1" >0</td>
      <td id="T_4c92c_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_4c92c_row1_col3" class="data row1 col3" >9724</td>
      <td id="T_4c92c_row1_col4" class="data row1 col4" >19435.295718</td>
      <td id="T_4c92c_row1_col5" class="data row1 col5" >35530.987199</td>
      <td id="T_4c92c_row1_col6" class="data row1 col6" >1.000000</td>
      <td id="T_4c92c_row1_col7" class="data row1 col7" >1199.000000</td>
      <td id="T_4c92c_row1_col8" class="data row1 col8" >2991.000000</td>
      <td id="T_4c92c_row1_col9" class="data row1 col9" >8122.000000</td>
      <td id="T_4c92c_row1_col10" class="data row1 col10" >193609.000000</td>
    </tr>
    <tr>
      <th id="T_4c92c_level0_row2" class="row_heading level0 row2" >rating</th>
      <td id="T_4c92c_row2_col0" class="data row2 col0" >float64</td>
      <td id="T_4c92c_row2_col1" class="data row2 col1" >0</td>
      <td id="T_4c92c_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_4c92c_row2_col3" class="data row2 col3" >10</td>
      <td id="T_4c92c_row2_col4" class="data row2 col4" >3.501557</td>
      <td id="T_4c92c_row2_col5" class="data row2 col5" >1.042529</td>
      <td id="T_4c92c_row2_col6" class="data row2 col6" >0.500000</td>
      <td id="T_4c92c_row2_col7" class="data row2 col7" >3.000000</td>
      <td id="T_4c92c_row2_col8" class="data row2 col8" >3.500000</td>
      <td id="T_4c92c_row2_col9" class="data row2 col9" >4.000000</td>
      <td id="T_4c92c_row2_col10" class="data row2 col10" >5.000000</td>
    </tr>
    <tr>
      <th id="T_4c92c_level0_row3" class="row_heading level0 row3" >timestamp</th>
      <td id="T_4c92c_row3_col0" class="data row3 col0" >int64</td>
      <td id="T_4c92c_row3_col1" class="data row3 col1" >0</td>
      <td id="T_4c92c_row3_col2" class="data row3 col2" >0.000000</td>
      <td id="T_4c92c_row3_col3" class="data row3 col3" >85043</td>
      <td id="T_4c92c_row3_col4" class="data row3 col4" >1205946087.368469</td>
      <td id="T_4c92c_row3_col5" class="data row3 col5" >216261035.995132</td>
      <td id="T_4c92c_row3_col6" class="data row3 col6" >828124615.000000</td>
      <td id="T_4c92c_row3_col7" class="data row3 col7" >1019123866.000000</td>
      <td id="T_4c92c_row3_col8" class="data row3 col8" >1186086662.000000</td>
      <td id="T_4c92c_row3_col9" class="data row3 col9" >1435994144.500000</td>
      <td id="T_4c92c_row3_col10" class="data row3 col10" >1537799250.000000</td>
    </tr>
  </tbody>
</table>




lowest rating is 0.5, highest rating is 5 across all movies


```python
summary(tags).style.background_gradient(cmap='YlOrBr')
```

    shape of data: (3683, 4)





<table id="T_fc592">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fc592_level0_col0" class="col_heading level0 col0" >data type</th>
      <th id="T_fc592_level0_col1" class="col_heading level0 col1" >#missing</th>
      <th id="T_fc592_level0_col2" class="col_heading level0 col2" >%missing</th>
      <th id="T_fc592_level0_col3" class="col_heading level0 col3" >unique</th>
      <th id="T_fc592_level0_col4" class="col_heading level0 col4" >mean</th>
      <th id="T_fc592_level0_col5" class="col_heading level0 col5" >std</th>
      <th id="T_fc592_level0_col6" class="col_heading level0 col6" >min</th>
      <th id="T_fc592_level0_col7" class="col_heading level0 col7" >25%</th>
      <th id="T_fc592_level0_col8" class="col_heading level0 col8" >50%</th>
      <th id="T_fc592_level0_col9" class="col_heading level0 col9" >75%</th>
      <th id="T_fc592_level0_col10" class="col_heading level0 col10" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fc592_level0_row0" class="row_heading level0 row0" >userId</th>
      <td id="T_fc592_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_fc592_row0_col1" class="data row0 col1" >0</td>
      <td id="T_fc592_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_fc592_row0_col3" class="data row0 col3" >58</td>
      <td id="T_fc592_row0_col4" class="data row0 col4" >431.149335</td>
      <td id="T_fc592_row0_col5" class="data row0 col5" >158.472553</td>
      <td id="T_fc592_row0_col6" class="data row0 col6" >2.000000</td>
      <td id="T_fc592_row0_col7" class="data row0 col7" >424.000000</td>
      <td id="T_fc592_row0_col8" class="data row0 col8" >474.000000</td>
      <td id="T_fc592_row0_col9" class="data row0 col9" >477.000000</td>
      <td id="T_fc592_row0_col10" class="data row0 col10" >610.000000</td>
    </tr>
    <tr>
      <th id="T_fc592_level0_row1" class="row_heading level0 row1" >movieId</th>
      <td id="T_fc592_row1_col0" class="data row1 col0" >int64</td>
      <td id="T_fc592_row1_col1" class="data row1 col1" >0</td>
      <td id="T_fc592_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_fc592_row1_col3" class="data row1 col3" >1572</td>
      <td id="T_fc592_row1_col4" class="data row1 col4" >27252.013576</td>
      <td id="T_fc592_row1_col5" class="data row1 col5" >43490.558803</td>
      <td id="T_fc592_row1_col6" class="data row1 col6" >1.000000</td>
      <td id="T_fc592_row1_col7" class="data row1 col7" >1262.500000</td>
      <td id="T_fc592_row1_col8" class="data row1 col8" >4454.000000</td>
      <td id="T_fc592_row1_col9" class="data row1 col9" >39263.000000</td>
      <td id="T_fc592_row1_col10" class="data row1 col10" >193565.000000</td>
    </tr>
    <tr>
      <th id="T_fc592_level0_row2" class="row_heading level0 row2" >tag</th>
      <td id="T_fc592_row2_col0" class="data row2 col0" >object</td>
      <td id="T_fc592_row2_col1" class="data row2 col1" >0</td>
      <td id="T_fc592_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_fc592_row2_col3" class="data row2 col3" >1589</td>
      <td id="T_fc592_row2_col4" class="data row2 col4" >nan</td>
      <td id="T_fc592_row2_col5" class="data row2 col5" >nan</td>
      <td id="T_fc592_row2_col6" class="data row2 col6" >nan</td>
      <td id="T_fc592_row2_col7" class="data row2 col7" >nan</td>
      <td id="T_fc592_row2_col8" class="data row2 col8" >nan</td>
      <td id="T_fc592_row2_col9" class="data row2 col9" >nan</td>
      <td id="T_fc592_row2_col10" class="data row2 col10" >nan</td>
    </tr>
    <tr>
      <th id="T_fc592_level0_row3" class="row_heading level0 row3" >timestamp</th>
      <td id="T_fc592_row3_col0" class="data row3 col0" >int64</td>
      <td id="T_fc592_row3_col1" class="data row3 col1" >0</td>
      <td id="T_fc592_row3_col2" class="data row3 col2" >0.000000</td>
      <td id="T_fc592_row3_col3" class="data row3 col3" >3411</td>
      <td id="T_fc592_row3_col4" class="data row3 col4" >1320031966.823785</td>
      <td id="T_fc592_row3_col5" class="data row3 col5" >172102450.437126</td>
      <td id="T_fc592_row3_col6" class="data row3 col6" >1137179352.000000</td>
      <td id="T_fc592_row3_col7" class="data row3 col7" >1137521216.000000</td>
      <td id="T_fc592_row3_col8" class="data row3 col8" >1269832564.000000</td>
      <td id="T_fc592_row3_col9" class="data row3 col9" >1498456765.500000</td>
      <td id="T_fc592_row3_col10" class="data row3 col10" >1537098603.000000</td>
    </tr>
  </tbody>
</table>




Insights from the summary of these files:
1. Not much missing values. Great !
2. It seems like movie id is what we will be using to connect these tables
3. There were quite alot of duplicates for movie Id, for now I will just drop them



```python
# drop duplicates
movies.drop_duplicates(subset=['movieId'], inplace=True)
ratings.drop_duplicates(subset=['userId', 'movieId'], inplace=True)
tags.drop_duplicates(subset=['userId', 'movieId', 'tag'], inplace=True)
```

## Feature Engineering


```python
# Extract the year of release of the movie and create a new column for it
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)

# Converet genres into a list of genres
movies['genres'] = movies['genres'].apply(lambda x: x.split(" | "))

# Create a new df for movie ratings, containing the movieId and its average rating
average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']
```


```python
movies_df = movies.merge(average_ratings, on='movieId')
movies_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>average_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>[Adventure|Animation|Children|Comedy|Fantasy]</td>
      <td>1995</td>
      <td>3.920930</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>[Adventure|Children|Fantasy]</td>
      <td>1995</td>
      <td>3.431818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>[Comedy|Romance]</td>
      <td>1995</td>
      <td>3.259615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>[Comedy|Drama|Romance]</td>
      <td>1995</td>
      <td>2.357143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>[Comedy]</td>
      <td>1995</td>
      <td>3.071429</td>
    </tr>
  </tbody>
</table>
</div>



I will just use a [jaccard similarity test](https://en.wikipedia.org/wiki/Jaccard_index) as I think using a ML model is an overkill for this simple project.


```python
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection 
    return intersection / union

# genre and rating based recommendations
def recommend_by_genres_and_ratings(genres, movies_df, top_n=10):
    input_genres = set(genres)
    movies_df['similarity'] = movies_df['genres'].apply(lambda x: jaccard_similarity(input_genres, set(x)))
    return movies_df.sort_values(by=['similarity', 'average_rating'], ascending=[False, False]).head(top_n)
```

## Let's test !


```python
try:
    input_movie = 'Up'
    input_genres = movies_df[movies_df['title'].str.contains(input_movie)]['genres'].iloc[0]
    recommendations = recommend_by_genres_and_ratings(input_genres, movies_df)
    print(f'Movie: {input_movie}, Genre: {input_genres}\n')
    print("Recommended Movies:")
    print(recommendations[['title', 'genres', 'average_rating']])

# If the movie does not exists in the original list
except ValueError and IndexError:
    print ("There are no related movies !")
```

    Movie: Up, Genre: ['Drama|Romance']
    
    Recommended Movies:
                                                      title           genres  \
    2232  Man and a Woman, A (Un homme et une femme) (1966)  [Drama|Romance]   
    2317                              Sandpiper, The (1965)  [Drama|Romance]   
    3499  Moscow Does Not Believe in Tears (Moskva sleza...  [Drama|Romance]   
    3802                                        Rain (2001)  [Drama|Romance]   
    4103         Cruel Romance, A (Zhestokij Romans) (1984)  [Drama|Romance]   
    4245                                   Lady Jane (1986)  [Drama|Romance]   
    4667                                   Jane Eyre (1944)  [Drama|Romance]   
    5417                             Mr. Skeffington (1944)  [Drama|Romance]   
    2878  Affair of Love, An (Liaison pornographique, Un...  [Drama|Romance]   
    4946  Happy Together (a.k.a. Buenos Aires Affair) (C...  [Drama|Romance]   
    
          average_rating  
    2232            5.00  
    2317            5.00  
    3499            5.00  
    3802            5.00  
    4103            5.00  
    4245            5.00  
    4667            5.00  
    5417            5.00  
    2878            4.75  
    4946            4.75  


## Next Steps

That's it ! It's just a weekend project so I am not spending that much time on it. Some of next steps worth considering if you want to expand it include:
1. Use the larger/full dataset.
2. Do content-based filtering such as director, actors or other relevant keyword. Can use natural language techniques like TF-IDF.
3. Do collaborative filtering, using past behaviour of users (their ratings or interactions) to make personalized recommendations.
4. Combine content-based filtering and collaborative filtering (Hybrid systems).

One great example is to understand how [Netflix's recommendation system](https://help.netflix.com/en/node/100639) works, which could really gives you an idea on how real-world recommendation system works.


