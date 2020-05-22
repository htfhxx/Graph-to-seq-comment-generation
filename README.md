# Graph-to-seq-comment-generation
Code for the paper "Coherent Comments Generation for Chinese Articles with a Graph-to-Sequence Model"
The data is available at https://pan.baidu.com/s/1b5zAe7qqUBmuHz6nTU95UA The extraction code is 6xdw

segged_content can be extracted from the json file, the original document content needs to be mapped to the comments with url 

## graph: codes for building the topic interaction graph. 
"my_feature_extractor.py": The main entry for building the graph
"write_csv.py": generat the input for which is a csv file.The columns of the csv file are as follow, url, gold comment, topic words extracted from the article, the original title of the article. 
"ccig.py":  The main work horse of the script "my_feature_extractor.py" is from
Note that the extraction method of topic word is not given in this repository, one can use their own methods to extract topic words from articles.


## models : code for our model in "graph2seq.py" and baseline models we adopt.

## "Data.py": code to load the data. 
The class Vocabulary is to build the vocabulary according to the corpus. 
Each "Example" indicates one article and the corresponding title, comment, topic words and some other information. 
A "Batch" is a mini-batch of examples. 
In the "Dataloader", we load the data from the json file extracted by "my_feature_extractor.py" and build the final adjacency matrix of the topic interaction graph. 

##In "train.py", we give the main entrance for the program where one can train or do inference.


# 数据格式
#打出来看看

["label"]
["title"]
["text"]
#   ["g_vertices_betweenness_vec"]
#   ["g_vertices_pagerank_vec"]
#   ["g_vertices_katz_vec"]

["v_names"]
["v_text_features_mat"]     这四个数量对应？
["adj_mat_numsent"]
["adj_mat_tfidf"]



content:  g["v_text_features_mat"]
original_content:  g["text"]
title :   g["title"]
title_index:  g["v_text_features_mat"]的id
target：  g["label"]
adj：     g["adj_mat_tfidf"]    or    ["adj_mat_numsent"]
concept_names:   g["v_names"]  

e = Example(content, original_content, title, title_index, target, adj, concept_names, self.vocab, is_train)


转化为了id
self.content
self.bow      sentence_content（处理过的original_content）
self.title    
self.title_index
self.ori_target      self.ori_targets
self.adj     len(self.content) == self.adj.size(0)
self.concept 




self.content                                                  
self.bow      sentence_content（处理过的original_content）     
self.title    
self.title_index
self.ori_target      self.ori_targets
self.adj     len(self.content) == self.adj.size(0)
self.concept 





### train
python3 train.py  -notrain True


### log文件
'save_epoch_updates_score_updates_checkpoint.pt'

candidate.txt
log.txt   ： time,epoch, updates,loss,accuracy
observe_result.tsv:  title,content
record.csv: epoch, updates, result
command = "perl multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
reference_1.txt




perl multi-bleu.perl /root/tfhuo/Graph-to-seq-comment-generation/data/log/2020-05-11-18:14:39/reference.txt <    /root/tfhuo/Graph-to-seq-comment-generation/data/log/2020-05-11-18:14:39/candidate.txt    >  /root/tfhuo/Graph-to-seq-comment-generation/data/log/2020-05-11-18:14:39/result.txt 

perl multi-bleu.perl ./data/reference.txt  <   ./data/candidate.txt ./data/candidate2.txt 

perl multi-bleu.perl  ./data/reference_3.txt < ./data/reference_4.txt




# seq2seq
if self.use_content:
    src, src_len, src_mask = batch.title_content, batch.title_content_len, batch.title_content_mask
else:
    src, src_len, src_mask = batch.title, batch.title_len, batch.title_mask
tgt = batch.tgt




nohup python3 -u sort_data.py   > myout.file 2>&1 &
31824

nohup python3 -u train.py   > myout3.file 2>&1 &

cat myout3.file | grep 'Finished an epoch'

# 数据处理
# loss等于0 的是啥
# gpu指定的问题（在gpu测试机上不能用呢还）


较为详细的预处理，取前三条热评：2020-05-20-21_00_16

加入普通评论，取前三条相似度>0.7的：   


测试10： 2020-05-21-10_30_45
测试30： 2020-05-21-10_38_48