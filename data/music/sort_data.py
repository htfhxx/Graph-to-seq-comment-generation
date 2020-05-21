
import re
import json
import requests

# from torchtext.data import Field, BucketIterator
# from torchtext.datasets import Multi30k


def check_chinese(check_str):
    chinese=0
    not_chinese=0
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fa5':  #筛选出来中文
            chinese+=1
        else:
            not_chinese+=1
    return chinese/(chinese+not_chinese)


def get_sort(clear_path, sorted_path ):
    with open( sorted_path, "w", encoding='utf-8') as f:
        aaa=0
        for line in open(clear_path, "r", encoding='utf-8'):
            aaa += 1
            if aaa  % 10 == 0:
                print(aaa)
            #if i>10:
            #    break
            #print('-' * 50)
            #print('\n')
            sample = json.loads(line)

            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            total_comments = sample['total_comments']
            post_total_comment=[ {"text": h} for h in total_comments]

            source = ""
            if check_chinese(songname)>0.8:
                source = source + '歌名：'+songname +'；'
            if check_chinese(singer) > 0.8:
                source = source + '作者：' + singer + '；'

            source = source + '歌词：' + ','.join(lyric.split('\n')) + '。'
            #print(source.encode("utf-8").decode("latin1"))
            #print('\n')
            #print(len(total_comments))
            #print(str(total_comments).encode("utf-8").decode("latin1"))
            #print('\n')
            if len(source) < 50:
                continue
            all_data = {}
            all_data['query'] = source
            all_data['candidates'] = post_total_comment

            url = 'http://9.148.194.19:8081/retrieval_casual_chat/rank'

            # data= {"query": "aaaaaaaaaaaa","candidates": [{"text": "aaaaabbbb"},{"text": "aabababbabbba"},{"text": "bbbbbbbabbb"}]}
            results = requests.post(url=url, data=json.dumps(all_data))
            #print(str(results.json()).encode("utf-8").decode("latin1"))
              
            
            result_comments=[]
            count=0
            index=0
            while index <len(total_comments) and  count<5:
                if results.json()['results'][index]['score']>= 0.7:
                    result_comments.append(results.json()['results'][index]['question'])
                    #print(results.json()['results'][index]['question'].encode("utf-8").decode("latin1"))
                    count+=1
                index+=1
 
            write_dict = {'songname': songname, 'singer': singer, 'lyric': lyric, 'result_comments': result_comments}
            f.write(json.dumps(write_dict, ensure_ascii=False) + '\n')
    print("Finished! ")



def change2string(sorted_path, string_path):
    with open(string_path, "w", encoding='utf-8') as f:
        for line in open(sorted_path, "r", encoding='utf-8'):
            sample = json.loads(line)
            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            result_comments = sample['result_comments']

            source = ""
            try:
                if check_chinese(songname)>0.8:
                    source = source + '歌名：'+songname +'；'
            except:
                #print(line)
                continue
            if check_chinese(singer) > 0.8:
                source = source + '作者：' + singer + '；'

            source = source + '歌词：' + ','.join(lyric.split('\n')) + '。'

            if len(source) < 50 :
                continue
            if len(result_comments) <2:
                continue
            for i in range(len(result_comments)):
                f.write(source.replace(' ', ''))
                f.write('\n')
                f.write(result_comments[i].replace(' ', ''))
                f.write('\n')
                f.write('\n')
    print("finished!")

if __name__ == '__main__':

    clear_path = 'addcomments_clear_data.json'
    sorted_path = 'addcomments_sorted_data.json'
    string_path = 'addcomments_newnew_data.json'


    '''  处理数据  '''
    #get_sort(clear_path, sorted_path)  #    len_all_data:   30106    len_new_data:  29496

    '''  处理格式  '''
    change2string(sorted_path,string_path)





