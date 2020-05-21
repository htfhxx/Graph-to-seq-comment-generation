import re
import json
# from torchtext.data import Field, BucketIterator
# from torchtext.datasets import Multi30k



#   1111111111111111
def get_data(file_path,notnull_path):
    i=0
    len_all_data=0
    len_new_data=0
    with open(notnull_path, "w",encoding='utf=8') as f:
        for line in open(file_path, "r",encoding='utf=8'):
            i+=1
            if i%10000==0:
                print(i)
            sample = json.loads(line)
            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            hot_comment = sample['hot_comments']

            len_all_data += 1
            if songname is None or singer is None :
                continue
            # 忽略掉没有歌名作者
            if len(songname)==0 or len(singer)==0:
                continue
            # 忽略掉没有热评的
            if len(hot_comment) == 0  :
                continue
            # 忽略掉没有歌词的
            if len(lyric.replace('\n',' ').replace(' ','')) == 0:
                continue
            len_new_data += 1
            write_dict={'songname':songname, 'singer':singer,'lyric':lyric,'hot_comments':hot_comment}
            f.write(json.dumps(write_dict,ensure_ascii=False)+'\n')

    print(f"    len_all_data:  ", len_all_data)
    print(f"    len_new_data:  ", len_new_data)


def check_chinese(check_str):
    chinese=0
    not_chinese=0
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fa5':  #筛选出来中文
            chinese+=1
        else:
            not_chinese+=1
    return chinese/(chinese+not_chinese)

#  2222222222222222222222
def get_chinese(notnull_path,chinese_path):
    len_all_data=0
    len_new_data=0
    len_new_data1 = 0
    len_new_data2 = 0
    len_new_data3 = 0
    len_new_data4 = 0
    len_new_data5 = 0
    len_new_data6= 0
    len_new_data7 = 0
    len_new_data8 =  0
    len_new_data9 = 0

    with open(chinese_path, "w", encoding='utf-8') as f:
        i=0
        for line in open(notnull_path, "r", encoding='utf-8'):
            i += 1
            if i % 10000 == 0:
                print(i)
            sample = json.loads(line)

            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            hot_comments = sample['hot_comments']

            len_all_data += 1
            #lyric=lyric.replace('\n',' ').replace(' ','')
            lyric_pro=lyric.replace('\n', ' ').replace(' ', '')

            # 外文歌忽略掉
            # if(check_chinese(lyric_pro) >0.8 and check_chinese(lyric_pro) < 1):
            #     print(lyric_pro)
            if check_chinese(lyric_pro) < 0.1:
                continue
            len_new_data1 += 1
            if check_chinese(lyric_pro) < 0.2:
                continue
            len_new_data2 += 1
            if check_chinese(lyric_pro) < 0.3:
                continue
            len_new_data3 += 1
            if check_chinese(lyric_pro) < 0.4:
                continue
            len_new_data4 += 1
            if check_chinese(lyric_pro) < 0.5:
                continue
            len_new_data5 += 1
            if check_chinese(lyric_pro) < 0.6:
                continue
            len_new_data6 += 1
            if check_chinese(lyric_pro) < 0.7:
                continue
            len_new_data7 += 1
            if check_chinese(lyric_pro) < 0.8:
                continue
            len_new_data8 += 1
            # if check_chinese(lyric_pro) < 0.9:
            #     continue
            # len_new_data9 += 1
            # if check_chinese(lyric_pro) < 0.95:
            #     continue
            # if len(hot_comments)<3:
            #     continue
            len_new_data+=1
            write_dict = {'songname': songname, 'singer': singer, 'lyric': lyric, 'hot_comments': hot_comments}
            f.write(json.dumps(write_dict, ensure_ascii=False) + '\n')
        print(f"    len_all_data:  ", len_all_data)
        print(f"    len_new_data:  ", len_new_data)


    # print(f"    len_all_data:  ", len_all_data)
    # print(f"    len_new_data1:  ", len_new_data1)
    # print(f"    len_new_data2:  ", len_new_data2)
    # print(f"    len_new_data3:  ", len_new_data3)
    # print(f"    len_new_data4:  ", len_new_data4)
    # print(f"    len_new_data5:  ", len_new_data5)
    # print(f"    len_new_data6:  ", len_new_data6)
    # print(f"    len_new_data7:  ", len_new_data7)
    # print(f"    len_new_data8:  ", len_new_data8)
    # print(f"    len_new_data9:  ", len_new_data9)
    # print(f"    len_new_data:  ", len_new_data)

'''
    len_all_data:   84172
    len_new_data1:   49509
    len_new_data2:   45443
    len_new_data3:   39425
    len_new_data4:   37151
    len_new_data5:   35152
    len_new_data6:   32618
    len_new_data7:   28897
    len_new_data8:   18132
    len_new_data9:   491
    len_new_data:   3
'''




def get_clear(chinese_path, clear_path):
    len_all_data=0
    meijiule_songname=0
    rest_data=0
    with open(clear_path, "w", encoding='utf-8') as f:
        i=0
        for line in open(chinese_path, "r", encoding='utf-8'):
            i += 1
            if i % 10000 == 0:
                print(i)
            sample = json.loads(line)

            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            hot_comments = sample['hot_comments']

            len_all_data += 1


            '''songname处理格式'''
            songname_pro=songname.replace(' ','')    #(Live)
            #中英文括号内的
            songname_pro = re.sub("（.*）", "", songname_pro)
            songname_pro = re.sub("\(.*\)", "", songname_pro)
            songname_pro = re.sub("（.*\)", "", songname_pro)
            songname_pro = re.sub("\(.*）", "", songname_pro)
            # try:
            #     if check_chinese(songname_pro)<1:
            #         meijiule_songname+=1
            # except:
            #     #print('sommmmmmm         ',songname)
            #     continue

            '''lyric处理格式'''
            # if '\n' not in lyric_pro:
            #     print(lyric_pro)
            lyric = re.sub("\n+", "\n", lyric)
            # lyric_pro = re.sub("作词.*\n", "", lyric_pro)
            # lyric_pro = re.sub("作曲.*\n", "", lyric_pro)
            # lyric_pro = re.sub("混音：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("演唱：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("微博：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("私人微信：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("吉他：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("录音：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("词：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("曲：.*\n", "", lyric_pro)
            # lyric_pro = re.sub("编惊喜：.*\n", "", lyric_pro)

            lyric_list =lyric.split('\n')
            start=0
            while start < len(lyric_list):
                if len(lyric_list[start])<2:
                    start+=1
                    continue
                if '：' in lyric_list[start] or ':' in lyric_list[start]:
                    start+=1
                else:
                    if start+1 < len(lyric_list) and ('：' in lyric_list[start+1] or ':' in lyric_list[start+1] ):
                        start= start + 1
                        continue
                    break

            lyric_list=lyric_list[start:]
            if len(lyric_list)<5:
                continue
            lyric_pro = '\n'.join(lyric_list)
            # if start == 0:
            #     print(i, '-----', ''.join(lyric_list))
            #     rest_data += 1
            #     continue

            # try:
            #     if check_chinese(lyric_pro) < 1 :
            #         print(lyric_pro)
            # except:
            #     continue

            rest_data+=1
            write_dict = {'songname': songname_pro, 'singer': singer, 'lyric': lyric_pro, 'hot_comments': hot_comments}
            f.write(json.dumps(write_dict, ensure_ascii=False) + '\n')

    print(f"    len_all_data:  ", len_all_data)
    print(f"    rest_data:  ", rest_data)






def change2string(clear_path, string_path):
    with open(string_path, "w", encoding='utf-8') as f:
        for line in open(clear_path, "r", encoding='utf-8'):
            sample = json.loads(line)
            songname = sample['songname']
            singer = sample['singer']
            lyric = sample['lyric']
            hot_comment = sample['hot_comments']

            source = ""
            try:
                if check_chinese(songname)>0.8:
                    source = source + '歌名：'+songname +'；'
            except:
                #print(line)
                continue
            if check_chinese(singer) > 0.8:
                source = source + '作者：' + singer + '；'
            #之前版本  歌词多了缩进
            source = source + '歌词：' + ','.join(lyric.split('\n')) + '。'
            if len(source) < 50 :
                continue
            for i in range(min(3, len(hot_comment))):
                f.write(source.replace(' ', ''))
                f.write('\n')
                f.write(hot_comment[i].replace(' ', ''))
                f.write('\n')
                f.write('\n')
    print("finished!")

if __name__ == '__main__':
    file_path = 'netease_music.json'
    notnull_path = 'notnull_data.json'
    chinese_path = 'chinese_data.json'
    clear_path = 'clear_data.json'
    # file_path = 'data//music/netease_music.json'  # 源文档
    # notnull_path = 'data//music/notnull_data.json'
    string_path = 'newnew_data.json'


    '''  处理数据  '''

    '''  处理空数据，提取关键字段  '''
    #get_data(file_path,notnull_path)  #    len_all_data:   172183    len_new_data:   84163

    '''  筛掉外文数据  '''
    #get_chinese(notnull_path, chinese_path)  #    len_all_data:   84163    len_new_data:  30106

    '''  处理格式  '''
    #get_clear(chinese_path, clear_path)  #    len_all_data:   30106    len_new_data:  29496

    '''  处理格式  '''
    change2string(clear_path,string_path)


