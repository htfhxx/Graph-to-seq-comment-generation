import os
import json
import sys
import numpy as np
import torch
import random
import copy
from util.nlp_utils import split_chinese_sentence, remove_stopwords
from util.utils import bow
import pickle

PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4
TITLE = 5
MAX_LENGTH = 300


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=50000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        #if not os.path.exists(vocab_file):
        self.build_vocab(content_file, vocab_file)
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

    @staticmethod
    def build_vocab(corpus_file, vocab_file):
        word2count = {}
        for line in open(corpus_file,'r',encoding='utf-8'):
            line=line.replace(' ','')
            for word in line:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
        word2count = list(word2count.items())
        word2count.sort(key=lambda k: k[1], reverse=True)
        write = open(vocab_file, 'w',encoding='utf-8')
        for word_pair in word2count:
            write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
        write.close()

    def load_vocab(self, vocab_file, vocab_size):
        for line in open(vocab_file,'r',encoding='utf-8'):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result


class Example:
    """
    Each example is one data pair
        src: title (has oov)
        tgt: comment (oov has extend ids if use_oov else has oov)
        memory: tag (oov has extend ids)
    """

    def __init__(self, source,target,  vocab):
        self.ori_source= list(source)
        self.ori_target = list(target)

        self.ori_sentence_source = split_chinese_sentence(self.ori_source)
        self.sentence_source = [vocab.sent2id(sentence) for sentence in
                                 self.ori_sentence_source]
        self.ori_sentence_target = split_chinese_sentence(self.ori_target)
        self.sentence_target = [vocab.sent2id(sentence) for sentence in
                                 self.ori_sentence_target]

        self.source = vocab.sent2id(self.ori_source, add_start=True, add_end=True)
        self.target = vocab.sent2id(self.ori_target, add_start=True, add_end=True)



class Batch:
    """
    Each batch is a mini-batch of data

    """

    def __init__(self, example_list):
        max_len = MAX_LENGTH
        self.examples = example_list

        self.src_len = self.get_length([e.source for e in example_list], max_len)
        self.src, self.src_mask = self.padding(
            [e.source for e in example_list],
            max(self.src_len))

        self.tgt_len = self.get_length([e.target for e in example_list], max_len)
        max_tgt_len = max(self.tgt_len)
        batch_tgt, self.tgt_mask = self.padding([e.target for e in example_list], max_tgt_len)
        self.tgt = np.array(batch_tgt, dtype=np.long)

        self.to_tensor()

    def get_length(self, examples, max_len):
        length = []
        for e in examples:
            length.append(min(len(e),max_len))
        assert len(length) == len(examples)
        return length

    def to_tensor(self):
        self.src = torch.from_numpy(np.array(self.src, dtype=np.long))
        self.src_len = torch.from_numpy(np.array(self.src_len, dtype=np.long))
        self.src_mask = torch.from_numpy(np.array(self.src_mask, dtype=np.long))

        self.tgt = torch.from_numpy(self.tgt)
        self.tgt_len = torch.from_numpy(np.array(self.tgt_len, dtype=np.long))
        self.tgt_mask = torch.from_numpy(np.array(self.tgt_mask, dtype=np.int32))
    @staticmethod
    def padding(batch, max_len, limit_length=True):
        if limit_length:
            max_len = min(max_len, MAX_LENGTH)
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(l) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(m)
        return result, mask_batch


class DataLoader:
    def __init__(self, config, data_path, batch_size, vocab,  model, no_train=False, debug=False):
        assert MAX_LENGTH == config.max_sentence_len, (MAX_LENGTH, config.max_sentence_len)
        self.debug = debug
        self.vocab = vocab
        self.batch_size = batch_size

        self.all_data = self.read_json(os.path.join(data_path, 'newnew_data.json'),vocab)
        # train_data  =[Example1, Example2, ...]
        # Example1 .source
        # print('self.all_data[0].ori_source:  ',  self.all_data[0].ori_source)
        # print('self.all_data[1].ori_source:  ', self.all_data[1].ori_source)
        # print('self.all_data[2].ori_source:  ', self.all_data[2].ori_source)
        # print('self.all_data[0].ori_target:  ',  self.all_data[0].ori_target)
        # print('self.all_data[1].ori_target:  ', self.all_data[1].ori_target)
        # print('self.all_data[2].ori_target:  ', self.all_data[2].ori_target)
        #
        # print('self.all_data[0].source:  ',  self.all_data[0].source)
        # print('self.all_data[1].source:  ', self.all_data[1].source)
        # print('self.all_data[2].source:  ', self.all_data[2].source)
        #
        # print('self.all_data[0].target:  ',  self.all_data[0].target)
        # print('self.all_data[1].target:  ', self.all_data[1].target)
        # print('self.all_data[2].target:  ', self.all_data[2].target)
        self.train_data, self.dev_data, self.test_data = self.split_data(self.all_data)
        print('self.train_data[0].ori_source:  ',  self.train_data[0].ori_source)
        print('self.train_data[1].ori_source:  ', self.train_data[1].ori_source)
        print('self.train_data[2].ori_source:  ', self.train_data[2].ori_source)
        print('self.dev_data[1].ori_source:  ', self.dev_data[1].ori_source)
        print('self.test_data[2].ori_source:  ', self.test_data[2].ori_source)

        print('self.train_data[0].ori_target:  ',  self.train_data[0].ori_target)
        print('self.train_data[1].ori_target:  ', self.train_data[1].ori_target)
        print('self.train_data[2].ori_target:  ', self.train_data[2].ori_target)
        print('self.dev_data[1].ori_target:  ', self.dev_data[1].ori_target)
        print('self.test_data[2].ori_target:  ', self.test_data[2].ori_target)

        print('self.train_data[0].source:  ',  self.train_data[0].source)
        print('self.train_data[1].source:  ', self.train_data[1].source)
        print('self.train_data[2].source:  ', self.train_data[2].source)
        print('self.dev_data[1].source:  ', self.dev_data[1].source)
        print('self.test_data[2].source:  ', self.test_data[2].source)

        print('self.train_data[0].target:  ',  self.train_data[0].target)
        print('self.train_data[1].target:  ', self.train_data[1].target)
        print('self.train_data[2].target:  ', self.train_data[2].target)
        print('self.dev_data[1].target:  ', self.dev_data[1].target)
        print('self.test_data[2].target:  ', self.test_data[2].target)

        self.train_batches = self.make_batch(self.train_data, batch_size)
        self.dev_batches = self.make_batch(self.dev_data, batch_size)
        self.test_batches = self.make_batch(self.test_data, batch_size)
        # print('self.test_batches[0].src:  ', self.test_batches[0].src)
        # print('self.test_batches[0].tgt:  ', self.test_batches[0].tgt)
        # print('self.test_batches[1].src:  ', self.test_batches[1].src)
        # print('self.test_batches[1].tgt:  ', self.test_batches[1].tgt)
        # print('self.test_batches[2].src:  ', self.test_batches[2].src)
        # print('self.test_batches[2].tgt:  ', self.test_batches[2].tgt)

        random.shuffle(self.train_batches)

    @staticmethod
    def split_data(data):
        total_num = len(data)
        train = data[:round(0.9 * total_num)]
        dev = data[round(0.9 * total_num):round(0.95 * total_num)]
        test = data[round(0.95 * total_num):]
        return train, dev, test

    def read_json(self, filename,vocab):
        result = []
        f=open(filename, 'r', encoding='utf-8').read()
        data_list=f.split('\n\n')

        #with open('data/data3.txt','w',encoding='utf-8') as f2:
        for i in range(len(data_list)):
            sample_list=data_list[i].split('\n')
            if i>500:
                break
            if len(sample_list) !=2:
                #print(sample_list)
                source = ''.join(sample_list[:-2])
                target = sample_list[-1]
                # print('source:   ',source)
                # print('target:   ',target)
            else:
                source = sample_list[0]
                try:
                    target = sample_list[1]
                except:
                    continue
            # print('source:   ',source)
            # print('target:   ',target)
            # f2.write('-------------------------------------------------\n')
            # f2.write(source+'\n')
            # f2.write(target+'\n')

            e = Example(source,target,vocab)
            result.append(e)
        return result

    def make_batch(self, data, batch_size):
        batches = []
        for i in range(0, len(data), batch_size):
            batches.append(Batch(data[i:i + batch_size]))
        return batches


def data_stats(fname, is_test):
    content_word_num = []
    content_char_num = []
    title_word_num = []
    title_char_num = []
    comment_word_num = []
    comment_char_num = []
    keyword_num = []
    urls = {}

    for line in open(fname, "r"):
        g = json.loads(line)
        url = g["url"]
        if url not in urls:
            urls[url] = 0
        if is_test:
            targets = [s.split() for s in g["label"].split("$$")]
            urls[url] += len(targets)
            for target in targets:
                comment_word_num.append(len(target))
                comment_char_num.append(len("".join(target)))
        else:
            urls[url] += 1
            target = g["label"].split()
            comment_word_num.append(len(target))
            comment_char_num.append(len("".join(target)))
        title = g["title"].split()
        title_word_num.append(len(title))
        title_char_num.append(len("".join(title)))
        original_content = g["text"].split()
        content_word_num.append(len(original_content))
        content_char_num.append(len("".join(original_content)))

        # betweenness = g["g_vertices_betweenness_vec"]
        # pagerank = g["g_vertices_pagerank_vec"]
        # katz = g["g_vertices_katz_vec"]
        concept_names = g["v_names"]
        keyword_num.append(len(concept_names))
        text_features = g["v_text_features_mat"]
        content = []

        adj_numsent = g["adj_mat_numsent"]
        # adj_numsent is a list(list)
        adj_tfidf = g["adj_mat_tfidf"]
    print('number of documents', len(urls))
    print('number of total comments', sum(list(urls.values())))
    print('average number of comments', np.mean(list(urls.values())))
    content_word_num = np.mean(content_word_num)
    content_char_num = np.mean(content_char_num)
    title_word_num = np.mean(title_word_num)
    title_char_num = np.mean(title_char_num)
    comment_word_num = np.mean(comment_word_num)
    comment_char_num = np.mean(comment_char_num)
    keyword_num = np.mean(keyword_num)
    print(
        'average content word number: %.2f, average content character number: %.2f, average title word number: %.2f, '
        % (content_word_num, content_char_num, title_word_num),
        'average title character numerb: %.2f, average comment word number %.2f, average comment character number %.2f'
        % (title_char_num, comment_word_num, comment_char_num),
        'average keyword number %.2f' % keyword_num)


def eval_bow(feature_file, cand_file):
    stop_words = {word.strip() for word in open('../data/stop_words.txt').readlines()}
    contents = []
    for line in open(feature_file, "r"):
        g = json.loads(line)
        contents.append(remove_stopwords(g["text"].split(), stop_words))
    candidates = []
    for line in open(cand_file):
        words = line.strip().split()
        candidates.append(remove_stopwords(words, stop_words))
    assert len(contents) == len(candidates), (len(contents), len(candidates))
    results = []
    for content, candidate in zip(contents, candidates):
        results.append(cosine_sim(bow(content), bow(candidate)))
    return results, np.mean(results)


def eval_unique_words(cand_file):
    stop_words = {word.strip() for word in open('../data/stop_words.txt').readlines()}
    result = set()
    for line in open(cand_file):
        words = set(line.strip().split())
        result.update(words)
    result = result.difference(stop_words)
    return result


def eval_distinct(cand_file):
    unigram, bigram, trigram = set(), set(), set()
    sentence = set()
    for line in open(cand_file):
        words = line.strip().split()
        sentence.add(line)
        unigram.update(set(words))
        for i in range(len(words) - 1):
            bigram.add((words[i], words[i + 1]))
        for i in range(len(words) - 2):
            trigram.add((words[i], words[i + 1], words[i + 2]))
    return unigram, bigram, trigram, sentence


if __name__ == '__main__':
    '''
    print('entertainment')
    data_stats('./data/train_graph_features.json', False)
    data_stats('./data/dev_graph_features.json', True)
    print('sport')
    data_stats('./sport_data/train_graph_features.json', False)
    data_stats('./sport_data/dev_graph_features.json', True)
    '''
    topic = sys.argv[1]
    cand_log = sys.argv[2]
    # print(eval_bow(os.path.join(topic, 'dev_graph_features.json'), os.path.join(topic, 'log', cand_log, 'candidate.txt'))[1])
    unigram, bigram, trigram, sentence = eval_distinct(os.path.join(topic, 'log', cand_log, 'candidate.txt'))
    print('unigram', len(unigram), 'bigram', len(bigram), 'trigram', len(trigram), 'sentence', len(sentence))
    print(len(eval_unique_words(os.path.join(topic, 'log', cand_log, 'candidate.txt'))))
