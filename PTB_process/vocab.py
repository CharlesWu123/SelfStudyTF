import codecs
import collections
from operator import itemgetter

RAW_DATA = '../data/simple-examples/data/ptb.train.txt'  # 训练集数据文件
VACAB_OUTPUT = 'ptb.vocab'  # 输出的词汇表文件

counter = collections.Counter()     # 统计单词出现频率
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
# 需要在文本换行处加入句子结束符'<eos>'
sorted_words = ['<eos>'] + sorted_words
# 在PTB中，输入数据已经将低频词汇替换成了 '<unk>',因此不需要这一步骤
# sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[:10000]

with codecs.open(VACAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
















