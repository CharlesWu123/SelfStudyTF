import codecs
import collections
from operator import itemgetter

MODE = "TRANSLATE_EN"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "../data/simple-examples/data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "../data/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "./data/zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "../data/TED_data/train.txt.en"
    VOCAB_OUTPUT = "./data/en.vocab"
    VOCAB_SIZE = 10000

counter = collections.Counter()     # 统计单词出现频率
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
# 需要在文本换行处加入句子结束符'<eos>'
# sorted_words = ['<eos>'] + sorted_words
# 在PTB中，输入数据已经将低频词汇替换成了 '<unk>',因此不需要这一步骤
sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
if len(sorted_words) > VOCAB_SIZE:
    sorted_words = sorted_words[:VOCAB_SIZE]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
















