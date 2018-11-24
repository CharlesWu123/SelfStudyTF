import codecs
import sys


MODE = "TRANSLATE_EN"    # 将MODE设置为"PTB_TRAIN", "PTB_VALID", "PTB_TEST", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB_TRAIN":        # PTB训练数据
    RAW_DATA = "../data/simple-examples/data/ptb.train.txt"  # 训练集数据文件
    VOCAB = "ptb.vocab"                                 # 词汇表文件
    OUTPUT_DATA = "ptb.train"                           # 将单词替换为单词编号后的输出文件
elif MODE == "PTB_VALID":      # PTB验证数据
    RAW_DATA = "../data/simple-examples/data/ptb.valid.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.valid"
elif MODE == "PTB_TEST":       # PTB测试数据
    RAW_DATA = "../data/simple-examples/data/ptb.test.txt"
    VOCAB = "ptb.vocab"
    OUTPUT_DATA = "ptb.test"
elif MODE == "TRANSLATE_ZH":   # 中文翻译数据
    RAW_DATA = "../data/TED_data/train.txt.zh"
    VOCAB = "./data/zh.vocab"
    OUTPUT_DATA = "./data/train.zh"
elif MODE == "TRANSLATE_EN":   # 英文翻译数据
    RAW_DATA = "../data/TED_data/train.txt.en"
    VOCAB = "./data/en.vocab"
    OUTPUT_DATA = "./data/train.en"



# 读取词汇表，并简历词汇到单词编号的映射
with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 如果出现了被删除的低频词，则替换为'<unk>'
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']  # 读取单词并添加<eos>结束符
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()