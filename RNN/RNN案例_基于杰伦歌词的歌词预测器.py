import torch
import torch.nn as nn
import jieba
import time

from torch.utils.data import DataLoader

# 设备配置：优先cuda，没有就cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 实现步骤：
# 1．获取数据，进行分词，获取词表。
def build_vocab():
    # 1，定义变量，记录：去重后所有的词，每行文本分词结果。
    unique_words,all_words=[],[]
    # 2，遍历数据集，获取到每行文本.
    with open("./data/jaychou_lyrics.txt","r",encoding='utf-8')as f:
        for line in f:
            word=jieba.lcut(line)#[‘想要’，‘有'，‘直升机’，‘n']
            all_words.append(word)
            for word in word:
                if word not in unique_words:
                    unique_words.append(word)
        word_count=len(unique_words)#5703个词（去重后）
        # 构建词表，字典形式，key是词，value是词的索引.
        word_index={word:i for i,word in enumerate(unique_words)}
        # 歌词文本用词表索引表示。
        corpus_idx=[]
        # 遍历每一行的分词结果，将每一行歌词转为词索引
        for words in all_words:
            tmp=[]
            for word in words:
                tmp.append(word_index[word])
            # 在每行词之间，添加空格隔开。
            # tmp.append(word_index[' '])
            # 获取文档中每个词的索引，添加到corpus_idx中.
            corpus_idx.extend(tmp)
    #7.返回结果：唯一词列表（5703个词），词表{想要'：0，‘有’：1‘要大卖’：5702}，（去重后）词的数量，歌词文本用词表索引表示。
    return unique_words,word_index,word_count,corpus_idx

# 2．数据预处理，构建数据集。
# 定义数据集类，继承Dataset
class lyrics_dataset(torch.utils.data.Dataset):
    # 1.初始化词索引，词个数等
    def __init__(self,corpus_idx,num_chars):
        self.corpus_idx=corpus_idx
        # 每次丢入模型训练的词的个数
        self.num_chars=num_chars
        # 文档数据中词的数量，不去重
        self.word_count=len(corpus_idx)
        # 丢入模型的次数
        self.number=self.word_count-self.num_chars-1
    # 2.返回句子数量
    def __len__(self):
        return self.number

    # 3.当使用obj[index]时，自动调用此方法.
    def __getitem__(self,index):
        # idx：指的是词的索引，并将其修正索引值到文档的范围里边，
        # 确保索引start在合法范围内，避免越界，start:当前样本的起始位置
        start=min(max(index,0),self.word_count-self.num_chars-1)
        # 结束索引
        end=start+self.num_chars
        # 取输入值
        x=self.corpus_idx[start:end]
        # 取输出值
        y=self.corpus_idx[start+1:end+1]
        return torch.tensor(x),torch.tensor(y)

# 3，搭建RNN神经网络。
class text_generater(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        self.ebd=nn.Embedding(unique_word_count,128)
        self.rnn=nn.RNN(128,256,1)#词向量维度，隐藏层维度（超参数），网络层数
        self.out=nn.Linear(256,unique_word_count)
    def forward(self,inputs,hidden):
        # embd格式：（batch句子的数量，句子的长度，词向量维度）
        embd=self.ebd(inputs)
        # rnn格式：（句子的长度，batch句子的数量，隐藏层维度）
        output, hidden = self.rnn(embd.transpose(0, 1),hidden)
        # 全连接，输入内容必须是二维数据，即：词的数量*词的维度
        # 输入维度：（seq_len句子数量*batch，词向量维度256）
        # 输出维度：（seq_len句子数量*batch，词表中词的个数）
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 2.4返回结果，预测结果，隐藏层。
        return output,hidden
    def init_hidden(self,bs):
        # 网络层数，batchsize，隐藏层向量维度
        return torch.zeros(1,bs,256,device=device)

# 4．训练模型.
def train():
    # 1.构建词典
    unique_words, word_index, unique_word_count, corpus_idx = build_vocab()
    # 2.获取数据集
    lyrics=lyrics_dataset(corpus_idx,32)
    # 3.初始化网络模型
    model=text_generater(unique_word_count).to(device)
    # 4.创建数据加载器
    lyrics_dataloader=DataLoader(lyrics,batch_size=5,shuffle=True)
    # 5.损失函数
    criterion=nn.CrossEntropyLoss()
    # 6.优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    # 7.模型训练
    epochs=10
    start_time, iter_num, total_loss = time.time(), 0, 0.0
    for epoch in range(epochs):

        for x,y in lyrics_dataloader:
            x = x.to(device)
            y = y.to(device)
            # 隐藏层初始化
            hidden=model.init_hidden(x.shape[0])
            output,hidden=model(x,hidden)
            # y的形状：（batch批次数，seq_len句子长度，词向量维度）→转成一维向量→每个词的下标索引：
            # output形状为：（seq_len，batch，词向量维度）
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss=criterion(output,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            iter_num+=1
        print(f'epoch:{epoch + 1}，time:{time.time() - start_time:.2f}s,loss:{total_loss / iter_num:.4f}')
        torch.save(model.state_dict(),"./text_model.pth")
# 5．模型预测。
def evaluate(start_word,sentence_length):
    # 1.构建词典
    unique_words, word_index, unique_word_count, corpus_idx = build_vocab()
    # 2.获取模型
    model=text_generater(unique_word_count).to(device)
    # 3.加载模型参数
    model.load_state_dict(torch.load('./text_model.pth'))
    # 4.获取隐藏层初始值
    hidden=model.init_hidden(1)
    # 5.将输入文字转换为索引
    word_idx=word_index[start_word]
    # 6.定义列表存放预测的词的索引
    generate_sentence=[word_idx]
    # 7.循环遍历句子长度，获取每一个词
    for i in range(sentence_length):
        output,hidden=model(torch.tensor([[word_idx]]).to(device),hidden)
        word_idx=torch.argmax(output).item()
        generate_sentence.append(word_idx)
    print("\n")
    # 8.将索引转为歌词
    for idx in generate_sentence:
        print(unique_words[idx],end='')

# 6.测试
if __name__=='__main__':
    train()
    
    input_word=input("请输入开始歌词：")
    evaluate(input_word,50)
