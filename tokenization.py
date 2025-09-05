import regex as re
from collections import defaultdict

class tokenizer_train():
    def find_merge(self,utf8_encode_list:list,count_dict:dict):
        '''
        本方法的作用就是两两组合列表内容，找出出现次数最多的组合，如果有并列就取字符最大的那个
        :param utf8_encode_list:为每次需要选出需要合并的初始列表
        :return: 返回最终要合并在一起的组合
        '''
        num=0
        max_value=0
        max_list=[]
        pair_freqs=defaultdict(int)
        for word,freq in count_dict.items():
            split=utf8_encode_list[num][word]
            if len(split)!=1:
                for i in range(len(split)-1):
                    pre_merge=(split[i],split[i+1])
                    pair_freqs[pre_merge]+=freq
                    if pair_freqs[pre_merge]>max_value:
                        max_value=pair_freqs[pre_merge]
            num+=1
        #print(pre_merge_dict)

        for i in pair_freqs.keys():
            if pair_freqs[i]==max_value:
                max_list.append(i)
        return [max(max_list),max_value]

    def change_list(self,begin_list:list[{str:list}],word_dict:dict,useful_tuple:tuple)->list:
        '''
        此函数的作用就是在得到出现次数最多的组合之后，在原始列表中替换掉他们
        :param begin_list:待合并的列表
        :param useful_tuple: 需要合并的两个元素的元组
        :param times 调用过几次此函数
        :return: 删除更改后的列表
        '''
        num=0
        for word,frq in word_dict.items():
            waitting_delete_list = []
            split=begin_list[num][word]
            for i in range(len(split)-1):
                if split[i]==useful_tuple[0] and split[i+1]==useful_tuple[1]:
                    split[i]=useful_tuple[0]+useful_tuple[1]
                    waitting_delete_list.append(i)
            for i in waitting_delete_list:
                split.pop(i+1)
            begin_list[num][word]=split
            num+=1

        return begin_list

    def train_BPE_tokenizar(self,path:str,voc:dict):
        '''
        此函数为字节级BPE分词器的训练函数
        :param path: 文本文件的存储地址
        :param voc: 已有的词汇表
        :return: 返回经过训练补充的词汇表以及文本内容的分词情况，数据类型为list[dict{str:list}]
        '''
        #加载语料库
        data=open(path,encoding="utf-8")
        text=data.read()
        data.close()
        #正则化分词
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        word_dict = {}      #用来存储每个词汇的出现次数
        word_list=[]
        list_re = re.findall(PAT,text)
        #分词后统计每个词汇出现的出现频率
        for i in list_re:
            try:
                word_dict[i] += 1
            except:
                word_dict[i] = 1
        #将每个词分解成字符
        for i in word_dict.keys():
            word_list.append({i:list(i)})
        #统计每个组合出现的次数，找到频率最高的组合，对原word_list进行修改，将合并的字符记入词汇表
        change_times=1
        while True:
            useful_list=self.find_merge(word_list,word_dict)
            if useful_list[1]==1:
                break
            else:
                word_list=self.change_list(word_list,word_dict,useful_list[0])      #完成对原列表的修改，将分开的组合进行合并
                voc[useful_list[0][0]+useful_list[0][1]]=256+change_times       #写入词汇表
                change_times+=1
        return voc,word_list

if __name__=="__main__":
    # 初始化词汇表
    voc = {bytes([i]): i for i in range(256)}
    voc['<|endoftext|>'] = 256
    path="D:\\abc.txt"
    fist_tokenizer=tokenizer_train()
    voc_final,final_list=fist_tokenizer.train_BPE_tokenizar(path,voc)
    print(f"最终词汇表为：{voc_final}\n最终分词情况：{final_list}")





