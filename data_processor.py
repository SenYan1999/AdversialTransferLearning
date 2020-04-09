import os
import json
import random
from utils import logger
from args import args
import sys
sys.path.append('..')
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer

def init_params():
    processors = {"bert_ner": MyPro}
    task_name = 'bert_ner'
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    tokenizer = BertTokenizer.from_pretrained('pretrained_model/vocab.txt')
    return processor, tokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """创建一个输入实例
        Args:
            guid: 每个example拥有唯一的id
            text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
            text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
            label: example对应的标签，对于训练集和验证集应非None，测试集为None
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.output_mask = output_mask
        self.guid = guid


class DataProcessor(object):
    """数据预处理的基类，自定义的MyPro继承该类"""

    def get_train_examples(self, data_dir):
        """读取训练集 Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """读取验证集 Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """读取标签 Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        jsons=[]
        with open(input_file, 'r', encoding='utf8') as f:
            for line_ in f:
                line=json.loads(line_)
                jsons.append(line)
            return jsons



class MyPro(DataProcessor):
    """将数据构造成example格式"""

    def _create_example(self, lines):
        examples = []
        for i, line in enumerate(lines):
            sent =  line['source'].split()

            if 'target' in line:
                label = line['target'].split()
            else:
                label = ['O'] * len(sent)
            guid = i
            assert len(sent) == len(label)
            example = InputExample(guid=guid, text_a=sent, label=label)
            examples.append(example)
        return examples

    def get_train_examples(self, data_mode):
        trin_dir=os.path.join(args.data_dir, data_mode, 'train.json')
        lines = self._read_json(trin_dir)
        examples = self._create_example(lines)
        return examples

    def get_dev_examples(self, data_mode):
        trin_dir = os.path.join(args.data_dir, data_mode, 'dev.json')
        lines = self._read_json(trin_dir)
        examples = self._create_example(lines)
        return examples

    def get_test_examples(self, data_mode):
        trin_dir = os.path.join(args.data_dir, data_mode, 'test.json')
        lines = self._read_json(trin_dir)
        examples = self._create_example(lines)
        return examples

    def get_pre_examples(self, data_mode):
        trin_dir = os.path.join(args.data_dir, data_mode, 'pre.json')
        lines = self._read_json(trin_dir)
        examples = self._create_example(lines)
        return examples

    def get_labels(self):
        labels = []
        with open(os.path.join(args.data_dir,'word_class_dic_revised.txt'), encoding='utf8',errors='ignore') as word_class_file:
            for line in word_class_file:
                word_class_items = line.strip().split('\t')
                labels.append(word_class_items[0])
        return labels


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        idx = example.guid
        labellist = example.label
        textlist = example.text_a

        tokens_a = []
        labels = []
        for i, word in enumerate(textlist):
            # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
            token = tokenizer.tokenize(word)
            tokens_a.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:  # 一般不会出现else
                    print("bert分字异常")
                    raise Exception

        assert len(tokens_a) == len(labels)

        if len(tokens_a) == 0 or len(labels) == 0:
            continue

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels = labels[:(max_seq_length - 2)]
        # ----------------处理source--------------
        ## 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        labels = ["[CLS]"] + labels + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        ## 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        label_id = [label_map[l] for l in labels]

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        assert len(label_id) == max_seq_length

        output_mask = [1 for t in tokens_a]
        output_mask = [1] + output_mask + [1]
        output_mask += padding
        assert len(output_mask) == max_seq_length

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------

        if ex_index < 1:
            logger.info("-----------------Example-----------------")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("label: %s " % " ".join([str(x) for x in label_id]))
            logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
        # ----------------------------------------------------

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               output_mask=output_mask,
                               guid=idx
                               )
        features.append(feature)

    return features


def create_batch_iter(mode, datamode):
    """构造迭代器"""
    processor, tokenizer = init_params()

    if mode == "train":
        examples = processor.get_train_examples(datamode)  # [InputExample1(注,属性有：guid,text_a（str类型，空格分开）,text_b（str类型，空格分开）,label),InputExample2,....]

        num_train_steps = int(
            len(examples) / args.batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = args.batch_size

        logger.info("  Num steps = %d", num_train_steps)

    elif mode == "dev":
        examples = processor.get_dev_examples(datamode)
        batch_size = args.batch_size
    elif mode == "test":
        examples = processor.get_test_examples(datamode)
        batch_size = args.batch_size
    elif mode == "predict":
        examples = processor.get_pre_examples(datamode)
        batch_size = args.batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    label_list = processor.get_labels()

    # 特征[InputFeature1(属性input_ids,input_mask,segment_ids,label_idoutput_mask), InputFeature2]
    features = convert_examples_to_features(examples, label_list, args.max_len, tokenizer)

    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)

    all_input_guid = torch.tensor([f.guid for f in features], dtype=torch.int)
    # all_input_text= torch.tensor([f.tokens for f in features], dtype=torch.char)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_guid,all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)

    if mode == "train":
        sampler = RandomSampler(data)
    elif mode == "dev" or mode == "test" or 'predict':
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator
    elif mode == "dev" :
        return iterator
    elif mode=='predict'or mode == "test":
        return iterator, examples
    else:
        raise ValueError("Invalid mode %s" % mode)


if __name__ == '__main__':
    source_train_iter = create_batch_iter(mode="train", datamode='source')
    source_dev_iter = create_batch_iter(mode="dev", datamode='source')
    source_test_iter, source_test_examples = create_batch_iter(mode="test", datamode='source')
    target_train_iter = create_batch_iter(mode="train", datamode='target')
    target_dev_iter = create_batch_iter(mode="dev", datamode='target')
    target_test_iter, target_test_examples = create_batch_iter(mode="test", datamode='target')
