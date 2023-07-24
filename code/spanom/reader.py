import torch
import numpy as np
#from sortedcontainers import SortedSet
LABEL_TO_ID={'O':0,'TARGET':1, "AGENT":2}
ID_TO_LABEL=['O','TARGET', "AGENT"]
EMO_TO_ID={'O':0,'negative':1, "positive":2}
ID_TO_EMO=['O','negative', "positive"]
class srl_sentence():
    def __init__(self, obj):
        self.sentences = obj["sentences"]
        self.srl = obj["orl"]
        self.dse = []
        self.argus = []
        self.emo=dict()
        self.reset_sentence()
        for srl_label in self.srl:
            label= srl_label[-1]
            if label== "DSE":
                self.dse.append(srl_label)
            elif label in ['AGENT','TARGET']:
                self.argus.append(srl_label)
            elif label in ['negative','positive']:
                self.emo[(srl_label[0],srl_label[1])]=EMO_TO_ID[srl_label[-1]]

    def reset_sentence(self):
        for i in range(len(self.sentences)):
            if self.sentences[i] in ["[", "]", "(", ")", "{", "}", "-LRB-", "-RRB-", "-LSB-", "-RSB-", "-LCB-", "-RCB-"]:
                self.sentences[i] = '-'



    def tokenize_argument_spans(self):
        srl_span = []
        for srl_label in self.srl:  # remove self-loop V-V
            # print(srl_label)
            if srl_label[-1] in ['AGENT','TARGET']:
                srl_span.append([int(srl_label[0]), int(srl_label[1]),
                                 int(srl_label[2]), int(srl_label[3]),
                                 int(LABEL_TO_ID[srl_label[4]])])
        if len(srl_span) == 0:  # if the sentence has no arguments.
            return [[], [], [], [], []]
        tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = \
            zip(*srl_span)
        tmp = list(zip(tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends,
                       tokenized_arg_labels))
        #orl_span = list(SortedSet(tmp))
        orl_span = list(set(tmp))  # remove the same [dse_start, dse_end, x_start, x_end, label]
        tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels = \
            zip(*orl_span)
        return tokenized_dse_starts, tokenized_dse_ends, tokenized_arg_starts, tokenized_arg_ends, tokenized_arg_labels

    def tokenize_dse_spans(self):
        dse_span = []
        for dse in self.dse:
            dse_start, dse_end, _, _, dse_label = dse
            assert dse_label == "DSE"
            dse_span.append([int(dse_start), int(dse_end)])
        tokenized_dse_starts, tokenized_dse_ends = zip(*dse_span)
        dse_span = list(zip(tokenized_dse_starts, tokenized_dse_ends))
        dse_span = list(set(dse_span))
        tokenized_dse_starts, tokenized_dse_ends = zip(*dse_span)
        return tokenized_dse_starts, tokenized_dse_ends

    def tokenize_arg_spans(self):
        arg_span = []
        for arg in self.argus:
            _, _, arg_starts, arg_ends, label = arg
            assert label in ["TARGET", "AGENT"]
            arg_span.append([int(arg_starts), int(arg_ends)])
        if len(arg_span) == 0:
            return [], []
        tokenized_arg_starts, tokenized_arg_ends = zip(*arg_span)
        arg_span = list(zip(tokenized_arg_starts, tokenized_arg_ends))
        arg_span = list(set(arg_span))
        tokenized_arg_starts, tokenized_arg_ends = zip(*arg_span)
        return tokenized_arg_starts, tokenized_arg_ends
def tokenize_data(data):
    """
    :param data: the raw input sentences
    :param word_dict: word dictionary
    :param char_dict: character dictionary
    :param label_dict: srl label dictionary
    :param lowercase: bool value, if the word or character needs to lower
    :param pretrained_word_embedding: pre-trained word embedding
    :return: a list storing the [sentence id, length, [words], [heads], [characters], [srl argument spans]]
    """
    sample_sentence_words = [sent.sentences for sent in data]
    
    
    sample_lengths = [len(sent.sentences)for sent in data]
    sample_orl_span_tokens = [sent.tokenize_argument_spans() for sent in data]
    sample_dse_span_tokens = [sent.tokenize_dse_spans() for sent in data]
    sample_arg_span_tokens = [sent.tokenize_arg_spans() for sent in data]
   
    sample_lengths = np.array(sample_lengths)
   
    sample_orl_span_tokens = np.array(sample_orl_span_tokens)
    sample_dse_span_tokens = np.array(sample_dse_span_tokens)
    sample_arg_span_tokens = np.array(sample_arg_span_tokens)
   
    return list(zip(sample_lengths, sample_orl_span_tokens, sample_dse_span_tokens, sample_arg_span_tokens, sample_sentence_words))
def tensorize_batch_samples(samples):
        """
        tensorize the batch samples
        :param samples: List of samples
        :return: tensorized batch samples
        """
        batch_sample_size = len(samples)#batch_size
        max_sample_length = max([sam[0] for sam in samples])
        max_sample_word_length = max([sam[2].shape[1] for sam in samples])

        max_sample_orl_number = max([len(sam[3][0]) for sam in samples])
        max_sample_dse_number = max([len(sam[4][0]) for sam in samples])
        max_sample_arg_number = max([len(sam[5][0]) for sam in samples])

        # input
        padded_sample_lengths = np.zeros(batch_sample_size, dtype=np.int64)
        padded_word_tokens = np.zeros([batch_sample_size, max_sample_length], dtype=np.int64)
        padded_char_tokens = np.zeros([batch_sample_size, max_sample_length, max_sample_word_length], dtype=np.int64)
        # gold dse
        padded_gold_dses_starts = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
        padded_gold_dses_ends = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
        padded_num_gold_dses = np.zeros(batch_sample_size, dtype=np.int64)
        padded_gold_dses_starts_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_gold_dses_ends_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        # gold arg
        padded_gold_argus_starts = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_gold_argus_ends = np.zeros([batch_sample_size, max_sample_arg_number], dtype=np.int64)
        padded_num_gold_argus = np.zeros(batch_sample_size, dtype=np.int64)
        padded_gold_argus_starts_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)
        padded_gold_argus_ends_one_hot = np.zeros([batch_sample_size, max_sample_length], dtype=np.int)

        # if max_sample_arg_number == 0:
        #     max_sample_arg_number = 1
        padded_orl_dses_starts = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_dses_ends = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_argus_starts = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_argus_ends = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_labels = np.zeros([batch_sample_size, max_sample_orl_number], dtype=np.int64)
        padded_orl_nums = np.zeros([batch_sample_size], dtype=np.int64)

        sample_words = []
        for i, sample in enumerate(samples):
            sample_length = sample[0]
            padded_sample_lengths[i] = sample_length
            # input
            padded_word_tokens[i, :sample_length] = sample[1]
            padded_char_tokens[i, :sample[2].shape[0], : sample[2].shape[1]] = sample[2]
            # gold dse
            sample_dse_number = len(sample[4][0])
            padded_gold_dses_starts[i, :sample_dse_number] = sample[4][0]
            padded_gold_dses_ends[i, :sample_dse_number] = sample[4][1]
            padded_num_gold_dses[i] = sample_dse_number
            padded_gold_dses_starts_one_hot[i][list(sample[4][0])] = 1
            padded_gold_dses_ends_one_hot[i][list(sample[4][1])] = 1
            # gold arg
            sample_arg_number = len(sample[5][0])
            padded_gold_argus_starts[i, :sample_arg_number] = sample[5][0]
            padded_gold_argus_ends[i, :sample_arg_number] = sample[5][1]
            padded_num_gold_argus[i] = sample_arg_number
            padded_gold_argus_starts_one_hot[i][list(sample[5][0])] = 1
            padded_gold_argus_ends_one_hot[i][list(sample[5][1])] = 1
            # output
            sample_orl_number = len(sample[3][0])
            padded_orl_dses_starts[i, :sample_orl_number] = sample[3][0]
            padded_orl_dses_ends[i, :sample_orl_number] = sample[3][1]
            padded_orl_argus_starts[i, :sample_orl_number] = sample[3][2]
            padded_orl_argus_ends[i, :sample_orl_number] = sample[3][3]
            padded_orl_labels[i, :sample_orl_number] = sample[3][4]
            padded_orl_nums[i] = sample_orl_number

            sample_words.append(sample[-1])
        return torch.from_numpy(padded_sample_lengths),\
                torch.from_numpy(padded_word_tokens), torch.from_numpy(padded_char_tokens),\
                torch.from_numpy(padded_gold_dses_starts), torch.from_numpy(padded_gold_dses_ends),\
                torch.from_numpy(padded_num_gold_dses),\
                torch.from_numpy(padded_gold_dses_starts_one_hot), torch.from_numpy(padded_gold_dses_ends_one_hot),\
                torch.from_numpy(padded_gold_argus_starts), torch.from_numpy(padded_gold_argus_ends),\
                torch.from_numpy(padded_num_gold_argus),\
                torch.from_numpy(padded_gold_argus_starts_one_hot), torch.from_numpy(padded_gold_argus_ends_one_hot),\
                torch.from_numpy(padded_orl_dses_starts), torch.from_numpy(padded_orl_dses_ends),\
                torch.from_numpy(padded_orl_argus_starts), torch.from_numpy(padded_orl_argus_ends),\
                torch.from_numpy(padded_orl_labels), torch.from_numpy(padded_orl_nums), \
               sample_words
def get_dse_goldens(samples):
    batch_sample_size = len(samples)#batch_size
    max_sample_dse_number = max([len(sam[2][0]) for sam in samples])
    padded_gold_dses_starts = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
    padded_gold_dses_ends = np.zeros([batch_sample_size, max_sample_dse_number], dtype=np.int64)
    padded_num_gold_dses = np.zeros(batch_sample_size, dtype=np.int64)
    for i, sample in enumerate(samples):
        sample_dse_number = len(sample[2][0])
        padded_gold_dses_starts[i, :sample_dse_number] = sample[2][0]
        padded_gold_dses_ends[i, :sample_dse_number] = sample[2][1]
        padded_num_gold_dses[i] = sample_dse_number
    return torch.from_numpy(padded_gold_dses_starts), torch.from_numpy(padded_gold_dses_ends),\
            torch.from_numpy(padded_num_gold_dses)
            
def get_arg_goldens(samples):
    return None