import json
import os
import nltk
import csv
import codecs
import argparse

from os import walk

# read a json file
def read_json(filename):
    with open(filename, "r") as inp:
        data = json.load(inp)
    return data

def parse_wiki(dir_name):
    wiki_data = {}
    for (dirn, _, filenames) in walk(dir_name):
        for fname in filenames:
            curr_data = read_json(dirn+'/'+fname)
            wiki_data[curr_data["wikiDocumentIdx"]] = curr_data

    return wiki_data

def process_sec_0(sec):
    sec_sents = []

    mov_str = sec["movieName"] 
    mov_str = 'The movie we are discussing today is ' + mov_str + '.'
    sec_sents.append(mov_str)

    int_str = sec["introduction"]
    int_sent = nltk.sent_tokenize(int_str)
    sec_sents.extend(int_sent)

    dir_str = sec["director"]
    gen_str = sec["genre"]
    tmp_str = 'The movie is directed by ' + dir_str + ' and belongs to ' \
                + gen_str + ' genre.'
    sec_sents.append(tmp_str)

    cas_lis = sec["cast"]
    cast_str = ', '.join(cas_lis[:-1])
    cast_str += ' and ' + cas_lis[-1] + '.'
    cast_str = 'The cast is ' + cast_str
    sec_sents.append(cast_str)

    cri_sen = sec["critical_response"]
    sec_sents.extend(cri_sen)

    rat_sen = sec["rating"]
    rat_str = ', '.join(rat_sen[:-1])
    rat_str = ' and ' + rat_sen[-1] + '.'
    rat_str = 'The movie ratings are ' + rat_str
    sec_sents.append(rat_str)

    return sec_sents

def process_wiki_sections(data):
    
    processed_wiki = {}
    for key in data.keys():
        wiki = data[key]
        processed_wiki[key] = {}
        for i in range(4):
            sec = wiki[str(i)]
            if i == 0:
                sentences = process_sec_0(sec)
            else:
                sentences = nltk.sent_tokenize(sec)
            sentences = [sent.lower() for sent in sentences]
            sec_string = " ".join(sentences)
            processed_wiki[key][i] = sec_string
    return processed_wiki

# take care of multiple utts by same user. concatenate them together.
def map_utt_doc(conv, wiki):
    
    document, utterances = [], []
    prev_uid = None
    for utt in conv:
        sec_idx = utt["docIdx"]
        sec = wiki[sec_idx]
        curr_text = utt["text"].strip('\n').strip('\t').lower()

        curr_uid = utt["uid"]
        if prev_uid == None:
            prev_uid = curr_uid 
            document.append(sec)
        elif curr_uid == prev_uid:
            prev_utt = utterances.pop(-1)
            curr_text = prev_utt + ' ' + curr_text           
        else:
            prev_uid = curr_uid 
            document.append(sec)
        utterances.append(curr_text)
    return document, utterances

def strip_newl_tab(string):

    split_string = string.split('\n')
    if len(split_string) > 1:
        string = " ".join(split_string)
    else:
        string = split_string[0]
    split_string = string.split('\t')
    if len(split_string) > 1:
        string = " ".join(split_string)
    else:
        string = split_string[0]

    return string

def create_instances(conv, doc, chat_history_len):

    source, documents, target = [], [], []
    for i in range(len(conv)-1):
        if i < chat_history_len:
            chat_history = conv[:i+1]
        else:
            chat_history = conv[i-chat_history_len+1:i+1]
        curr_doc = doc[i]

        if len(chat_history) > 1:
            chat_str = " ".join(chat_history)
        else:
            chat_str = chat_history[0]

        chat_str = strip_newl_tab(chat_str)
        source.append(chat_str)

        doc_str = strip_newl_tab(curr_doc)
        documents.append(doc_str)

        resp_str = conv[i+1]
        resp_str = strip_newl_tab(resp_str)
        target.append(resp_str)

    return source, target, documents

# for each utt, get section and save in chat_history, document and target form
def parse_conversations(
    filenames, 
    wiki_data, 
    chat_hist_len
    ):

    source_list, target_list, doc_list = [], [], []
    for fnam in filenames:
        data = read_json(fnam)
        wiki_idx = data["wikiDocumentIdx"]
        chat = data["history"]
        wiki_doc = wiki_data[wiki_idx]
        document, utterances = map_utt_doc(chat, wiki_doc)
        chat_hist, target, documents = create_instances(
                                                utterances, 
                                                document, 
                                                chat_hist_len
                                            )
        if len(chat_hist) != len(documents) or len(chat_hist) != len(target):
            print("Length of conversation is not same as length of documents")
        source_list.extend(chat_hist)
        target_list.extend(target)
        doc_list.extend(documents)

    return source_list, target_list, doc_list

def write_files(out_dir, all_data, split):
        
    with codecs.open(out_dir + '/' + split + '.tsv', "w", "utf-8") as out:
        spam = csv.writer(out, delimiter='\t')
        for i in range(len(all_data['source'])):
            spam.writerow([i, all_data['source'][i], all_data['target'][i], all_data['docs'][i]])

def main():
    parser = argparse.ArgumentParser(description='preprocess_cmu_dog.py')
    parser.add_argument('--data_dir', required=True,
                    help='Path to the raw file from preprocess.py')
    parser.add_argument('--out_dir', required=True,
                    help='Path to save file from preprocess.py')
    parser.add_argument('--chat_history_len', type=int, default=1,
                    help='Window of chat history to be considered')
    opt = parser.parse_args()

    # read all the wiki docs and hash them with wikiDocumentIdx
    wiki_data = parse_wiki(opt.data_dir  + 'WikiData/')
    # For each section, sentence tokenize, and save 
    proc_wiki_data = process_wiki_sections(wiki_data)

    # get conversation filenames
    split = ["train", "valid", "test"]
    dir_name = opt.data_dir + "Conversations/"
    filenames = {}
    for spl in split:
        for (_, _, fnams) in walk(dir_name + spl + '/'):
            fnams = [dir_name + spl + '/' + fnam for fnam in fnams]
            filenames[spl] = fnams

    expt_name = 'cmu_dog_histLen_' + str(opt.chat_history_len)
    out_path = os.path.join(opt.out_dir, expt_name)
    try:
        os.mkdir(out_path)
    except OSError as error:
        print(error)

    # process conv files
    for spl in split:
        source, target, docs = parse_conversations(
                                        filenames[spl], 
                                        proc_wiki_data, 
                                        opt.chat_history_len
                                        )
        all_data = {
            'source': source,
            'target': target,
            'docs': docs
        }
        if spl == 'valid':
            write_files(out_path, all_data, 'dev')
        else:
            write_files(out_path, all_data, spl)

if __name__ == "__main__":
    main()
