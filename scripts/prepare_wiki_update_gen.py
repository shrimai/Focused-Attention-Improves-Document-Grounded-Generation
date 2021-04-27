import csv
import ast
import codecs
import argparse
import pdb
import sys

csv.field_size_limit(sys.maxsize)

def read_file(data_dir, split):
    wiki, tgt, news = [], [], []
    with codecs.open(data_dir + split + '_info.csv', 'r', 'utf-8') as inp:
        reader_ = csv.reader(inp, delimiter=',')
        for row in reader_:
            wiki.append(row[4])
            tgt.append(row[5])
            news.append(row[10])
    wiki, tgt, news = wiki[1:], tgt[1:], news[1:]
    print(len(wiki), len(tgt), len(news))

    return wiki, tgt, news

def write_file(data_dir, split, wiki, tgt, news):

    with codecs.open(data_dir + split + '.tsv', 'w', 'utf-8') as out:
        writer_ = csv.writer(out, delimiter='\t')
        for i in range(len(tgt)):
            wiki_text = ast.literal_eval(wiki[i])
            wiki_text = ' '.join(wiki_text)
            news_text = news[i].split('\n')
            news_text = ' '.join(news_text)
            news_text = news_text.split('\t')
            news_text = ' '.join(news_text)
            news_text = news_text.split()
            news_text = ' '.join(news_text)
            writer_.writerow([i, wiki_text, tgt[i], news_text])

def main():
    parser = argparse.ArgumentParser(description='preprocess_wiki_update_gen.py')
    parser.add_argument('--data_dir', required=True,
                    help='Path to the raw file from preprocess.py')
    parser.add_argument('--out_dir', required=True,
                    help='Path to save file from preprocess.py')
    opt = parser.parse_args()

    wiki, tgt, news = read_file(opt.data_dir, 'train')
    write_file(opt.out_dir, 'train', wiki, tgt, news)

    wiki, tgt, news = read_file(opt.data_dir, 'valid')
    write_file(opt.out_dir, 'dev', wiki, tgt, news)

    wiki, tgt, news = read_file(opt.data_dir, 'test')
    write_file(opt.out_dir, 'test', wiki, tgt, news)

if __name__ == "__main__":
    main()
