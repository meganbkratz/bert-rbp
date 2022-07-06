import re

def search_training_data(sequence_file, training_file):
    """Function for parts of an RNA sequence that are present in the training data."""
    matches = []
    with open(sequence_file, 'r') as f:
        for line in f.readlines():
            if line[0] == '>':
                p = re.compile(r'chr[0-9MXY]+:[0-9]+-[0-9]+')
                m = p.search(line)
                p2 = re.compile('\d+')
                target_chromosome, target_start, target_end = p2.findall(m.group())
                target_start = int(target_start)
                target_end = int(target_end)
                print('targets: ', target_chromosome, target_start, target_end)


    with open(training_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line[0] == '>':
            p = re.compile('[\dMXY]+')
            m = p.findall(line)
            chromosome, start, end = m
            if chromosome == target_chromosome:
                start = int(start)
                end = int(end)
                length = end-start
                if target_start-length <= start <= target_end:
                    matches.append((line, lines[i+1]))
    return matches

sequence_file = '/home/megan/work/lnc_rna/data/sequences/TARDBP/Tardbp-human_genomic.fasta'
pos_training_file = '/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TARDBP.positive.fa'
neg_training_file = '/home/megan/work/lnc_rna/code/bert-rbp/RBP_training_data/TARDBP.negative.fa'
pos_matches = search_training_data(sequence_file, pos_training_file)
neg_matches = search_training_data(sequence_file, neg_training_file)