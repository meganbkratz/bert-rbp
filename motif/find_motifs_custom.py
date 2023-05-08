#### ::: DNABERT-viz find motifs ::: ####

import os
import pandas as pd
import numpy as np
import argparse
import sys
sys.path.append('./motif')
import motif_utils as utils
import Bio.motifs
from Bio.Seq import Seq
import pyqtgraph as pg
import weblogo
import scipy


def save_motifs(dir_name, merged_seqs, merged_dict=None, params=None, linkage_data=None):
    os.makedirs(dir_name, exist_ok=True)
    
    if merged_dict is not None:
        with open(os.path.join(dir_name, 'motif_dict.txt'), 'w') as f:
            for dict_item in merged_dict:
                f.write('{}: {}\n'.format(dict_item, merged_dict[dict_item]))

    if params is not None:
        params['max_merge_distance']=linkage_data.get('max_distance')
        with open(os.path.join(dir_name, 'motif_parameters.txt'), 'w') as f:
            for k, v in params.items():
                f.write('{}:{}\n'.format(k, v))
        
    for motif, instances in merged_seqs.items():
        # saving to files

        m = Bio.motifs.create([Seq(v.replace('U', 'T')) for v in instances['seqs']])
        key = m.degenerate_consensus.replace('T', 'U')
        with open(os.path.join(dir_name, 'motif_{:0=3}_{}.txt'.format(len(instances['seq_idx']), key)), 'w') as f:
            for seq in instances['seqs']:
                f.write(seq+'\n')

        # make weblogo
        seqs = weblogo.SeqList([weblogo.Seq(x) for x in instances['seqs']], alphabet=weblogo.Alphabet('ACGU'))
        logo_data = weblogo.LogoData.from_seqs(seqs)
        logo_options = weblogo.LogoOptions()
        logo_options.title = motif
        logo_options.color_scheme = weblogo.classic
        logo_format = weblogo.LogoFormat(logo_data, logo_options)
        eps = weblogo.eps_formatter(logo_data, logo_format)

        with open(os.path.join(dir_name, 'motif_{:0=3}_{}_weblogo.eps'.format(len(instances['seq_idx']), key)), 'wb') as f:
            f.write(eps)

    if linkage_data is not None:
        motif_list = [m.replace('T', 'U') for m in linkage_data['motif_list']]
        import matplotlib.pyplot as plt
        plt.figure()
        dn = scipy.cluster.hierarchy.dendrogram(linkage_data['linkage_matrix'], labels=motif_list, orientation='right', color_threshold=0.75)
        plt.savefig(os.path.join(dir_name, 'merge_tree.pdf'))


def convert_seqs_to_RNA(merged_motif_seqs):
    #import IPython
    #IPython.embed()

    converted = {}
    for k, v in merged_motif_seqs.items():
        k_new = k.replace('T', 'U')
        converted[k_new] = {
            'seq_idx': v['seq_idx'],
            'atten_region_pos': v['atten_region_pos'],
            }
        new_seqs = []
        for seq in v['seqs']:
            new_seqs.append(seq.replace('T', 'U'))
        converted[k_new]['seqs'] = new_seqs 

    return converted

def convert_dict_to_RNA(merged_motif_dict):
    #import IPython
    #IPython.embed()

    converted = {}
    for k, v in merged_motif_dict.items():
        k_new = k.replace('T', 'U')
        l = []
        for x in v:
            if type(x) == str:
                l.append(x.replace('T', 'U'))
            else:
                l.append(x)
        converted[k_new] = l

    return converted


if __name__ == '__main__':
    pg.dbg()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the sequence+label .tsv files (or other data files) for the task.",)
    parser.add_argument("--predict_dir", default=None, type=str, required=True, help="Path where the attention scores were saved. Should contain both pred_results.npy and atten.npy",)

    args = parser.parse_args()
    params = dict(
        window_size = 12,
        min_len = 5,
        max_len = 10,
        p_val_cutoff = 0.005,
        min_n_motif = 3,
        top_n_motif = 10,
        align_all_ties = True,
        verbose = True,
        return_idx = bool,
        kmer = 3,
        )


    atten_scores = np.load(os.path.join(args.predict_dir,"atten.npy"))
    dev = pd.read_csv(os.path.join(args.data_dir,"dev.tsv"),sep='\t')
    dev.columns = ['sequence','label']
    dev['seq'] = dev['sequence'].apply(utils.kmer2seq)
    dev_pos = dev[dev['label'] == 1]
    dev_neg = dev[dev['label'] == 0]
    pos_atten_scores = atten_scores[dev_pos.index.values]
    assert len(dev_pos) == len(pos_atten_scores)


    ## bert-rbp analysis - top 10 min_len_5
    params = dict(
        window_size = 12,
        min_len = 5,
        max_len = 10,
        p_val_cutoff = 0.005,
        min_n_motif = 3,
        top_n_motif = 10,
        align_all_ties = True,
        verbose = True,
        return_idx = False,
        kmer = 3,
        merge_method = 'UPGMA'
        )
    # run motif analysis
    merged_motif_seqs, merged_motif_dict, linkage_data = utils.motif_analysis(dev_pos['seq'], dev_neg['seq'], pos_atten_scores, save_file_dir=None, **params)
    merged_motif_seqs = convert_seqs_to_RNA(merged_motif_seqs)
    merged_motif_dict = convert_dict_to_RNA(merged_motif_dict)
    save_motifs(os.path.join(args.predict_dir, 'UPGMA_merged_motifs'), merged_motif_seqs, merged_dict=merged_motif_dict, params=params, linkage_data=linkage_data)

    # params['min_len'] = 6
    # merged_motif_seqs, merged_motif_dict = utils.motif_analysis(dev_pos['seq'], dev_neg['seq'], pos_atten_scores, save_file_dir=None, **params)
    # merged_motif_seqs = convert_seqs_to_RNA(merged_motif_seqs)
    # merged_motif_dict = convert_dict_to_RNA(merged_motif_dict)
    # save_motifs(os.path.join(args.predict_dir, 'bert-rbp_10_minlen_6'), merged_motif_seqs, merged_motif_dict, params)

    # params['min_len'] = 5
    # params['top_n_motif'] = 5
    # merged_motif_seqs, merged_motif_dict = utils.motif_analysis(dev_pos['seq'], dev_neg['seq'], pos_atten_scores, save_file_dir=None, **params)
    # merged_motif_seqs = convert_seqs_to_RNA(merged_motif_seqs)
    # merged_motif_dict = convert_dict_to_RNA(merged_motif_dict)
    # save_motifs(os.path.join(args.predict_dir, 'bert-rbp_5'), merged_motif_seqs, merged_motif_dict, params)

    # params['top_n_motif'] = 3
    # merged_motif_seqs, merged_motif_dict = utils.motif_analysis(dev_pos['seq'], dev_neg['seq'], pos_atten_scores, save_file_dir=None, **params)
    # merged_motif_seqs = convert_seqs_to_RNA(merged_motif_seqs)
    # merged_motif_dict = convert_dict_to_RNA(merged_motif_dict)    
    # save_motifs(os.path.join(args.predict_dir, 'bert-rbp_3'), merged_motif_seqs, merged_motif_dict, params)
