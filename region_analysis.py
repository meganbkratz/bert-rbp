import os, re, argparse
from analyze_RNA import load_probabilities
import util
import config
from Bio import SeqIO
import numpy as np


def find_contiguous_sites(probs, threshold=0.95, n_contiguous=3):
    """Return a list of tuples [(start_index, end_index), ] where each entry is a site where there are >= n_contiguous segments <= threshold.
    probs: an array or list of binding probabilities
    """
    regions = []
    start = None

    for i, p in enumerate(probs):
        ## 4 conditions:
        #    - region not started and we're low - continue
        #    - region not started and we're high -> start region
        #    - region started and we're low -> end region
        #    - region started and we're high -> continue

        if start == None:
            if p < threshold:
                continue
            elif p >= threshold:
                start = i

        else:
            if p >= threshold:
                continue
            elif p < threshold:
                ## count the region if it's big enough, otherwise just move on
                if i - start >= n_contiguous:
                    regions.append((start, i-1))
                start = None

    if start is not None and (i-start >= n_contiguous): ## make sure to add the last region if we end on it
        regions.append((start, i))

    return regions


def find_binding_regions(probs, threshold=0.95, n_contiguous=3):

    results = {}
    for k in probs.keys():
        if k in ['indices', 'metainfo', 'attention']:
            continue
        regions = []
        sites = find_contiguous_sites(probs[k], threshold=threshold, n_contiguous=n_contiguous)
        for site in sites:
            d={}
            d['indices'] = site
            try:
                d['dna_indices'] = (probs['indices']['dna_indices'][k][site[0]], probs['indices']['dna_indices'][k][site[1]])
            except KeyError:
                d['dna_indices'] = None
            d['rna_indices'] = (probs['indices']['rna_indices'][k][site[0]], probs['indices']['rna_indices'][k][site[1]])
            d['mean_probability'] = probs[k][site[0]:site[1]+1].mean()
            d['region_length'] = (site[1]+1)-site[0]
            regions.append(d)

        results[k] = regions

    return results

def load_sequence(fasta_file):
    splices = [x for x in SeqIO.parse(fasta_file, 'fasta')]

    sequences = {}

    for splice in splices:
        dna = np.array([s for s in splice.seq])
        mask = np.char.isupper(dna)
        rna = dna[mask]
        indices = np.argwhere(mask)[:,0]

        chromosome, start, end = util.parse_dna_range(splice.description)
        sequences[splice.id] = {'sequence':''.join(list(rna)).replace('T', 'U'), 'dna_indices':indices+start}

    return sequences


def export_regions(model_dir, threshold=None, n_contiguous=3, save_prefix="binding_regions"):
    ### save 2 .csv files:
    #       1) one record per RBP per splice - fields: Splice, RBP, n_regions, model_type, threshold, n_contiguous
    #       2) one record per region (or site?) fields: Splice, RBP, model_type, threshold, n_contiguous, binding_probability (individual or mean), region or site coordinates, region length?, region id 
    
    print("Starting region analysis for {}".format(model_dir))
    summary_file = os.path.join(model_dir, save_prefix+'_summary.csv')
    region_file = os.path.join(model_dir, save_prefix+'.csv')

    with open(summary_file, 'w') as f:
        f.write("parent_directory:, %s, \n" % model_dir)
        f.write("RBP, Splice, n_regions, sum_of_region_lengths, longest_region_length, model_type, threshold, n_contiguous, \n")

    with open(region_file, 'w') as f:
        f.write("parent_directory:, %s, \n" % model_dir)
        f.write("RBP, Splice, model_type, threshold, n_contiguous, mean_binding_probability, dna_coordinates, rna_coordinates, region_length, prob_file_indices, sequence \n")

    if threshold is None:
        use_dynamic_thresholding = True
    else:
        use_dynamic_thresholding = False

    model_type = os.path.split(model_dir)[1]

    for prob_file in sorted(os.listdir(model_dir)):
        if os.path.splitext(prob_file)[1] == '.pk':
            probs = load_probabilities(os.path.join(model_dir, prob_file), quiet=True)
            rbp = parseRBP(probs, filename=prob_file)

            sequence_file = probs.get('metainfo', {}).get('sequence_path')
            if sequence_file is None:
                sequence_file = util.find_fasta_file(os.path.join(model_dir, os.listdir(model_dir)[0]))
            try:
                sequences = load_sequence(sequence_file)
            except FileNotFoundError:
                sequence_file = util.find_fasta_file(os.path.join(model_dir, os.listdir(model_dir)[0]))
                sequences = load_sequence(sequence_file)

            if use_dynamic_thresholding:
                stats_file = os.path.join(config.rbp_performance_dir, model_type, rbp+'_eval_performance.csv')
                rbp_stats = util.load_performance_data(stats_file)
                lowest = util.get_threshold(rbp_stats, min_precision=0.9, min_recall=0.1, mode='lowest')
                if lowest is None:
                    continue
                else:
                    threshold = util.get_threshold(rbp_stats, min_precision=0.9, min_recall=0.1, mode='high_f0.2')

            regions = find_binding_regions(probs, threshold=threshold, n_contiguous=n_contiguous)
            with open(summary_file, 'a') as f:
                for splice in sorted(regions.keys()):
                    f.write("{rbp}, {splice}, {n_regions}, {total}, {max_length}, {model_type}, {threshold}, {n_contiguous}, \n".format(
                        rbp=rbp, 
                        splice=splice, 
                        n_regions=len(regions[splice]),
                        total=sum([r['region_length'] for r in regions[splice]]),
                        max_length=max([r['region_length'] for r in regions[splice]]) if len(regions[splice]) > 0 else 0,
                        model_type=model_type, 
                        threshold=threshold, 
                        n_contiguous=n_contiguous))
            with open(region_file, 'a') as f:
                for splice in sorted(regions.keys()):
                    if splice[-10:] == '_unspliced':  # don't worry about finding regions in introns for now
                        continue

                    chrom = probs['indices']['metainfo'][splice]['chromosome']
                    start = probs['indices']['metainfo'][splice]['range_start']
                    for r in regions[splice]:
                        f.write("{rbp}, {splice}, {model_type}, {threshold}, {n_contiguous}, {probability}, {dna_coordinates}, {rna_coordinates}, {region_len}, {prob_file_indices}, {sequence} \n".format(
                            rbp=rbp,
                            splice=splice,
                            model_type=os.path.split(model_dir)[1],
                            threshold=threshold,
                            n_contiguous=n_contiguous,
                            probability=r['mean_probability'],
                            dna_coordinates="chr{chrom}:{start}-{end}".format(
                                chrom=chrom,
                                start=r['dna_indices'][0]-50+start,
                                end=r['dna_indices'][1]+50+start),
                            rna_coordinates="({} {})".format(r['rna_indices'][0]-50, r['rna_indices'][1]+50),
                            prob_file_indices="({} {})".format(r['indices'][0], r['indices'][1]),
                            region_len=r['region_length'],
                            sequence=''.join(sequences[splice]['sequence'][r['rna_indices'][0]-50:r['rna_indices'][1]+50])
                            ))
    print("     Finished exporting binding region info for %s. (saved as: %s & %s)" % (model_dir, os.path.split(summary_file)[1], os.path.split(region_file)[1]))

def parseRBP(probs, filename=None):
        rbp = probs.get('metainfo', {}).get('rbp_name')
        if rbp is not None:
            return rbp

        if filename is not None:
            pattern = re.compile('_[A-Z0-9]*_')
            names = pattern.findall(filename)
            if len(names) == 1:
                return names[0].strip('_')
            else:
                print("Could not parse RBP name from filename: {}. Found {} matches: {}".format(filename, len(names), names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parent_directory", type=str, help="The path to the directory where the model_output.pk files are stored.")
    parser.add_argument("--threshold", type=str, required=False)
    parser.add_argument("--n_contiguous", type=int, default=3)

    args = parser.parse_args()

    if args.threshold is None:
        threshold = None
    else:
        threshold = float(args.threshold)

    save_prefix='binding_regions_%iN'%args.n_contiguous
    export_regions(args.parent_directory, threshold=args.threshold, n_contiguous=args.n_contiguous, save_prefix=save_prefix)
