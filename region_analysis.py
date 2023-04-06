import os, re, argparse
from analyze_RNA import load_probabilities


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
        if k in ['indices', 'metainfo']:
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


def export_regions(model_dir, threshold=0.95, n_contiguous=3, save_prefix="binding_regions"):
    ### save 2 .csv files:
    #       1) one record per RBP per splice - fields: Splice, RBP, n_regions, model_type, threshold, n_contiguous
    #       2) one record per region (or site?) fields: Splice, RBP, model_type, threshold, n_contiguous, binding_probability (individual or mean), region or site coordinates, region length?, region id 
    summary_file = os.path.join(model_dir, save_prefix+'_summary.csv')
    region_file = os.path.join(model_dir, save_prefix+'.csv')

    with open(summary_file, 'w') as f:
        f.write("parent_directory:, %s, \n" % model_dir)
        f.write("RBP, Splice, n_regions, model_type, threshold, n_contiguous, \n")

    with open(region_file, 'w') as f:
        f.write("parent_directory:, %s, \n" % model_dir)
        f.write("RBP, Splice, model_type, threshold, n_contiguous, mean_binding_probability, dna_coordinates, region_length, prob_file_indices \n")


    for prob_file in sorted(os.listdir(model_dir)):
        if os.path.splitext(prob_file)[1] == '.pk':
            probs = load_probabilities(os.path.join(model_dir, prob_file), quiet=True)
            regions = find_binding_regions(probs, threshold=threshold, n_contiguous=n_contiguous)
            rbp = parseRBP(probs, filename=prob_file)
            with open(summary_file, 'a') as f:
                for splice in sorted(regions.keys()):
                    f.write("{rbp}, {splice}, {n_regions}, {model_type}, {threshold}, {n_contiguous}, \n".format(
                        rbp=rbp, 
                        splice=splice, 
                        n_regions=len(regions[splice]), 
                        model_type=os.path.split(model_dir)[1], 
                        threshold=threshold, 
                        n_contiguous=n_contiguous))
            with open(region_file, 'a') as f:
                for splice in sorted(regions.keys()):
                    chrom = probs['indices']['metainfo'][splice]['chromosome']
                    start = probs['indices']['metainfo'][splice]['range_start']
                    for r in regions[splice]:
                        f.write("{rbp}, {splice}, {model_type}, {threshold}, {n_contiguous}, {probability}, {coordinates}, {region_len}, {prob_file_indices}, \n".format(
                            rbp=rbp,
                            splice=splice,
                            model_type=os.path.split(model_dir)[1],
                            threshold=threshold,
                            n_contiguous=n_contiguous,
                            probability=r['mean_probability'],
                            coordinates="chr{chrom}:{start}-{end}".format(
                                chrom=chrom,
                                start=r['dna_indices'][0]+start,
                                end=r['dna_indices'][1]+start),
                            prob_file_indices=r['indices'],
                            region_len=r['region_length']
                            ))
    print("Finished exporting binding region info for %s. (saved as: %s & %s)" % (model_dir, os.path.split(summary_file)[1], os.path.split(region_file)[1]))

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
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--n_contiguous", type=int, default=3)

    args = parser.parse_args()

    export_regions(args.parent_directory, threshold=args.threshold, n_contiguous=args.n_contiguous, save_prefix='binding_regions_%iN'%args.n_contiguous)
