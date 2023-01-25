


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

    if start is not None: ## make sure to add the last region if we end on it
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
            d['dna_indices'] = (probs['indices']['dna_indices'][k][site[0]], probs['indices']['dna_indices'][k][site[1]])
            d['rna_indices'] = (probs['indices']['rna_indices'][k][site[0]], probs['indices']['rna_indices'][k][site[1]])
            d['mean_probability'] = probs[k][site[0]:site[1]+1].mean()
            d['region_length'] = (site[1]+1)-site[0]
            regions.append(d)

        results[k] = regions

    return results


def export_regions():
    ### save 2 .csv files:
    #       1) one record per RBP per splice - fields: Splice, RBP, n_regions, model_type, threshold, n_contiguous
    #       2) one record per region (or site?) fields: Splice, RBP, model_type, threshold, n_contiguous, binding_probability (individual or mean), region or site coordinates, region length?, region id 
    pass
