# Pixel matching stages

Currently, due to the large amount of data involved and the need to apply different processing styles to the data depending on, for example, if we're processing raster or tabular data, we've split pixel matching into the following distinct stages, where each stage flows into the next one:

1. calculate_k.py - Selects pixels from within the project area to be used in the matching process
2. find_potential_matches.py - For each pixel in K, generates a raster of potential counterfactual pixels
3. build_m_raster.py - Combines all the rasters from the previous stage into a single raster, which is M, the set of pixels from which counterfactuals can be selected.
4. build_m_table.py - Takes the raster for M, and generates a table with a row per pixel, adding in the data from JRC, CPC, etc. related to that pixel.
5. find_pairs.py - Uses the pixel set K and counterfactual pool M to provide sets of matches. We generate 100 sets of matches, that will then later be compared.

If calculating leakage rather than additionality you can just input the leakage zone rather than the project zone, and suitably adjusted matching areas. Otherwise the process is the same for each.