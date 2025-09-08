# -*- coding: utf-8 -*-
# Amina BEDRAT
# G4Hunter
# Modified for Python 3 compatibility

from __future__ import division
import sys, re, os
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
from collections import defaultdict
import argparse

# This function takes as input a sequence and a window size
def G4Hunter(sequence, window):
    i = 0
    RunG = []
    # This loop calculates the number of consecutive G's
    while (i < len(sequence)):
        if sequence[i] == 'G':
            j = i
            c = 0
            while (j < len(sequence) and sequence[j] == 'G'):
                c += 1
                j += 1
            RunG.append(c)
            i = j
        else:
            RunG.append(0)
            i += 1

    RunC = []
    i = 0
    # This loop calculates the number of consecutive C's
    while (i < len(sequence)):
        if sequence[i] == 'C':
            j = i
            c = 0
            while (j < len(sequence) and sequence[j] == 'C'):
                c += 1
                j += 1
            RunC.append(c)
            i = j
        else:
            RunC.append(0)
            i += 1

    # This function calculates the score for a given window
    def score(sub):
        s = []
        for l in sub:
            if abs(l) == 1:
                s.append(1)
            elif abs(l) == 2:
                s.append(2)
            elif abs(l) == 3:
                s.append(3)
            elif abs(l) >= 4:
                s.append(4)
            else:
                s.append(0)
        if len(s) == 0:
            return 0
        return sum(s) / len(s)

    i = 0
    G4H_score = []
    # This loop calculates the G4Hunter score for each window
    while (i <= len(sequence) - window):
        sub_seq = sequence[i:i + window]
        RunG_w = []
        j = i
        # This loop calculates the number of consecutive G's in the window
        while j < (i + window):
            if sequence[j] == 'G':
                c = 0
                k = j
                while k < len(sequence) and sequence[k] == 'G':
                    c += 1
                    k += 1
                RunG_w.append(c)
                j = k
            else:
                RunG_w.append(0)
                j += 1

        RunC_w = []
        j = i
        # This loop calculates the number of consecutive C's in the window
        while j < (i + window):
            if sequence[j] == 'C':
                c = 0
                k = j
                while k < len(sequence) and sequence[k] == 'C':
                    c += 1
                    k += 1
                RunC_w.append(-c)
                j = k
            else:
                RunC_w.append(0)
                j += 1
        
        # Calculate the score for G and C runs
        sc_g = score([x for x in RunG_w if x != 0])
        sc_c = score([x for x in RunC_w if x != 0])
        
        # Final G4Hunter score for the window
        G4H_score.append(sc_g - sc_c)
        i += 1
    return G4H_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="G4Hunter: A tool for predicting G-quadruplexes.")
    parser.add_argument('-i', '--input', required=True, help="Input FASTA file")
    # Python 3 version: changed default output to a BED file as requested in prompt history
    parser.add_argument('-o', '--output', help="Output BED file name") 
    parser.add_argument('-w', '--window', type=int, default=25, help="Window size (default: 25)")
    parser.add_argument('-s', '--score', type=float, default=1.2, help="Score threshold (default: 1.2)")

    args = parser.parse_args()
    
    # Use the provided output file name, or create a default one if not provided.
    if args.output:
        result_file = args.output
    else:
        output_prefix = os.path.splitext(os.path.basename(args.input))[0]
        result_file = "{0}_G4Hunter_W{1}_S{2}.bed".format(output_prefix, args.window, args.score)

    with open(result_file, 'w') as f_out:
        print("Processing file: {}".format(args.input)) # Changed for Python 3
        # Removed header for BED format compatibility
        # f_out.write("Sequence_name\tStart\tEnd\tSequence\tLength\tG4Hunter_score\n")
        
        for record in SeqIO.parse(args.input, "fasta"):
            sequence_name = record.id
            sequence = str(record.seq).upper()
            
            # Forward strand
            scores_fwd = G4Hunter(sequence, args.window)
            
            i = 0
            while i < len(scores_fwd):
                if scores_fwd[i] >= args.score:
                    start_region = i
                    end_region = i
                    max_score_in_region = scores_fwd[i]
                    
                    j = i + 1
                    while j < len(scores_fwd) and scores_fwd[j] >= args.score:
                        if scores_fwd[j] > max_score_in_region:
                            max_score_in_region = scores_fwd[j]
                        end_region = j
                        j += 1
                        
                    start_pos = start_region # BED format is 0-based
                    end_pos = end_region + args.window
                    
                    # Writing in BED format: chrom, start, end, name, score, strand
                    f_out.write("{0}\t{1}\t{2}\tG4_fwd\t{3:.2f}\t+\n".format(
                        sequence_name, start_pos, end_pos, max_score_in_region
                    ))
                    i = end_region + 1
                else:
                    i += 1

            # Reverse complement strand
            rev_comp_seq = str(record.seq.reverse_complement()).upper()
            scores_rev = G4Hunter(rev_comp_seq, args.window)
            
            i = 0
            while i < len(scores_rev):
                if scores_rev[i] >= args.score:
                    start_region = i
                    end_region = i
                    max_score_in_region = scores_rev[i]

                    j = i + 1
                    while j < len(scores_rev) and scores_rev[j] >= args.score:
                        if scores_rev[j] > max_score_in_region:
                            max_score_in_region = scores_rev[j]
                        end_region = j
                        j += 1
                    
                    start_pos_rev = start_region
                    end_pos_rev = end_region + args.window
                    
                    orig_start = len(sequence) - end_pos_rev
                    orig_end = len(sequence) - start_pos_rev

                    f_out.write("{0}\t{1}\t{2}\tG4_rev\t{3:.2f}\t-\n".format(
                        sequence_name, orig_start, orig_end, max_score_in_region
                    ))
                    i = end_region + 1
                else:
                    i += 1

    print("Analysis complete. Results saved to: {}".format(result_file)) # Changed for Python 3
