G4Hunter: High-Performance G-Quadruplex Prediction Tool
G4Hunter is a high-performance Python script that uses the G4Hunter algorithm to predict G-quadruplex (G4) structures in DNA sequences from a FASTA-formatted file. Unlike traditional tools, this version significantly improves efficiency for large genome files through GPU acceleration and multi-process parallel computing.

Key Features
GPU Acceleration: Utilizes the PyTorch library to offload core computational tasks to the GPU, providing a speed boost of several times.

Parallel Processing: Supports multi-process parallel computing to effectively leverage the performance of multi-core CPUs and multiple GPUs.

Double-Strand Analysis: Searches for G-quadruplexes on both the forward and reverse complement strands.

Customizable Parameters: Users can adjust the sliding window size and score threshold as needed.

Standard Output: Generates a standard BED file for easy integration with popular bioinformatics tools.

Dependencies
To run this script, you need to install the following Python libraries:

Biopython: For handling biological sequence files, such as FASTA format.

NumPy: A powerful library for numerical computation.

PyTorch: A machine learning framework for GPU acceleration and tensor computing.

You can easily install these dependencies using pip:

pip install biopython numpy torch

How to Use
Execute the script via the command line and use parameters to specify the input file, output directory, and other analysis settings.

Command-Line Example
Here is a specific example of how to run the script:

python G4hunter.py -i GRCh38.p14.genome.fa -o ./result -w 25 -s 1.5

Parameter Descriptions
-i or --input: Required. Specifies the path to the input FASTA file.

-o or --output: Required. Specifies the output directory.

-w or --window: Optional. Sets the sliding window size (default: 25).

-s or --score: Optional. Sets the score threshold (default: 1.5), and regions with scores above this value will be reported.

Output File Format
The script generates a standard BED6 format file, and the columns have the following meanings:

chrom: The name of the sequence, usually the chromosome ID.

chromStart: The starting position of the G4 region (0-based).

chromEnd: The ending position of the G4 region (1-based).

name: The name of the region (e.g., G4_fwd or G4_rev).

score: The G4Hunter score for the region.

strand: The sequence strand (+ or -).
