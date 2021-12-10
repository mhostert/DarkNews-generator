#!/bin/sh

for FLAVOR in "nue" "nuebar" "numu" "numubar"
do
	wget --user-agent="Mozilla" 'https://arxiv.org/src/1607.00704v2/anc/'"${FLAVOR}"'FHC.txt' -O 'Nu_Flux_wandcsplinefix_'"${FLAVOR}"'FHC.txt'
	wget --user-agent="Mozilla" 'https://arxiv.org/src/1607.00704v2/anc/'"${FLAVOR}"'FHC.txt' -O "${FLAVOR}"'FHC.txt'
	wget --user-agent="Mozilla" 'https://arxiv.org/src/1607.00704v2/anc/'"${FLAVOR}"'RHC.txt' -O 'Nu_Flux_wandcsplinefix_'"${FLAVOR}"'RHC.txt'
	wget --user-agent="Mozilla" 'https://arxiv.org/src/1607.00704v2/anc/'"${FLAVOR}"'RHC.txt' -O "${FLAVOR}"'RHC.txt'
done	
