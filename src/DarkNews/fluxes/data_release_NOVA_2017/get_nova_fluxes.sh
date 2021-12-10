#!/bin/sh

for FLAVOR in "nue" "nuebar" "numu" "numubar"
do
	wget 'https://nova-docdb.fnal.gov/cgi-bin/RetrieveFile?docid=25266&filename=FHC_Flux_'"${FLAVOR}"'_NOvA_ND_2017.txt&version=5' -O 'FHC_'"${FLAVOR}"'.dat' -v 0
	wget 'https://nova-docdb.fnal.gov/cgi-bin/RetrieveFile?docid=25266&filename=RHC_Flux_'"${FLAVOR}"'_NOvA_ND_2017.txt&version=5' -O 'RHC_'"${FLAVOR}"'.dat' -v 0
done	
