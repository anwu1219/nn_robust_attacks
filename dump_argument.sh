for network in mnist2x256_cw mnist4x256_cw mnist6x256_cw
do
    for target in 0 1 2 3 4 5 6 7 8 9
    do
	for ind in 0 1 2 3 4
	do
	    echo $network $target $ind
	done
    done
done

