for g in blues classical country disco hiphop jazz metal pop reggae rock
do
    mkdir ./val/$g
    #echo "mkdir ./val/$g"
    for i in {60..79}
    do
	mv ./train/$g/$g.000$i.png ./val/$g/$g.000$i.png
	#echo "mv ./train/$g/$g.000$i.png ./val/$g/$g.000$i.png"
    done
done
