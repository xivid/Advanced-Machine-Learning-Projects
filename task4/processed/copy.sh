mkdir -p 0
mkdir -p 1
lines=`tail -n +2 train_target.csv | sed 's/,/ /g'`
echo "$lines" | while read id y
do
	if [[ $y == 0 ]]; then
		echo $id $y "good"
		cp train/${id}.avi 0/
	else
		echo $id $y "bad"
		cp train/${id}.avi 1/
	fi
done
