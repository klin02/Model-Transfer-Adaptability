if [ -z $1 ];then
    cat $(ls -1 ret/ret-*.out | sort -n -t - -k 2 | tail -n 1)
else
    cat ret/ret-"$1".out
fi